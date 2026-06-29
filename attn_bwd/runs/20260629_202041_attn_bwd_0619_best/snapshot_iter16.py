"""
Optimized attention-backward kernel:
- Avoids the expensive dO.permute(0,2,1,3).contiguous() [bs,80,sq,128] flat copy.
- Instead, reshapes dO to [bs, sq, 8, 10, 128] and does ONE permute to
  [bs, 8, 10, sq, 128] contiguous — same size but structured for 5D matmuls.
- dP: dO_perm5 [bs, 8, 10, sq, 128] @ vs_T [bs, 8, 1, 128, skv] -> [bs, 8, 10, sq, skv]
- dV: attn_5d^T [bs, 8, 10, skv, sq] @ dO_perm5 [bs, 8, 10, sq, 128]
  -> [bs, 8, 10, skv, 128] -> sum dim=2 -> [bs, 8, skv, 128]
- Softmax backward via torch.compile(mode="reduce-overhead") for kernel fusion.
- Two-stream concurrency: dP and dV computed on separate CUDA streams.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128

# ---------------------------------------------------------------------------
# Softmax-backward as a torch.compile-fused function
# ---------------------------------------------------------------------------

def _softmax_bwd_elementwise(dP_dropped, P, dropout_mask, scale):
    """Pure PyTorch softmax backward — compiled for kernel fusion."""
    # dP_dropped: [bs, 80, sq, skv] float32
    # P:          [bs, 80, sq, skv] float32
    # dropout_mask: [bs, 80, sq, skv] bool
    dp = torch.where(dropout_mask, dP_dropped * scale, torch.zeros_like(dP_dropped))
    row_sum = (dp * P).sum(dim=-1, keepdim=True)
    ds = P * (dp - row_sum)
    return ds.to(torch.bfloat16)


# Compile once at module load with reduce-overhead mode (avoids JIT warmup cost)
_compiled_softmax_bwd = torch.compile(
    _softmax_bwd_elementwise,
    mode="reduce-overhead",
    fullgraph=True,
)

# Side CUDA stream for dV concurrency
_side_stream = None


def _get_side_stream(device):
    global _side_stream
    if _side_stream is None:
        _side_stream = torch.cuda.Stream(device=device)
    return _side_stream


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    device = grad_attn_output.device

    # =========================================================================
    # Step 1: ONE contiguous copy of dO, in [bs, 8, 10, sq, 128] layout.
    # grad_attn_output: [bs, sq, 80, 128] -> view as [bs, sq, 8, 10, 128]
    # -> permute(0,2,3,1,4) -> [bs, 8, 10, sq, 128] contiguous
    # =========================================================================
    dO_gqa_raw = grad_attn_output.reshape(bs, seq_q, n_kv_heads, n_groups, HEAD_DIM)
    dO_perm5 = dO_gqa_raw.permute(0, 2, 3, 1, 4).contiguous()
    # dO_perm5: [bs, 8, 10, sq, 128], bfloat16, contiguous

    main_stream = torch.cuda.current_stream(device)
    side_stream = _get_side_stream(device)

    # =========================================================================
    # Step 2 (main stream): dP = dO @ V^T
    # dO_perm5: [bs, 8, 10, sq, 128]
    # vs_T:     [bs, 8, 1, 128, skv]  (broadcast over groups)
    # Result:   [bs, 8, 10, sq, skv]
    # =========================================================================
    vs_T_bc = value_states.transpose(-2, -1).unsqueeze(2)  # [bs, 8, 1, 128, skv]
    dP_5d = torch.matmul(dO_perm5, vs_T_bc)  # [bs, 8, 10, sq, skv]

    # =========================================================================
    # Step 3 (side stream, concurrent with softmax bwd): dV = attn_dropped^T @ dO
    # attn_5d:  [bs, 8, 10, sq, skv]
    # dO_perm5: [bs, 8, 10, sq, 128]
    # Result:   [bs, 8, 10, skv, 128] -> sum dim=2 -> [bs, 8, skv, 128]
    # =========================================================================
    # dO_perm5 is ready on main stream; record event so side stream can use it
    event_dO_ready = torch.cuda.Event()
    event_dO_ready.record(main_stream)

    with torch.cuda.stream(side_stream):
        side_stream.wait_event(event_dO_ready)
        attn_5d = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups, seq_q, seq_kv)
        dV_5d = torch.matmul(attn_5d.transpose(-2, -1), dO_perm5)  # [bs, 8, 10, skv, 128]
        dV_flat = dV_5d.sum(dim=2).to(torch.bfloat16)               # [bs, 8, skv, 128]
        event_dV_done = torch.cuda.Event()
        event_dV_done.record(side_stream)

    # =========================================================================
    # Step 4 (main stream): Fused softmax backward via torch.compile
    # dP_5d: [bs, 8, 10, sq, skv] -> reshape to [bs, 80, sq, skv]
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    dP_flat = dP_5d.reshape(bs, n_heads, seq_q, seq_kv)

    # Cast to float32 for softmax backward precision
    dP_f32 = dP_flat.float()
    P_f32  = attn_weights.float()

    dS = _compiled_softmax_bwd(dP_f32, P_f32, dropout_mask, scale)
    # dS: [bs, 80, sq, skv] bfloat16

    # Wait for dV to finish on main stream
    main_stream.wait_event(event_dV_done)

    return dS, dV_flat
