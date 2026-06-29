"""
Optimized attention-backward kernel:
- Both matmuls use group-reshape [bs*8, 10*sq, ...] — no GQA expansion anywhere.
- dP and dV bmms launched concurrently on separate CUDA streams.
- Module-level cached stream to avoid creation overhead in hot path.
- Pre-allocated output tensors before any stream switching.
- dV: direct attn.T @ dO -> [bs*8, skv, 128] (no post-transpose copy).
- Softmax-backward done with pure PyTorch elementwise ops (no Triton JIT overhead).
- All in bfloat16.

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

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128

# Module-level cached CUDA stream and event (created once, reused every call)
_side_stream = None
_dO_ready_event = None

def _get_side_stream(device):
    global _side_stream, _dO_ready_event
    if _side_stream is None:
        _side_stream = torch.cuda.Stream(device)
        _dO_ready_event = torch.cuda.Event()
    return _side_stream, _dO_ready_event


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
    # Step 1: Make dO contiguous in [bs, 80, sq, 128] layout (bfloat16).
    # =========================================================================
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()
    # dO: [bs, 80, sq, 128], bfloat16, contiguous

    # Shared group-reshape for both matmuls: [bs*8, 10*sq, 128] — free view
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # Prepare matmul operands (all free views, no copies)
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # =========================================================================
    # Pre-allocate output tensors on the CURRENT stream before any switching.
    # =========================================================================
    # dP output: [bs*8, 10*sq, skv]
    dP_groups = torch.empty(
        (bs * n_kv_heads, n_groups * seq_q, seq_kv),
        dtype=torch.bfloat16, device=device
    )
    # dV output: [bs*8, skv, 128] — direct final layout, no post-transpose needed.
    dV_flat = torch.empty(
        (bs * n_kv_heads, seq_kv, HEAD_DIM),
        dtype=torch.bfloat16, device=device
    )

    # =========================================================================
    # Step 2: Concurrent stream execution.
    # Both matmuls read from dO_groups_flat (concurrent reads are safe).
    # - Side stream: dV bmm (attn.T @ dO → directly contiguous [bs*8, skv, 128])
    # - Main stream: dP bmm → PyTorch softmax backward
    # =========================================================================
    main_stream = torch.cuda.current_stream(device)
    side_stream, dO_ready_event = _get_side_stream(device)

    # Record event: dO is ready on the main stream
    dO_ready_event.record(main_stream)

    # Side stream waits for dO to be ready, then launches dV
    side_stream.wait_event(dO_ready_event)
    with torch.cuda.stream(side_stream):
        # dV: bmm([bs*8, skv, 10*sq], [bs*8, 10*sq, 128]) -> [bs*8, skv, 128]
        torch.bmm(attn_groups_flat.transpose(-2, -1), dO_groups_flat, out=dV_flat)

    # Launch dP on main stream (concurrent with dV on side stream)
    # dP: bmm([bs*8, 10*sq, 128], [bs*8, 128, skv]) -> [bs*8, 10*sq, skv]
    torch.bmm(dO_groups_flat, vs_flat.transpose(-2, -1), out=dP_groups)

    # =========================================================================
    # Step 3: Softmax backward + dropout correction via pure PyTorch ops.
    # Runs on main stream — overlaps with dV on side stream.
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # dP_groups is [bs*8, 10*sq, skv]; reshape to [bs, 80, sq, skv]
    dP_dropped = dP_groups.reshape(bs, n_heads, seq_q, seq_kv).float()
    P = attn_weights.float()

    # Apply dropout mask and scale
    dp = torch.where(dropout_mask, dP_dropped * scale, torch.zeros_like(dP_dropped))

    # Softmax backward: dS = P * (dp - sum(dp * P, dim=-1, keepdim=True))
    dp_times_p = dp * P
    row_sum = dp_times_p.sum(dim=-1, keepdim=True)
    dS = P * (dp - row_sum)

    dS = dS.to(torch.bfloat16)

    # Wait for side stream (dV) to complete — dV_flat is already in final layout
    main_stream.wait_stream(side_stream)

    # dV_flat is already [bs*8, skv, 128] contiguous — just reshape
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV
