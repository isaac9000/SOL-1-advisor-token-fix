"""
Optimized attention-backward kernel — three-stage pipeline with CUDA stream overlap.

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
import math

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10


@triton.jit
def _softmax_bwd_dropout_kernel(
    # Inputs
    dP_ptr,    # [bs, n_heads, seq_q, seq_kv]  float32
    P_ptr,     # [bs, n_heads, seq_q, seq_kv]  bfloat16
    mask_ptr,  # [bs, n_heads, seq_q, seq_kv]  bool
    # Output
    dS_ptr,    # [bs, n_heads, seq_q, seq_kv]  bfloat16
    # Scalars
    scale: tl.constexpr,   # 1/(1-dropout)
    # Dimensions
    seq_kv: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """
    Each program handles one row (bs_idx, head_idx, q_idx).
    Computes: dS = P * (dP_dropped - sum(dP_dropped * P))
    where dP_dropped = dP * mask * scale
    """
    row_idx = tl.program_id(0)

    row_base = row_idx * seq_kv
    offs = tl.arange(0, BLOCK_KV)

    # Compute row sum: sum_kv(dP * mask * scale * P)
    row_sum = tl.zeros([1], dtype=tl.float32)

    for start in tl.range(0, seq_kv, BLOCK_KV):
        kv_offs = start + offs
        kv_mask = kv_offs < seq_kv

        dP_val = tl.load(dP_ptr + row_base + kv_offs,
                         mask=kv_mask, other=0.0)  # float32
        mk_val = tl.load(mask_ptr + row_base + kv_offs,
                         mask=kv_mask, other=0).to(tl.float32)
        P_val  = tl.load(P_ptr + row_base + kv_offs,
                         mask=kv_mask, other=0.0).to(tl.float32)

        dPd_val = dP_val * mk_val * scale
        row_sum += tl.sum(dPd_val * P_val, axis=0)

    # Second pass: write dS
    for start in tl.range(0, seq_kv, BLOCK_KV):
        kv_offs = start + offs
        kv_mask = kv_offs < seq_kv

        dP_val = tl.load(dP_ptr + row_base + kv_offs,
                         mask=kv_mask, other=0.0)  # float32
        mk_val = tl.load(mask_ptr + row_base + kv_offs,
                         mask=kv_mask, other=0).to(tl.float32)
        P_val  = tl.load(P_ptr + row_base + kv_offs,
                         mask=kv_mask, other=0.0).to(tl.float32)

        dPd_val = dP_val * mk_val * scale
        dS_val  = P_val * (dPd_val - row_sum)

        tl.store(dS_ptr + row_base + kv_offs,
                 dS_val.to(tl.bfloat16),
                 mask=kv_mask)


# Pre-create persistent streams for overlap
_stream1 = None
_stream2 = None


def _get_streams(device):
    global _stream1, _stream2
    if _stream1 is None:
        _stream1 = torch.cuda.Stream(device=device)
        _stream2 = torch.cuda.Stream(device=device)
    return _stream1, _stream2


def custom_kernel(data):
    (dO_in, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = dO_in.shape[0]
    seq_q  = dO_in.shape[1]
    seq_kv = value_states.shape[2]
    n_kv   = NUM_KEY_VALUE_HEADS   # 8
    n_g    = N_GROUPS              # 10
    d      = HEAD_DIM              # 128
    n_heads = NUM_ATTENTION_HEADS  # 80

    device = dO_in.device
    default_stream = torch.cuda.current_stream(device)

    # ── Shared setup on default stream ───────────────────────────────────────
    # [bs, sq, 80, d] -> [bs, 80, sq, d] contiguous bfloat16
    dO = dO_in.transpose(1, 2).contiguous()   # [bs, 80, sq, d] bfloat16

    # GQA expand for value states: [bs,8,skv,d] -> [bs,80,skv,d]
    vs_exp = value_states[:, :, None, :, :].expand(
        bs, n_kv, n_g, seq_kv, d
    ).reshape(bs, n_heads, seq_kv, d).contiguous()  # [bs,80,skv,d] bfloat16

    # Keep Pd as [bs, 80, seq_q, seq_kv] for dV BMM
    Pd_flat = attn_weights_dropped.contiguous()  # [bs,80,seq_q,seq_kv] bfloat16

    # Allocate outputs
    dS = torch.empty(bs, n_heads, seq_q, seq_kv,
                     dtype=torch.bfloat16, device=device)
    # dV uses float32 accumulator, we'll cast at end
    dV_flat = torch.empty(bs, n_heads, seq_kv, d,
                          dtype=torch.bfloat16, device=device)

    scale = 1.0 / (1.0 - attention_dropout)

    # Get or create streams
    stream1, stream2 = _get_streams(device)

    # ── Stream 1: dP BMM → softmax-bwd Triton ────────────────────────────────
    # Stream 2: dV BMM (independent) ─────────────────────────────────────────
    # Both streams wait for the default stream's work to complete
    stream1.wait_stream(default_stream)
    stream2.wait_stream(default_stream)

    # --- Stream 1: compute dP then dS ---
    with torch.cuda.stream(stream1):
        # dP = dO @ vs_exp^T : [bs,80,sq,d] @ [bs,80,d,skv] -> [bs,80,sq,skv]
        dP_flat = torch.bmm(
            dO.view(bs * n_heads, seq_q, d).float(),
            vs_exp.view(bs * n_heads, seq_kv, d).float().transpose(-2, -1)
        ).view(bs, n_heads, seq_q, seq_kv)  # float32

        # Softmax-bwd + dropout via Triton kernel (single-pass)
        n_rows = bs * n_heads * seq_q
        BLOCK_KV = max(triton.next_power_of_2(seq_kv), 64)
        BLOCK_KV = min(BLOCK_KV, 1024)

        grid = (n_rows,)
        _softmax_bwd_dropout_kernel[grid](
            dP_flat,
            attn_weights,
            dropout_mask,
            dS,
            scale=scale,
            seq_kv=seq_kv,
            BLOCK_KV=BLOCK_KV,
        )

    # --- Stream 2: compute dV ---
    with torch.cuda.stream(stream2):
        # dV_exp = Pd^T @ dO : [bs,80,skv,sq] @ [bs,80,sq,d] -> [bs,80,skv,d]
        dV_exp = torch.bmm(
            Pd_flat.view(bs * n_heads, seq_q, seq_kv).float().transpose(-2, -1),
            dO.view(bs * n_heads, seq_q, d).float()
        ).view(bs, n_heads, seq_kv, d)  # [bs,80,skv,d] float32

        # GQA reduction: [bs,80,skv,d] -> [bs,8,skv,d]
        dV_reduced = dV_exp.view(bs, n_kv, n_g, seq_kv, d).sum(dim=2)
        dV_flat_f32 = dV_reduced  # [bs,8,skv,d] float32

    # Synchronize both streams back to default stream
    default_stream.wait_stream(stream1)
    default_stream.wait_stream(stream2)

    # Convert dV to bfloat16 on default stream
    dV_out = dV_flat_f32.to(torch.bfloat16)

    return dS, dV_out
