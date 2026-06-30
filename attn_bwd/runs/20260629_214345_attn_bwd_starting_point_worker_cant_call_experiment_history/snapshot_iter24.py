"""
Optimized attention-backward kernel — cuBLAS BMMs with stream parallelism.

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
def _softmax_bwd_dropout_kernel_singlepass(
    dP_dropped_ptr,   # [N_rows, seq_kv]  bfloat16
    P_ptr,            # [N_rows, seq_kv]  bfloat16
    mask_ptr,         # [N_rows, seq_kv]  bool
    dS_ptr,           # [N_rows, seq_kv]  bfloat16  output
    seq_kv: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * seq_kv

    offsets = tl.arange(0, BLOCK_SIZE)
    mask_cond = offsets < seq_kv

    dp_dropped = tl.load(dP_dropped_ptr + row_start + offsets,
                         mask=mask_cond, other=0.0).to(tl.float32)
    p_val = tl.load(P_ptr + row_start + offsets,
                    mask=mask_cond, other=0.0).to(tl.float32)
    m_val = tl.load(mask_ptr + row_start + offsets,
                    mask=mask_cond, other=0).to(tl.float32)

    dp = dp_dropped * m_val * scale
    row_sum = tl.sum(dp * p_val, axis=0)
    ds = p_val * (dp - row_sum)

    tl.store(dS_ptr + row_start + offsets,
             ds.to(tl.bfloat16),
             mask=mask_cond)


@triton.jit
def _softmax_bwd_partial_sum_kernel(
    dP_dropped_ptr,   # [N_rows, seq_kv]  bfloat16
    P_ptr,            # [N_rows, seq_kv]  bfloat16
    mask_ptr,         # [N_rows, seq_kv]  bool
    row_sum_ptr,      # [N_rows, n_blocks]  float32  output partial sums
    seq_kv: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """First pass: compute partial row sums in parallel across seq_kv blocks."""
    row_idx   = tl.program_id(0)
    block_idx = tl.program_id(1)

    row_start   = row_idx * seq_kv
    block_start = block_idx * BLOCK_SIZE

    offsets   = block_start + tl.arange(0, BLOCK_SIZE)
    mask_cond = offsets < seq_kv

    dp_dropped = tl.load(dP_dropped_ptr + row_start + offsets,
                         mask=mask_cond, other=0.0).to(tl.float32)
    p_val = tl.load(P_ptr + row_start + offsets,
                    mask=mask_cond, other=0.0).to(tl.float32)
    m_val = tl.load(mask_ptr + row_start + offsets,
                    mask=mask_cond, other=0).to(tl.float32)

    dp = dp_dropped * m_val * scale
    partial = tl.sum(dp * p_val, axis=0)

    # Store partial sum: row_sum[row_idx, block_idx]
    n_blocks = tl.cdiv(seq_kv, BLOCK_SIZE)
    tl.store(row_sum_ptr + row_idx * n_blocks + block_idx, partial)


@triton.jit
def _softmax_bwd_apply_kernel(
    dP_dropped_ptr,   # [N_rows, seq_kv]  bfloat16
    P_ptr,            # [N_rows, seq_kv]  bfloat16
    mask_ptr,         # [N_rows, seq_kv]  bool
    row_sum_ptr,      # [N_rows, n_blocks]  float32  partial sums
    dS_ptr,           # [N_rows, seq_kv]  bfloat16  output
    seq_kv: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    n_blocks: tl.constexpr,
):
    """Second pass: reduce partial sums and compute dS in parallel across seq_kv blocks."""
    row_idx   = tl.program_id(0)
    block_idx = tl.program_id(1)

    # Reduce partial sums for this row
    sum_offsets = tl.arange(0, n_blocks)
    partial_sums = tl.load(row_sum_ptr + row_idx * n_blocks + sum_offsets).to(tl.float32)
    row_sum = tl.sum(partial_sums, axis=0)

    row_start   = row_idx * seq_kv
    block_start = block_idx * BLOCK_SIZE

    offsets   = block_start + tl.arange(0, BLOCK_SIZE)
    mask_cond = offsets < seq_kv

    dp_dropped = tl.load(dP_dropped_ptr + row_start + offsets,
                         mask=mask_cond, other=0.0).to(tl.float32)
    p_val = tl.load(P_ptr + row_start + offsets,
                    mask=mask_cond, other=0.0).to(tl.float32)
    m_val = tl.load(mask_ptr + row_start + offsets,
                    mask=mask_cond, other=0).to(tl.float32)

    dp = dp_dropped * m_val * scale
    ds = p_val * (dp - row_sum)

    tl.store(dS_ptr + row_start + offsets,
             ds.to(tl.bfloat16),
             mask=mask_cond)


@triton.jit
def _softmax_bwd_dropout_kernel_multiblock(
    dP_dropped_ptr,   # [N_rows, seq_kv]  bfloat16
    P_ptr,            # [N_rows, seq_kv]  bfloat16
    mask_ptr,         # [N_rows, seq_kv]  bool
    dS_ptr,           # [N_rows, seq_kv]  bfloat16  output
    seq_kv: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * seq_kv

    acc = tl.zeros([1], dtype=tl.float32)
    for block_start in tl.range(0, seq_kv, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask_cond = offsets < seq_kv

        dp_dropped = tl.load(dP_dropped_ptr + row_start + offsets,
                             mask=mask_cond, other=0.0).to(tl.float32)
        p_val = tl.load(P_ptr + row_start + offsets,
                        mask=mask_cond, other=0.0).to(tl.float32)
        m_val = tl.load(mask_ptr + row_start + offsets,
                        mask=mask_cond, other=0).to(tl.float32)

        dp = dp_dropped * m_val * scale
        acc += tl.sum(dp * p_val, axis=0)

    row_sum = tl.sum(acc, axis=0)

    for block_start in tl.range(0, seq_kv, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask_cond = offsets < seq_kv

        dp_dropped = tl.load(dP_dropped_ptr + row_start + offsets,
                             mask=mask_cond, other=0.0).to(tl.float32)
        p_val = tl.load(P_ptr + row_start + offsets,
                        mask=mask_cond, other=0.0).to(tl.float32)
        m_val = tl.load(mask_ptr + row_start + offsets,
                        mask=mask_cond, other=0).to(tl.float32)

        dp = dp_dropped * m_val * scale
        ds = p_val * (dp - row_sum)

        tl.store(dS_ptr + row_start + offsets,
                 ds.to(tl.bfloat16),
                 mask=mask_cond)


def fused_softmax_bwd_dropout(dP_dropped, P, mask, scale, seq_kv):
    N_rows = dP_dropped.shape[0]
    dS = torch.empty_like(dP_dropped)

    SINGLE_PASS_MAX = 4096
    pow2 = 2 ** math.ceil(math.log2(seq_kv)) if seq_kv > 1 else 1

    if pow2 <= SINGLE_PASS_MAX:
        # Single-pass: one program per row, entire row fits in registers
        BLOCK_SIZE = pow2
        grid = (N_rows,)
        _softmax_bwd_dropout_kernel_singlepass[grid](
            dP_dropped, P, mask, dS,
            seq_kv=seq_kv,
            scale=scale,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Two-pass parallel: 2D grid (N_rows, n_blocks)
        # First pass: compute partial sums per block
        # Second pass: reduce sums and apply
        BLOCK_SIZE = 2048
        n_blocks = math.ceil(seq_kv / BLOCK_SIZE)
        # n_blocks must be power-of-2 for the second kernel's tl.arange
        n_blocks_pow2 = 2 ** math.ceil(math.log2(n_blocks)) if n_blocks > 1 else 1

        row_sum_buf = torch.zeros(N_rows, n_blocks_pow2, dtype=torch.float32,
                                  device=dP_dropped.device)

        grid2d = (N_rows, n_blocks)
        _softmax_bwd_partial_sum_kernel[grid2d](
            dP_dropped, P, mask, row_sum_buf,
            seq_kv=seq_kv,
            scale=scale,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        _softmax_bwd_apply_kernel[grid2d](
            dP_dropped, P, mask, row_sum_buf, dS,
            seq_kv=seq_kv,
            scale=scale,
            BLOCK_SIZE=BLOCK_SIZE,
            n_blocks=n_blocks_pow2,
        )

    return dS


# Persistent side stream for BMM2 overlap
_side_stream = None

def _get_side_stream():
    global _side_stream
    if _side_stream is None:
        _side_stream = torch.cuda.Stream()
    return _side_stream


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

    # ── Step 1: Prepare dO in GQA-grouped layout ──────────────────────────────
    # dO_in: [bs, sq, 80, d] -> [bs, sq, 8, 10, d] -> [bs, 8, 10, sq, d]
    # -> contiguous [bs*8, 10*sq, d]  (batch over kv-heads, M = 10*sq)
    dO_grouped = dO_in.reshape(bs, seq_q, n_kv, n_g, d) \
                      .permute(0, 2, 3, 1, 4) \
                      .contiguous() \
                      .reshape(bs * n_kv, n_g * seq_q, d)  # [bs*8, 10*sq, d]

    # Value states: [bs, 8, skv, d] -> [bs*8, skv, d]
    vs_flat = value_states.reshape(bs * n_kv, seq_kv, d).contiguous()  # [bs*8, skv, d]

    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    # -> [bs*8, 10*sq, skv]
    Pd_grouped = attn_weights_dropped.reshape(bs, n_kv, n_g, seq_q, seq_kv) \
                                     .reshape(bs * n_kv, n_g * seq_q, seq_kv)
    # Note: reshape may not be contiguous but bmm should handle it if last dims are contiguous
    Pd_grouped = Pd_grouped.contiguous()

    # ── Step 2: Fork — launch BMM2 on side stream BEFORE BMM1 ────────────────
    side_stream    = _get_side_stream()
    default_stream = torch.cuda.current_stream()

    side_stream.wait_stream(default_stream)

    with torch.cuda.stream(side_stream):
        # BMM2: dV = Pd^T @ dO
        # [bs*8, 10*sq, skv]^T @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
        dV_flat = torch.bmm(
            Pd_grouped.float().transpose(-2, -1),   # [bs*8, skv, 10*sq]
            dO_grouped.float()                       # [bs*8, 10*sq, d]
        )  # [bs*8, skv, d]
        dV_fp32 = dV_flat.reshape(bs, n_kv, seq_kv, d)

    # ── Step 3: BMM1 on default stream ───────────────────────────────────────
    # dP_dropped = dO @ V^T
    # [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
    dP_dropped_grouped = torch.bmm(
        dO_grouped,                      # [bs*8, 10*sq, d]
        vs_flat.transpose(-2, -1)        # [bs*8, d, skv]
    )  # [bs*8, 10*sq, skv]

    # ── Step 4: Softmax backward (Triton, on default stream) ─────────────────
    # Reshape back to [bs, 80, sq, skv] for the softmax kernel
    # [bs*8, 10*sq, skv] -> [bs, 8, 10, sq, skv] -> [bs, 80, sq, skv]
    dP_dropped = dP_dropped_grouped.reshape(bs, n_kv, n_g, seq_q, seq_kv) \
                                   .reshape(bs, n_heads, seq_q, seq_kv)

    scale  = 1.0 / (1.0 - attention_dropout)
    N_rows = bs * n_heads * seq_q

    dP_dropped_2d = dP_dropped.reshape(N_rows, seq_kv)
    P_2d          = attn_weights.reshape(N_rows, seq_kv)
    mask_2d       = dropout_mask.reshape(N_rows, seq_kv)

    dS_2d = fused_softmax_bwd_dropout(dP_dropped_2d, P_2d, mask_2d, scale, seq_kv)
    dS    = dS_2d.reshape(bs, n_heads, seq_q, seq_kv)

    # ── Step 5: Wait for BMM2, finalize dV ───────────────────────────────────
    default_stream.wait_stream(side_stream)
    dV = dV_fp32.to(torch.bfloat16)

    return dS, dV
