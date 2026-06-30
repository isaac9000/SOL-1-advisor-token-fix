"""
Optimized attention-backward kernel — fused Triton softmax-bwd + GQA-aware BMMs.

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
    """
    Single-pass fused kernel: for each row r,
        dP[r,:] = dP_dropped[r,:] * mask[r,:] * scale
        dS[r,:] = P[r,:] * (dP[r,:] - sum(dP[r,:] * P[r,:]))

    When BLOCK_SIZE >= seq_kv, the entire row fits in registers so we do:
      1) Load everything into registers
      2) Compute row_sum in-register
      3) Compute dS and store
    This avoids a second pass over global memory entirely.

    For larger seq_kv (multiple blocks), we still need two passes over the
    blocks but keep data in registers within each block to avoid re-loading.
    We use the standard two-loop approach but with the key optimization that
    when seq_kv <= BLOCK_SIZE we only touch memory once per element.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * seq_kv

    # ── Single-pass: load all data into registers, accumulate, then store ────
    # This works when seq_kv fits in one block (BLOCK_SIZE >= seq_kv).
    # For the multi-block case we fall back to two passes, but the inner loop
    # benefit is still that we don't re-issue global loads for the second pass
    # within each block — instead we cache in registers.

    # We unroll by processing the whole row in one shot if it fits.
    # Triton will keep vectors in registers across the two loops.
    # The trick: use a static number of blocks and keep arrays alive.

    # Use tl.static_range for the case where seq_kv == BLOCK_SIZE (one block)
    # For the general case we compute row_sum first then write.

    # Since BLOCK_SIZE is constexpr and we pick BLOCK_SIZE >= seq_kv when
    # seq_kv is small (<=2048), we handle the single-block case specially.
    offsets = tl.arange(0, BLOCK_SIZE)
    mask_cond = offsets < seq_kv

    # Load all data — single read from global memory
    dp_dropped = tl.load(dP_dropped_ptr + row_start + offsets,
                         mask=mask_cond, other=0.0).to(tl.float32)
    p_val = tl.load(P_ptr + row_start + offsets,
                    mask=mask_cond, other=0.0).to(tl.float32)
    m_val = tl.load(mask_ptr + row_start + offsets,
                    mask=mask_cond, other=0).to(tl.float32)

    # Compute dP (in-register)
    dp = dp_dropped * m_val * scale

    # Compute row_sum = sum(dP * P) — in-register reduction, no extra global read
    row_sum = tl.sum(dp * p_val, axis=0)

    # Compute dS = P * (dP - row_sum) — all in-register
    ds = p_val * (dp - row_sum)

    # Single write to global memory
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
    """
    Two-pass kernel for large seq_kv (when full row doesn't fit in registers).
    Pass 1: accumulate row_sum = sum(dP * P)
    Pass 2: compute and store dS = P * (dP - row_sum)
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * seq_kv

    # First pass: compute sum(dP * P)
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

    # Second pass: compute dS and store
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
    """
    dP_dropped: [N_rows, seq_kv] bfloat16
    P:          [N_rows, seq_kv] bfloat16
    mask:       [N_rows, seq_kv] bool
    Returns dS: [N_rows, seq_kv] bfloat16
    """
    N_rows = dP_dropped.shape[0]
    dS = torch.empty_like(dP_dropped)

    # Determine block size
    # Single-pass kernel: BLOCK_SIZE must be >= seq_kv (power of 2)
    SINGLE_PASS_MAX = 4096  # register pressure limit

    pow2 = 2 ** math.ceil(math.log2(seq_kv)) if seq_kv > 1 else 1

    grid = (N_rows,)

    if pow2 <= SINGLE_PASS_MAX:
        # Single-pass: entire row fits in registers
        BLOCK_SIZE = pow2
        _softmax_bwd_dropout_kernel_singlepass[grid](
            dP_dropped, P, mask, dS,
            seq_kv=seq_kv,
            scale=scale,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Multi-block two-pass for very large seq_kv
        BLOCK_SIZE = 2048
        _softmax_bwd_dropout_kernel_multiblock[grid](
            dP_dropped, P, mask, dS,
            seq_kv=seq_kv,
            scale=scale,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return dS


def custom_kernel(data):
    (dO_in, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = dO_in.shape[0]
    seq_q  = dO_in.shape[1]
    seq_kv = value_states.shape[2]
    n_kv   = NUM_KEY_VALUE_HEADS   # 8
    n_g    = N_GROUPS              # 10
    d      = HEAD_DIM              # 128

    # [bs, sq, 80, d] -> [bs, 80, sq, d] -> contiguous bfloat16
    dO = dO_in.transpose(1, 2).contiguous()  # [bs, 80, sq, d] bfloat16

    # ── dP computation: dO @ V^T  (GQA-aware, no expansion) ──────────────────
    # Reshape dO: [bs, 80, sq, d] -> [bs*8, 10*sq, d]
    dO_reshaped = dO.reshape(bs, n_kv, n_g, seq_q, d).reshape(bs * n_kv, n_g * seq_q, d)

    # value_states: [bs, 8, skv, d] -> [bs*8, skv, d]
    vs_flat = value_states.reshape(bs * n_kv, seq_kv, d)

    # Launch both BMMs before softmax kernel so L2 can be warm
    # BMM: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]  (bfloat16)
    dP_dropped_flat = torch.bmm(dO_reshaped, vs_flat.transpose(-2, -1))  # [bs*8, 10*sq, skv]

    # dV BMM: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]  (bfloat16)
    # Use attn_weights_dropped reshaped: [bs, 80, sq, skv] -> [bs*8, 10*sq, skv]
    Pd_flat = attn_weights_dropped.reshape(bs * n_kv, n_g * seq_q, seq_kv)
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_reshaped)  # [bs*8, skv, d]

    # Reshape back: [bs*8, 10*sq, skv] -> [bs, 80, sq, skv]
    dP_dropped = dP_dropped_flat.reshape(bs, n_kv * n_g, seq_q, seq_kv)

    # ── Fused softmax backward + dropout (single-pass Triton kernel) ──────────
    scale = 1.0 / (1.0 - attention_dropout)

    # Flatten row dimensions: [bs, 80, sq, skv] -> [bs*80*sq, skv]
    N_rows = bs * NUM_ATTENTION_HEADS * seq_q
    dP_dropped_2d = dP_dropped.contiguous().reshape(N_rows, seq_kv)
    P_2d = attn_weights.contiguous().reshape(N_rows, seq_kv)
    mask_2d = dropout_mask.contiguous().reshape(N_rows, seq_kv)

    dS_2d = fused_softmax_bwd_dropout(dP_dropped_2d, P_2d, mask_2d, scale, seq_kv)
    dS = dS_2d.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # Reshape dV to [bs, 8, skv, d]
    dV = dV_flat.reshape(bs, n_kv, seq_kv, d).to(torch.bfloat16)

    return dS, dV
