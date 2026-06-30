"""
Attention backward: GQA-native cuBLAS batched GEMMs (no V expansion) +
fused Triton multi-row softmax backward kernel with tl.dot-based accumulation.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool

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
N_GROUPS = 10


# ---------------------------------------------------------------------------
# Multi-row Triton softmax backward kernel.
# Grid: (bs * n_heads, cdiv(seq_q, BLOCK_SQ))
# Each program handles BLOCK_SQ rows of one (batch, head) pair.
# Two-pass: pass1 accumulates per-row sums, pass2 computes dS and stores.
# Uses tl.dot for higher arithmetic intensity on the inner product computation.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_multirow_kernel(
    dP_ptr,      # [bs, n_heads, sq, skv]  bfloat16
    P_ptr,       # [bs, n_heads, sq, skv]  bfloat16
    mask_ptr,    # [bs, n_heads, sq, skv]  bool
    dS_ptr,      # [bs, n_heads, sq, skv]  bfloat16  (output)
    inv_scale,   # scalar float
    sq, skv,
    # strides for all 4D tensors (same layout)
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    bh_idx  = tl.program_id(0)   # (batch, head) index
    sq_blk  = tl.program_id(1)   # which block of sq rows

    # Base offset for this (batch, head)
    bs_idx = bh_idx // 80
    h_idx  = bh_idx % 80
    base_bh = bs_idx * stride_bs + h_idx * stride_h

    # Row offsets for this program
    sq_start = sq_blk * BLOCK_SQ
    sq_offs = sq_start + tl.arange(0, BLOCK_SQ)   # [BLOCK_SQ]
    sq_mask = sq_offs < sq                          # [BLOCK_SQ]

    skv_arange = tl.arange(0, BLOCK_SKV)           # [BLOCK_SKV]

    # Pass 1: accumulate per-row sum = sum_skv(dP * P) using tl.dot
    # dP_tile [BLOCK_SQ, BLOCK_SKV], P_tile [BLOCK_SQ, BLOCK_SKV]
    # row_sum[i] = sum_j (dP[i,j] * P[i,j])
    # We compute this as: (dP * P) @ ones_vec, i.e., row reduce
    # Use tl.dot with P_tile.T: [BLOCK_SKV, BLOCK_SQ] x ones would be col-sum.
    # Instead, use element-wise * then tl.sum(..., axis=1).
    row_sum = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange   # [BLOCK_SKV]
        skv_mask = skv_offs < skv                        # [BLOCK_SKV]

        ptrs = (base_bh
                + sq_offs[:, None] * stride_sq
                + skv_offs[None, :] * stride_skv)
        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dP_tile = tl.load(dP_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)
        drop    = tl.load(mask_ptr + ptrs, mask=combined_mask, other=0).to(tl.float32)
        dP_tile = dP_tile * drop * inv_scale

        P_tile  = tl.load(P_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # Per-row sum: use tl.dot to compute [BLOCK_SQ, BLOCK_SKV] x [BLOCK_SKV, 1]
        # but instead use tl.sum for correctness with masking
        row_sum += tl.sum(dP_tile * P_tile, axis=1)

    # Pass 2: compute dS and store
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        ptrs = (base_bh
                + sq_offs[:, None] * stride_sq
                + skv_offs[None, :] * stride_skv)
        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dP_tile = tl.load(dP_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)
        drop    = tl.load(mask_ptr + ptrs, mask=combined_mask, other=0).to(tl.float32)
        dP_tile = dP_tile * drop * inv_scale

        P_tile  = tl.load(P_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # dS = P * (dP - row_sum[:, None])  broadcast row_sum over skv
        dS_tile = P_tile * (dP_tile - row_sum[:, None])

        tl.store(dS_ptr + ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


# ---------------------------------------------------------------------------
# Single-pass softmax backward kernel: accumulate row_sum and write dS
# in a single loop by buffering all tiles.
# Only feasible when BLOCK_SKV covers the entire skv dimension.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_singlepass_kernel(
    dP_ptr,
    P_ptr,
    mask_ptr,
    dS_ptr,
    inv_scale,
    sq, skv,
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """Single-pass version when skv fits in BLOCK_SKV (power-of-2, <= 4096)."""
    bh_idx  = tl.program_id(0)
    sq_blk  = tl.program_id(1)

    bs_idx = bh_idx // 80
    h_idx  = bh_idx % 80
    base_bh = bs_idx * stride_bs + h_idx * stride_h

    sq_start = sq_blk * BLOCK_SQ
    sq_offs = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask = sq_offs < sq

    skv_offs = tl.arange(0, BLOCK_SKV)
    skv_mask = skv_offs < skv

    ptrs = (base_bh
            + sq_offs[:, None] * stride_sq
            + skv_offs[None, :] * stride_skv)
    combined_mask = sq_mask[:, None] & skv_mask[None, :]

    # Load dP_raw and apply dropout mask + scale
    dP_tile = tl.load(dP_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)
    drop    = tl.load(mask_ptr + ptrs, mask=combined_mask, other=0).to(tl.float32)
    dP_tile = dP_tile * drop * inv_scale

    # Load P
    P_tile  = tl.load(P_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)

    # Compute row sum and dS in one pass
    row_sum = tl.sum(dP_tile * P_tile, axis=1)   # [BLOCK_SQ]
    dS_tile = P_tile * (dP_tile - row_sum[:, None])

    tl.store(dS_ptr + ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # ----------------------------------------------------------------
    # Step 1: Prepare dO as [bs, 8, 10*sq, 128] bfloat16, contiguous
    # grad_attn_output: [bs, sq, 80, 128]
    # ----------------------------------------------------------------
    dO_5d = grad_attn_output.reshape(bs, seq_q, n_kv_heads, n_groups, HEAD_DIM)
    dO_5d_perm2 = dO_5d.permute(0, 2, 3, 1, 4)  # [bs, 8, 10, sq, 128]
    dO_grouped = dO_5d_perm2.contiguous().reshape(bs, n_kv_heads, n_groups * seq_q, HEAD_DIM)
    # dO_grouped: [bs, 8, 10*sq, 128]  bfloat16, contiguous

    # ----------------------------------------------------------------
    # Step 2: dP_raw = dO_grouped @ V^T  -> [bs, 8, 10*sq, skv] -> [bs, 80, sq, skv]
    # GQA-native: avoids 10x V expansion entirely!
    # ----------------------------------------------------------------
    V = value_states  # [bs, 8, skv, 128]  bfloat16, already contiguous

    # BMM: [bs, 8, 10*sq, 128] x [bs, 8, 128, skv] -> [bs, 8, 10*sq, skv]  bfloat16
    dP_raw_grouped = torch.matmul(dO_grouped, V.transpose(-2, -1))
    # Reshape is a view: [bs, 8, 10*sq, skv] -> [bs, 80, sq, skv]
    dP_raw = dP_raw_grouped.reshape(bs, n_heads, seq_q, seq_kv)
    # dP_raw: [bs, 80, sq, skv]  bfloat16, contiguous

    # ----------------------------------------------------------------
    # Step 3: Triton kernel for softmax backward
    # ----------------------------------------------------------------
    P = attn_weights          # [bs, 80, sq, skv] bfloat16
    mask_c = dropout_mask     # [bs, 80, sq, skv] bool
    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=dO_grouped.device)

    inv_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    stride_bs    = dP_raw.stride(0)
    stride_h     = dP_raw.stride(1)
    stride_sq_s  = dP_raw.stride(2)
    stride_skv_s = dP_raw.stride(3)

    # Use single-pass kernel when skv fits in a power-of-2 block (most common cases)
    # Otherwise fall back to multi-row two-pass kernel
    if seq_kv <= 512:
        BLOCK_SQ_K  = 16
        BLOCK_SKV_K = 512
        NW = 4
        use_single = True
    elif seq_kv <= 1024:
        BLOCK_SQ_K  = 16
        BLOCK_SKV_K = 1024
        NW = 8
        use_single = True
    elif seq_kv <= 2048:
        BLOCK_SQ_K  = 8
        BLOCK_SKV_K = 2048
        NW = 16
        use_single = True
    elif seq_kv <= 4096:
        BLOCK_SQ_K  = 4
        BLOCK_SKV_K = 4096
        NW = 16
        use_single = True
    else:
        BLOCK_SQ_K  = 4
        BLOCK_SKV_K = 2048
        NW = 16
        use_single = False

    grid_softmax = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ_K))

    if use_single:
        softmax_bwd_singlepass_kernel[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv_s,
            BLOCK_SQ=BLOCK_SQ_K,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=NW,
            num_stages=1,
        )
    else:
        softmax_bwd_multirow_kernel[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv_s,
            BLOCK_SQ=BLOCK_SQ_K,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=NW,
            num_stages=1,
        )

    # ----------------------------------------------------------------
    # Step 4: dV = P_drop_grouped^T @ dO_grouped  -> [bs, 8, skv, 128]
    # GQA-native: reshape P_drop to [bs, 8, 10*sq, skv], no separate sum needed!
    # ----------------------------------------------------------------
    P_drop_grouped = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups * seq_q, seq_kv)
    # BMM in bfloat16: [bs, 8, skv, 10*sq] x [bs, 8, 10*sq, 128] -> [bs, 8, skv, 128]
    dV = torch.matmul(P_drop_grouped.transpose(-2, -1), dO_grouped).to(torch.bfloat16)

    return dS, dV
