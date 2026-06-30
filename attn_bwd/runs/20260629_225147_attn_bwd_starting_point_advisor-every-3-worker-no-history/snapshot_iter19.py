"""
Attention backward: GQA-native cuBLAS batched GEMMs (no V expansion) +
fused Triton pointwise kernel for softmax backward.

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
# Fused pointwise kernel: given dP_raw [bs, 80, sq, skv] (bfloat16),
# dropout_mask [bs, 80, sq, skv] (bool), P [bs, 80, sq, skv] (bfloat16),
# compute dS = P * (dP - sum_skv(dP * P)) in one pass over rows.
#
# Single-pass variant: when BLOCK_SKV >= skv, we load everything once
# into registers, compute the row_sum, then compute dS in the same pass.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_kernel_single(
    dP_ptr,      # [bs, n_heads, sq, skv]  bfloat16
    P_ptr,       # [bs, n_heads, sq, skv]  bfloat16
    mask_ptr,    # [bs, n_heads, sq, skv]  bool
    dS_ptr,      # [bs, n_heads, sq, skv]  bfloat16  (output)
    inv_scale,   # scalar float
    sq, skv,
    # strides for all 4D tensors (same layout)
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SKV: tl.constexpr,
):
    """
    Single-pass variant: load all tiles once, accumulate row_sum in registers,
    then compute dS in a second sweep over the already-loaded tiles.
    Each program handles ONE row: (bs_idx, head_idx, sq_idx).
    dP_ptr is bfloat16 (halved bandwidth vs float32).
    """
    pid = tl.program_id(0)   # flattened (bs, head, sq) index
    n_heads = 80

    bh_idx = pid // sq
    sq_idx = pid % sq
    bs_idx = bh_idx // n_heads
    h_idx  = bh_idx % n_heads

    # Base offset for this row
    base = bs_idx * stride_bs + h_idx * stride_h + sq_idx * stride_sq

    # Single tile covers the full skv dimension
    skv_offs = tl.arange(0, BLOCK_SKV)
    skv_mask = skv_offs < skv

    # Load dP as bfloat16, convert to float32 for computation
    dP_tile = tl.load(
        dP_ptr + base + skv_offs * stride_skv,
        mask=skv_mask, other=0.0,
    ).to(tl.float32)  # bfloat16 -> float32

    drop = tl.load(
        mask_ptr + base + skv_offs * stride_skv,
        mask=skv_mask, other=0,
    )
    dP_tile = dP_tile * drop.to(tl.float32) * inv_scale

    P_tile = tl.load(
        P_ptr + base + skv_offs * stride_skv,
        mask=skv_mask, other=0.0,
    ).to(tl.float32)

    # Compute row_sum and dS in single pass
    row_sum = tl.sum(dP_tile * P_tile, axis=0)
    dS_tile = P_tile * (dP_tile - row_sum)

    tl.store(
        dS_ptr + base + skv_offs * stride_skv,
        dS_tile.to(tl.bfloat16),
        mask=skv_mask,
    )


@triton.jit
def softmax_bwd_kernel(
    dP_ptr,      # [bs, n_heads, sq, skv]  bfloat16
    P_ptr,       # [bs, n_heads, sq, skv]  bfloat16
    mask_ptr,    # [bs, n_heads, sq, skv]  bool
    dS_ptr,      # [bs, n_heads, sq, skv]  bfloat16  (output)
    inv_scale,   # scalar float
    sq, skv,
    # strides for all 4D tensors (same layout)
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SKV: tl.constexpr,
):
    """
    Two-pass variant: used when skv > BLOCK_SKV (needs multiple tiles).
    Each program handles ONE row: (bs_idx, head_idx, sq_idx).
    dP_ptr is bfloat16 (halved bandwidth vs float32).
    """
    pid = tl.program_id(0)
    n_heads = 80

    bh_idx = pid // sq
    sq_idx = pid % sq
    bs_idx = bh_idx // n_heads
    h_idx  = bh_idx % n_heads

    base = bs_idx * stride_bs + h_idx * stride_h + sq_idx * stride_sq

    skv_arange = tl.arange(0, BLOCK_SKV)

    # Pass 1: accumulate row_sum = sum_skv(dP * P)
    row_sum = tl.zeros([1], dtype=tl.float32)

    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        dP_tile = tl.load(
            dP_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        ).to(tl.float32)  # bfloat16 -> float32

        drop = tl.load(
            mask_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0,
        )
        dP_tile = dP_tile * drop.to(tl.float32) * inv_scale

        P_tile = tl.load(
            P_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        ).to(tl.float32)

        row_sum += tl.sum(dP_tile * P_tile, axis=0)

    # Pass 2: compute and store dS
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        dP_tile = tl.load(
            dP_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        ).to(tl.float32)  # bfloat16 -> float32

        drop = tl.load(
            mask_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0,
        )
        dP_tile = dP_tile * drop.to(tl.float32) * inv_scale

        P_tile = tl.load(
            P_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        ).to(tl.float32)

        dS_tile = P_tile * (dP_tile - row_sum)

        tl.store(
            dS_ptr + base + skv_offs * stride_skv,
            dS_tile.to(tl.bfloat16),
            mask=skv_mask,
        )


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
    # Reshape to [bs, sq, 8, 10, 128], then permute to [bs, 8, sq, 10, 128]
    # then contiguous + reshape to [bs, 8, 10*sq, 128]
    # This avoids creating an intermediate [bs, 80, sq, 128] tensor.
    # ----------------------------------------------------------------
    dO_5d = grad_attn_output.reshape(bs, seq_q, n_kv_heads, n_groups, HEAD_DIM)
    # [bs, sq, 8, 10, 128] -> [bs, 8, sq, 10, 128]
    dO_5d_perm = dO_5d.permute(0, 2, 1, 3, 4)
    # contiguous + merge sq and n_groups dims: [bs, 8, sq*10, 128] — NOT sq*10 but 10*sq
    # We want [bs, 8, 10*sq, 128] which means we need to merge (n_groups, sq) not (sq, n_groups)
    # So permute to [bs, 8, n_groups, sq, 128] then reshape
    dO_5d_perm2 = dO_5d.permute(0, 2, 3, 1, 4)  # [bs, 8, 10, sq, 128]
    dO_grouped = dO_5d_perm2.contiguous().reshape(bs, n_kv_heads, n_groups * seq_q, HEAD_DIM)
    # dO_grouped: [bs, 8, 10*sq, 128]  bfloat16, contiguous

    # ----------------------------------------------------------------
    # Step 2: dP_raw = dO_grouped @ V^T  -> [bs, 8, 10*sq, skv] -> [bs, 80, sq, skv]
    # GQA-native: avoids 10x V expansion entirely!
    # V: [bs, 8, skv, 128]  bfloat16
    # Keep result in bfloat16 to halve the bandwidth for softmax backward.
    # ----------------------------------------------------------------
    V = value_states  # [bs, 8, skv, 128]  bfloat16, already contiguous

    # BMM: [bs, 8, 10*sq, 128] x [bs, 8, 128, skv] -> [bs, 8, 10*sq, skv]  bfloat16
    dP_raw_grouped = torch.matmul(dO_grouped, V.transpose(-2, -1))
    # Stay in bfloat16 — reshape is a view: [bs, 8, 10*sq, skv] -> [bs, 80, sq, skv]
    dP_raw = dP_raw_grouped.reshape(bs, n_heads, seq_q, seq_kv)
    # dP_raw: [bs, 80, sq, skv]  bfloat16, contiguous

    # ----------------------------------------------------------------
    # Step 3: Fused Triton pointwise kernel for softmax backward
    # dP_raw is now bfloat16 — halved memory bandwidth for reads
    # ----------------------------------------------------------------
    P = attn_weights  # [bs, 80, sq, skv] bfloat16
    mask_c = dropout_mask  # [bs, 80, sq, skv] bool
    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=dO_grouped.device)

    inv_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    stride_bs    = dP_raw.stride(0)
    stride_h     = dP_raw.stride(1)
    stride_sq_s  = dP_raw.stride(2)
    stride_skv   = dP_raw.stride(3)

    grid_softmax = (bs * n_heads * seq_q,)

    # Use single-pass kernel when seq_kv fits in one tile (power-of-2 block)
    if seq_kv <= 1024:
        BLOCK_SKV_K = 1024
        softmax_bwd_kernel_single[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=4,
            num_stages=1,
        )
    elif seq_kv <= 2048:
        BLOCK_SKV_K = 2048
        softmax_bwd_kernel_single[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=8,
            num_stages=1,
        )
    elif seq_kv <= 4096:
        BLOCK_SKV_K = 4096
        softmax_bwd_kernel_single[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=16,
            num_stages=1,
        )
    else:
        BLOCK_SKV_K = 2048
        softmax_bwd_kernel[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=16,
            num_stages=2,
        )

    # ----------------------------------------------------------------
    # Step 4: dV = P_drop_grouped^T @ dO_grouped  -> [bs, 8, skv, 128]
    # GQA-native: reshape P_drop to [bs, 8, 10*sq, skv], no separate sum needed!
    # cuBLAS: [bs, 8, skv, 10*sq] x [bs, 8, 10*sq, 128] -> [bs, 8, skv, 128]
    # ----------------------------------------------------------------
    # Reshape attn_weights_dropped from [bs, 80, sq, skv] -> [bs, 8, 10*sq, skv]
    # attn_weights_dropped is contiguous [bs, 80, sq, skv], reshape is a view
    P_drop_grouped = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups * seq_q, seq_kv)
    # P_drop_grouped: [bs, 8, 10*sq, skv]  bfloat16

    # BMM in bfloat16: [bs, 8, skv, 10*sq] x [bs, 8, 10*sq, 128] -> [bs, 8, skv, 128]
    dV = torch.matmul(P_drop_grouped.transpose(-2, -1), dO_grouped).to(torch.bfloat16)
    # dV: [bs, 8, skv, 128]  bfloat16

    return dS, dV
