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
N_GROUPS = 10


# ---------------------------------------------------------------------------
# Fused pointwise kernel: given dP_raw [bs, 80, sq, skv] (float32),
# dropout_mask [bs, 80, sq, skv] (bool), P [bs, 80, sq, skv] (bfloat16),
# compute dS = P * (dP - sum_skv(dP * P)) in one pass over rows.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,      # [bs, n_heads, sq, skv]  float32
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
    Each program handles ONE row: (bs_idx, head_idx, sq_idx).
    We process the skv dimension in BLOCK_SKV tiles.
    Pass 1: accumulate row_sum = sum_skv(dP * P) in registers.
    Pass 2: compute dS = P * (dP - row_sum) and store.
    Both passes are sequential over HBM — but for a single row, the working
    set is small enough to stay in L2 cache between the two passes.
    """
    pid = tl.program_id(0)   # flattened (bs, head, sq) index
    n_heads = 80

    # Decompose pid into (bs_idx, h_idx, sq_idx)
    total_heads = 80
    bh_idx = pid // sq
    sq_idx = pid % sq
    bs_idx = bh_idx // total_heads
    h_idx  = bh_idx % total_heads

    # Base offset for this row
    base = bs_idx * stride_bs + h_idx * stride_h + sq_idx * stride_sq

    skv_arange = tl.arange(0, BLOCK_SKV)

    # Pass 1: compute row_sum = sum_skv(dP * P)
    row_sum = tl.zeros([1], dtype=tl.float32)

    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        dP_tile = tl.load(
            dP_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        )  # float32

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
        )

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
    # Step 1: Prepare dO as [bs, 80, sq, 128] bfloat16, contiguous
    # Reshape to [bs, 8, 10*sq, 128] for GQA-native BMMs
    # ----------------------------------------------------------------
    dO = grad_attn_output.transpose(1, 2).contiguous()
    # dO: [bs, 80, sq, 128]  bfloat16

    # Reshape dO to group by KV head: [bs, 8, 10*sq, 128]
    dO_grouped = dO.reshape(bs, n_kv_heads, n_groups * seq_q, HEAD_DIM)
    # dO_grouped: [bs, 8, 10*sq, 128]  bfloat16

    # ----------------------------------------------------------------
    # Step 2: dP_raw = dO_grouped @ V^T  -> [bs, 8, 10*sq, skv]
    # GQA-native: avoids 10x V expansion entirely!
    # V: [bs, 8, skv, 128]  bfloat16
    # dP_raw: [bs, 8, 10*sq, skv]  bfloat16 -> float32
    # ----------------------------------------------------------------
    V = value_states  # [bs, 8, skv, 128]  bfloat16, already contiguous

    # BMM in bfloat16: [bs, 8, 10*sq, 128] x [bs, 8, 128, skv] -> [bs, 8, 10*sq, skv]
    dP_raw_grouped = torch.matmul(dO_grouped, V.transpose(-2, -1)).to(torch.float32)
    # Reshape back to [bs, 80, sq, skv]
    dP_raw = dP_raw_grouped.reshape(bs, n_heads, seq_q, seq_kv)
    # dP_raw: [bs, 80, sq, skv]  float32

    # ----------------------------------------------------------------
    # Step 3: Fused Triton pointwise kernel for softmax backward
    # Each program handles one (bs, head, sq_row):
    #   dP = dP_raw * mask * inv_scale
    #   row_sum = sum(dP * P, dim=-1)
    #   dS = P * (dP - row_sum)
    # ----------------------------------------------------------------
    P = attn_weights.contiguous()
    mask_c = dropout_mask.contiguous()
    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=dO.device)

    inv_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # All tensors have the same [bs, 80, sq, skv] layout
    stride_bs  = dP_raw.stride(0)
    stride_h   = dP_raw.stride(1)
    stride_sq_s  = dP_raw.stride(2)
    stride_skv = dP_raw.stride(3)

    # One program per row: grid = bs * n_heads * seq_q
    BLOCK_SKV_K = 512  # 512 elements × 4 bytes = 2KB per tile, good L2 reuse

    grid_softmax = (bs * n_heads * seq_q,)

    softmax_bwd_kernel[grid_softmax](
        dP_raw, P, mask_c, dS,
        inv_scale,
        seq_q, seq_kv,
        stride_bs, stride_h, stride_sq_s, stride_skv,
        BLOCK_SKV=BLOCK_SKV_K,
        num_warps=4,
        num_stages=3,
    )

    # ----------------------------------------------------------------
    # Step 4: dV = P_drop_grouped^T @ dO_grouped  -> [bs, 8, skv, 128]
    # GQA-native: reshape P_drop to [bs, 8, 10*sq, skv], no separate sum needed!
    # cuBLAS: [bs, 8, skv, 10*sq] x [bs, 8, 10*sq, 128] -> [bs, 8, skv, 128]
    # ----------------------------------------------------------------
    # Reshape attn_weights_dropped from [bs, 80, sq, skv] -> [bs, 8, 10*sq, skv]
    P_drop_grouped = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups * seq_q, seq_kv)
    # P_drop_grouped: [bs, 8, 10*sq, skv]  bfloat16

    # BMM in bfloat16: [bs, 8, skv, 10*sq] x [bs, 8, 10*sq, 128] -> [bs, 8, skv, 128]
    dV = torch.matmul(P_drop_grouped.transpose(-2, -1), dO_grouped).to(torch.bfloat16)
    # dV: [bs, 8, skv, 128]  bfloat16

    return dS, dV
