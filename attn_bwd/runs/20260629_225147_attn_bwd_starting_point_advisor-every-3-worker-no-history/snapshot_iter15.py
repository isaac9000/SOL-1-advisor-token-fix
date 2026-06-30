"""
Attention backward: GQA-native cuBLAS batched GEMM for dV +
Triton fused kernel for dS (fuses dO@V.T matmul with softmax backward).

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
# Fused kernel: for each (bs, head80, sq_row):
#   - loads dO row [128] once from grad_attn_output
#   - tiles over skv, computing dP_tile = dot(dO_row, V_tile.T) in fp32
#   - Pass 1: accumulate row_sum = sum(dP_dropped * P)
#   - Pass 2: compute dS = P * (dP_dropped - row_sum), store bfloat16
#
# V is indexed with head//10 to handle GQA (8 KV heads, 10 groups each).
# This avoids materializing [bs, 80, sq, skv] float32 entirely.
# ---------------------------------------------------------------------------
@triton.jit
def fused_ds_kernel(
    dO_ptr,      # [bs, 80, sq, 128]   bfloat16   (contiguous after transpose)
    V_ptr,       # [bs,  8, skv, 128]  bfloat16
    P_ptr,       # [bs, 80, sq, skv]   bfloat16
    mask_ptr,    # [bs, 80, sq, skv]   bool
    dS_ptr,      # [bs, 80, sq, skv]   bfloat16   (output)
    inv_scale: tl.constexpr,
    sq, skv,
    # dO strides: [bs, 80, sq, 128]
    dO_s_bs, dO_s_h, dO_s_sq,  # stride_skv == 1 (head_dim innermost)
    # V strides: [bs, 8, skv, 128]
    V_s_bs, V_s_h, V_s_skv,    # stride_d == 1
    # P/mask/dS strides: [bs, 80, sq, skv]
    P_s_bs, P_s_h, P_s_sq,     # stride_skv == 1
    HEAD_DIM: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """
    Each program handles one (bs, head80, sq_row).
    Two-pass over skv tiles.
    """
    pid = tl.program_id(0)
    n_heads = 80

    # Decode pid -> (bs_idx, h_idx, sq_idx)
    sq_idx  = pid % sq
    bh_idx  = pid // sq
    h_idx   = bh_idx % n_heads
    bs_idx  = bh_idx // n_heads
    kv_h_idx = h_idx // 10  # GQA: map head80 -> KV head (0..7)

    # Load dO row [HEAD_DIM] once into registers
    d_offs = tl.arange(0, HEAD_DIM)
    dO_base = bs_idx * dO_s_bs + h_idx * dO_s_h + sq_idx * dO_s_sq
    dO_row = tl.load(dO_ptr + dO_base + d_offs).to(tl.float32)
    # dO_row: [HEAD_DIM] float32

    # Pass 1: accumulate row_sum = sum_skv(dP_dropped * P)
    row_sum = tl.zeros([1], dtype=tl.float32)

    V_base  = bs_idx * V_s_bs + kv_h_idx * V_s_h
    P_base  = bs_idx * P_s_bs + h_idx * P_s_h + sq_idx * P_s_sq
    skv_arange = tl.arange(0, BLOCK_SKV)

    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        # Load V tile [BLOCK_SKV, HEAD_DIM]
        V_tile = tl.load(
            V_ptr + V_base + skv_offs[:, None] * V_s_skv + d_offs[None, :],
            mask=skv_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        # dP_tile[j] = dot(dO_row, V_tile[j])
        dP_tile = tl.sum(dO_row[None, :] * V_tile, axis=1)  # [BLOCK_SKV]

        # Apply dropout mask
        drop = tl.load(
            mask_ptr + P_base + skv_offs,
            mask=skv_mask, other=0,
        )
        dP_dropped = dP_tile * drop.to(tl.float32) * inv_scale

        # Load P tile
        P_tile = tl.load(
            P_ptr + P_base + skv_offs,
            mask=skv_mask, other=0.0,
        ).to(tl.float32)

        row_sum += tl.sum(dP_dropped * P_tile, axis=0)

    # Pass 2: compute and store dS
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        # Recompute dP_tile (re-stream V)
        V_tile = tl.load(
            V_ptr + V_base + skv_offs[:, None] * V_s_skv + d_offs[None, :],
            mask=skv_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        dP_tile = tl.sum(dO_row[None, :] * V_tile, axis=1)  # [BLOCK_SKV]

        drop = tl.load(
            mask_ptr + P_base + skv_offs,
            mask=skv_mask, other=0,
        )
        dP_dropped = dP_tile * drop.to(tl.float32) * inv_scale

        P_tile = tl.load(
            P_ptr + P_base + skv_offs,
            mask=skv_mask, other=0.0,
        ).to(tl.float32)

        dS_tile = P_tile * (dP_dropped - row_sum)

        tl.store(
            dS_ptr + P_base + skv_offs,
            dS_tile.to(tl.bfloat16),
            mask=skv_mask,
        )


# ---------------------------------------------------------------------------
# Fallback pointwise softmax backward (used alongside cuBLAS dP_raw)
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_kernel_single(
    dP_ptr, P_ptr, mask_ptr, dS_ptr,
    inv_scale,
    sq, skv,
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SKV: tl.constexpr,
):
    pid = tl.program_id(0)
    n_heads = 80
    bh_idx = pid // sq
    sq_idx = pid % sq
    bs_idx = bh_idx // n_heads
    h_idx  = bh_idx % n_heads
    base = bs_idx * stride_bs + h_idx * stride_h + sq_idx * stride_sq
    skv_offs = tl.arange(0, BLOCK_SKV)
    skv_mask = skv_offs < skv
    dP_tile = tl.load(dP_ptr + base + skv_offs * stride_skv, mask=skv_mask, other=0.0)
    drop = tl.load(mask_ptr + base + skv_offs * stride_skv, mask=skv_mask, other=0)
    dP_tile = dP_tile * drop.to(tl.float32) * inv_scale
    P_tile = tl.load(P_ptr + base + skv_offs * stride_skv, mask=skv_mask, other=0.0).to(tl.float32)
    row_sum = tl.sum(dP_tile * P_tile, axis=0)
    dS_tile = P_tile * (dP_tile - row_sum)
    tl.store(dS_ptr + base + skv_offs * stride_skv, dS_tile.to(tl.bfloat16), mask=skv_mask)


@triton.jit
def softmax_bwd_kernel(
    dP_ptr, P_ptr, mask_ptr, dS_ptr,
    inv_scale, sq, skv,
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SKV: tl.constexpr,
):
    pid = tl.program_id(0)
    n_heads = 80
    bh_idx = pid // sq
    sq_idx = pid % sq
    bs_idx = bh_idx // n_heads
    h_idx  = bh_idx % n_heads
    base = bs_idx * stride_bs + h_idx * stride_h + sq_idx * stride_sq
    skv_arange = tl.arange(0, BLOCK_SKV)
    row_sum = tl.zeros([1], dtype=tl.float32)
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv
        dP_tile = tl.load(dP_ptr + base + skv_offs * stride_skv, mask=skv_mask, other=0.0)
        drop = tl.load(mask_ptr + base + skv_offs * stride_skv, mask=skv_mask, other=0)
        dP_tile = dP_tile * drop.to(tl.float32) * inv_scale
        P_tile = tl.load(P_ptr + base + skv_offs * stride_skv, mask=skv_mask, other=0.0).to(tl.float32)
        row_sum += tl.sum(dP_tile * P_tile, axis=0)
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv
        dP_tile = tl.load(dP_ptr + base + skv_offs * stride_skv, mask=skv_mask, other=0.0)
        drop = tl.load(mask_ptr + base + skv_offs * stride_skv, mask=skv_mask, other=0)
        dP_tile = dP_tile * drop.to(tl.float32) * inv_scale
        P_tile = tl.load(P_ptr + base + skv_offs * stride_skv, mask=skv_mask, other=0.0).to(tl.float32)
        dS_tile = P_tile * (dP_tile - row_sum)
        tl.store(dS_ptr + base + skv_offs * stride_skv, dS_tile.to(tl.bfloat16), mask=skv_mask)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    device = grad_attn_output.device

    inv_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # ----------------------------------------------------------------
    # Step 1: Compute dS via fused Triton kernel
    # fused_ds_kernel handles dO@V.T and softmax backward together.
    # dO input: [bs, sq, 80, 128] bfloat16 — need to make [bs, 80, sq, 128] contiguous
    # ----------------------------------------------------------------
    dO_bhsd = grad_attn_output.permute(0, 2, 1, 3).contiguous()  # [bs, 80, sq, 128] bf16

    V = value_states  # [bs, 8, skv, 128] bf16

    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=device)

    # Strides for dO [bs, 80, sq, 128] — innermost dim=128
    dO_s_bs = dO_bhsd.stride(0)
    dO_s_h  = dO_bhsd.stride(1)
    dO_s_sq = dO_bhsd.stride(2)

    # Strides for V [bs, 8, skv, 128]
    V_s_bs  = V.stride(0)
    V_s_h   = V.stride(1)
    V_s_skv = V.stride(2)

    # Strides for P/mask/dS [bs, 80, sq, skv]
    P_s_bs = attn_weights.stride(0)
    P_s_h  = attn_weights.stride(1)
    P_s_sq = attn_weights.stride(2)

    grid = (bs * n_heads * seq_q,)

    # Choose BLOCK_SKV based on seq_kv
    # The dot product needs HEAD_DIM=128 width; BLOCK_SKV controls skv tiling
    if seq_kv <= 128:
        BLOCK_SKV = 128
        num_warps = 4
    elif seq_kv <= 256:
        BLOCK_SKV = 256
        num_warps = 4
    elif seq_kv <= 512:
        BLOCK_SKV = 512
        num_warps = 8
    elif seq_kv <= 1024:
        BLOCK_SKV = 1024
        num_warps = 8
    elif seq_kv <= 2048:
        BLOCK_SKV = 2048
        num_warps = 16
    else:
        BLOCK_SKV = 512
        num_warps = 8

    fused_ds_kernel[grid](
        dO_bhsd, V, attn_weights, dropout_mask, dS,
        inv_scale,
        seq_q, seq_kv,
        dO_s_bs, dO_s_h, dO_s_sq,
        V_s_bs, V_s_h, V_s_skv,
        P_s_bs, P_s_h, P_s_sq,
        HEAD_DIM=128,
        BLOCK_SKV=BLOCK_SKV,
        num_warps=num_warps,
        num_stages=2,
    )

    # ----------------------------------------------------------------
    # Step 2: dV = P_drop_grouped^T @ dO_grouped  -> [bs, 8, skv, 128]
    # GQA-native: reshape P_drop to [bs, 8, 10*sq, skv], no separate sum needed!
    # cuBLAS: [bs, 8, skv, 10*sq] x [bs, 8, 10*sq, 128] -> [bs, 8, skv, 128]
    # ----------------------------------------------------------------
    dO_grouped = dO_bhsd.reshape(bs, n_kv_heads, n_groups * seq_q, HEAD_DIM)
    P_drop_grouped = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups * seq_q, seq_kv)

    dV = torch.matmul(P_drop_grouped.transpose(-2, -1), dO_grouped).to(torch.bfloat16)

    return dS, dV
