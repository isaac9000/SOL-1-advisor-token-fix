"""
Attention backward: GQA-native cuBLAS batched GEMM for dV +
Fused Triton kernel for dS using tl.dot for tensor core acceleration.

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
# Fused kernel using tl.dot for tensor core acceleration:
# Each program handles BLOCK_SQ rows of one (bs, head80).
# Two-pass over skv tiles:
#   Pass 1: compute dP_dropped = dropout(dO @ V.T), accumulate row_sum per sq row
#   Pass 2: compute dS = P * (dP_dropped - row_sum), store bfloat16
#
# V is indexed with head//10 to handle GQA (8 KV heads, 10 groups each).
# ---------------------------------------------------------------------------
@triton.jit
def fused_ds_kernel(
    dO_ptr,      # [bs, 80, sq, 128]   bfloat16   (contiguous)
    V_ptr,       # [bs,  8, skv, 128]  bfloat16
    P_ptr,       # [bs, 80, sq, skv]   bfloat16
    mask_ptr,    # [bs, 80, sq, skv]   bool
    dS_ptr,      # [bs, 80, sq, skv]   bfloat16   (output)
    inv_scale,
    sq, skv,
    # dO strides: [bs, 80, sq, 128]
    dO_s_bs, dO_s_h, dO_s_sq,  # stride_d == 1
    # V strides: [bs, 8, skv, 128]
    V_s_bs, V_s_h, V_s_skv,    # stride_d == 1
    # P/mask/dS strides: [bs, 80, sq, skv]
    P_s_bs, P_s_h, P_s_sq,     # stride_skv == 1
    HEAD_DIM: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """
    Each program handles BLOCK_SQ rows of one (bs, head80).
    Uses tl.dot for tensor-core acceleration: [BLOCK_SQ, HEAD_DIM] @ [HEAD_DIM, BLOCK_SKV].
    Two-pass over skv tiles.
    """
    n_heads = 80

    pid = tl.program_id(0)
    # pid encodes (bs_idx, h_idx, sq_block_idx)
    n_sq_blocks = tl.cdiv(sq, BLOCK_SQ)
    sq_block_idx = pid % n_sq_blocks
    bh_idx = pid // n_sq_blocks
    h_idx = bh_idx % n_heads
    bs_idx = bh_idx // n_heads
    kv_h_idx = h_idx // 10  # GQA: map head80 -> KV head (0..7)

    sq_start = sq_block_idx * BLOCK_SQ
    sq_offs = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask = sq_offs < sq
    d_offs = tl.arange(0, HEAD_DIM)

    # Load dO block [BLOCK_SQ, HEAD_DIM] once
    dO_base = bs_idx * dO_s_bs + h_idx * dO_s_h
    dO_block = tl.load(
        dO_ptr + dO_base + sq_offs[:, None] * dO_s_sq + d_offs[None, :],
        mask=sq_mask[:, None],
        other=0.0,
    ).to(tl.float32)  # [BLOCK_SQ, HEAD_DIM]

    V_base = bs_idx * V_s_bs + kv_h_idx * V_s_h
    P_base = bs_idx * P_s_bs + h_idx * P_s_h

    skv_arange = tl.arange(0, BLOCK_SKV)

    # Accumulate row_sum [BLOCK_SQ]
    row_sum = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    # Pass 1: stream V and P to get row_sum
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        # Load V tile [BLOCK_SKV, HEAD_DIM]
        V_tile = tl.load(
            V_ptr + V_base + skv_offs[:, None] * V_s_skv + d_offs[None, :],
            mask=skv_mask[:, None],
            other=0.0,
        ).to(tl.float32)  # [BLOCK_SKV, HEAD_DIM]

        # dP_raw = dO @ V.T: [BLOCK_SQ, HEAD_DIM] x [HEAD_DIM, BLOCK_SKV] -> [BLOCK_SQ, BLOCK_SKV]
        dP_tile = tl.dot(dO_block, tl.trans(V_tile))  # [BLOCK_SQ, BLOCK_SKV]

        # Load dropout mask [BLOCK_SQ, BLOCK_SKV]
        drop = tl.load(
            mask_ptr + P_base + sq_offs[:, None] * P_s_sq + skv_offs[None, :],
            mask=sq_mask[:, None] & skv_mask[None, :],
            other=0,
        )
        dP_dropped = dP_tile * drop.to(tl.float32) * inv_scale

        # Load P tile [BLOCK_SQ, BLOCK_SKV]
        P_tile = tl.load(
            P_ptr + P_base + sq_offs[:, None] * P_s_sq + skv_offs[None, :],
            mask=sq_mask[:, None] & skv_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        row_sum += tl.sum(dP_dropped * P_tile, axis=1)  # [BLOCK_SQ]

    # Pass 2: recompute dP and store dS
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        # Reload V tile
        V_tile = tl.load(
            V_ptr + V_base + skv_offs[:, None] * V_s_skv + d_offs[None, :],
            mask=skv_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        dP_tile = tl.dot(dO_block, tl.trans(V_tile))  # [BLOCK_SQ, BLOCK_SKV]

        drop = tl.load(
            mask_ptr + P_base + sq_offs[:, None] * P_s_sq + skv_offs[None, :],
            mask=sq_mask[:, None] & skv_mask[None, :],
            other=0,
        )
        dP_dropped = dP_tile * drop.to(tl.float32) * inv_scale

        P_tile = tl.load(
            P_ptr + P_base + sq_offs[:, None] * P_s_sq + skv_offs[None, :],
            mask=sq_mask[:, None] & skv_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # dS = P * (dP_dropped - row_sum[:, None])
        dS_tile = P_tile * (dP_dropped - row_sum[:, None])

        tl.store(
            dS_ptr + P_base + sq_offs[:, None] * P_s_sq + skv_offs[None, :],
            dS_tile.to(tl.bfloat16),
            mask=sq_mask[:, None] & skv_mask[None, :],
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
    device = grad_attn_output.device

    inv_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # ----------------------------------------------------------------
    # Step 1: Compute dS via fused Triton kernel (tl.dot based)
    # dO input: [bs, sq, 80, 128] -> [bs, 80, sq, 128] contiguous
    # ----------------------------------------------------------------
    dO_bhsd = grad_attn_output.permute(0, 2, 1, 3).contiguous()  # [bs, 80, sq, 128] bf16

    V = value_states  # [bs, 8, skv, 128] bf16

    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=device)

    # Strides for dO [bs, 80, sq, 128]
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

    # Choose BLOCK_SQ and BLOCK_SKV based on seq sizes
    # BLOCK_SQ groups sq rows together for tl.dot efficiency
    # tl.dot requires at least 16x16 blocks
    BLOCK_SQ = 16
    if seq_kv <= 256:
        BLOCK_SKV = 256
        num_warps = 4
        num_stages = 3
    elif seq_kv <= 512:
        BLOCK_SKV = 256
        num_warps = 4
        num_stages = 3
    elif seq_kv <= 1024:
        BLOCK_SKV = 256
        num_warps = 4
        num_stages = 3
    else:
        BLOCK_SKV = 256
        num_warps = 4
        num_stages = 3

    n_sq_blocks = (seq_q + BLOCK_SQ - 1) // BLOCK_SQ
    grid = (bs * n_heads * n_sq_blocks,)

    fused_ds_kernel[grid](
        dO_bhsd, V, attn_weights, dropout_mask, dS,
        inv_scale,
        seq_q, seq_kv,
        dO_s_bs, dO_s_h, dO_s_sq,
        V_s_bs, V_s_h, V_s_skv,
        P_s_bs, P_s_h, P_s_sq,
        HEAD_DIM=128,
        BLOCK_SQ=BLOCK_SQ,
        BLOCK_SKV=BLOCK_SKV,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # ----------------------------------------------------------------
    # Step 2: dV = P_drop_grouped^T @ dO_grouped  -> [bs, 8, skv, 128]
    # GQA-native: reshape to [bs, 8, 10*sq, skv] x [bs, 8, 10*sq, 128]
    # ----------------------------------------------------------------
    dO_grouped = dO_bhsd.reshape(bs, n_kv_heads, n_groups * seq_q, HEAD_DIM)
    P_drop_grouped = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups * seq_q, seq_kv)

    dV = torch.matmul(P_drop_grouped.transpose(-2, -1), dO_grouped).to(torch.bfloat16)

    return dS, dV
