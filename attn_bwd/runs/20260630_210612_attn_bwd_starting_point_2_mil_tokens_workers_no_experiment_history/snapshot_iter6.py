"""
Optimized attention-backward kernel using BF16 GEMMs + fused dP+softmax-bwd Triton kernel.

Strategy:
  1. FUSED dP + softmax backward Triton kernel:
     For each (batch, KV-head, sq_tile) block, iterate over skv tiles to:
       - Compute dP tiles on-the-fly: dP_tile = dO_tile @ V^T_tile  (tl.dot)
       - Apply dropout mask and accumulate rowsum in one pass
       - Second pass: recompute dP_tile (V hot in L2), write dS
     This eliminates the large [bs, 80, sq, skv] dP intermediate tensor entirely.

  2. GQA-aware mapping: each kernel block handles one KV head, iterating over
     all 10 query heads in that group. V is shared across the 10 query heads.

  3. For dV: GQA-aware batched matmul (P_dropped_gqa^T @ dO_gqa) stays as-is
     since it's already clean and efficient.

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


@triton.jit
def fused_softmax_bwd_kernel(
    # dO: [bs, 80, sq, d] bfloat16
    dO_ptr, stride_do_bs, stride_do_h, stride_do_sq, stride_do_d,
    # V: [bs, 8, skv, d] bfloat16  (KV heads, not expanded)
    V_ptr, stride_v_bs, stride_v_kv, stride_v_skv, stride_v_d,
    # P (attn_weights): [bs, 80, sq, skv] bfloat16
    P_ptr, stride_p_bs, stride_p_h, stride_p_sq, stride_p_skv,
    # dropout_mask: [bs, 80, seq_q, seq_kv] bool
    mask_ptr, stride_m_bs, stride_m_h, stride_m_sq, stride_m_skv,
    # dS output: [bs, 80, sq, skv] bfloat16
    dS_ptr, stride_ds_bs, stride_ds_h, stride_ds_sq, stride_ds_skv,
    bs, n_heads, n_kv_heads, sq, skv, d,
    n_groups: tl.constexpr,
    inv_keep_prob: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused kernel: computes dP = dO @ V^T on-the-fly and applies softmax backward.
    Grid: (bs * n_heads, cdiv(sq, BLOCK_SQ))
    Each program handles one (batch, head, sq_tile) block.
    V is addressed via KV head (head // n_groups).
    """
    pid_bh = tl.program_id(0)
    pid_sq = tl.program_id(1)

    batch_idx = pid_bh // n_heads
    head      = pid_bh % n_heads
    kv_head   = head // n_groups  # maps query head -> KV head

    sq_start = pid_sq * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < sq

    skv_offs = tl.arange(0, BLOCK_SKV)
    d_offs   = tl.arange(0, BLOCK_D)

    # Base pointers for dO row [sq_offs, :] for this (batch, head)
    dO_base = dO_ptr + batch_idx * stride_do_bs + head * stride_do_h
    # Base pointers for V [kv_head, :, :] for this batch
    V_base  = V_ptr  + batch_idx * stride_v_bs  + kv_head * stride_v_kv
    # Base pointers for P, mask, dS
    P_base  = P_ptr   + batch_idx * stride_p_bs  + head * stride_p_h
    M_base  = mask_ptr + batch_idx * stride_m_bs + head * stride_m_h
    dS_base = dS_ptr  + batch_idx * stride_ds_bs + head * stride_ds_h

    num_skv_blocks = tl.cdiv(skv, BLOCK_SKV)
    num_d_blocks   = tl.cdiv(d,   BLOCK_D)

    # ----- Load dO tile: [BLOCK_SQ, d] -----
    # We load in BLOCK_D chunks if needed, but for HEAD_DIM=128 we can do it at once
    # if BLOCK_D == d. For generality, we accumulate over d blocks.

    # ----- Pass 1: compute rowsum(dP_masked * P) -----
    rowsum = tl.zeros((BLOCK_SQ,), dtype=tl.float32)

    for skv_tile in range(num_skv_blocks):
        skv_start_t = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start_t + skv_offs
        skv_mask = skv_tile_offs < skv
        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        # Compute dP_tile = dO[sq_offs, :] @ V[skv_tile_offs, :]^T  via block dot products
        # dP_tile: [BLOCK_SQ, BLOCK_SKV] in float32
        dP_tile = tl.zeros((BLOCK_SQ, BLOCK_SKV), dtype=tl.float32)

        for d_tile in range(num_d_blocks):
            d_start = d_tile * BLOCK_D
            d_tile_offs = d_start + d_offs
            d_mask = d_tile_offs < d

            # Load dO tile: [BLOCK_SQ, BLOCK_D]
            do_ptrs = dO_base + sq_offs[:, None] * stride_do_sq + d_tile_offs[None, :] * stride_do_d
            do_tile = tl.load(do_ptrs, mask=sq_mask[:, None] & d_mask[None, :], other=0.0)

            # Load V tile: [BLOCK_SKV, BLOCK_D]
            v_ptrs = V_base + skv_tile_offs[:, None] * stride_v_skv + d_tile_offs[None, :] * stride_v_d
            v_tile = tl.load(v_ptrs, mask=skv_mask[:, None] & d_mask[None, :], other=0.0)

            # Accumulate: dP_tile += dO_tile @ v_tile^T
            dP_tile = tl.dot(do_tile, tl.trans(v_tile), acc=dP_tile)

        # Load dropout mask and apply
        m_ptrs = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile = tl.load(m_ptrs, mask=combined_mask, other=0)
        dp_masked = dP_tile * m_tile.to(tl.float32) * inv_keep_prob

        # Load P tile
        p_ptrs = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # Accumulate rowsum
        rowsum += tl.sum(dp_masked * p_tile, axis=1)

    # ----- Pass 2: recompute dP_tile (V hot in L2), write dS -----
    for skv_tile in range(num_skv_blocks):
        skv_start_t = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start_t + skv_offs
        skv_mask = skv_tile_offs < skv
        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        # Recompute dP_tile
        dP_tile = tl.zeros((BLOCK_SQ, BLOCK_SKV), dtype=tl.float32)

        for d_tile in range(num_d_blocks):
            d_start = d_tile * BLOCK_D
            d_tile_offs = d_start + d_offs
            d_mask = d_tile_offs < d

            do_ptrs = dO_base + sq_offs[:, None] * stride_do_sq + d_tile_offs[None, :] * stride_do_d
            do_tile = tl.load(do_ptrs, mask=sq_mask[:, None] & d_mask[None, :], other=0.0)

            v_ptrs = V_base + skv_tile_offs[:, None] * stride_v_skv + d_tile_offs[None, :] * stride_v_d
            v_tile = tl.load(v_ptrs, mask=skv_mask[:, None] & d_mask[None, :], other=0.0)

            dP_tile = tl.dot(do_tile, tl.trans(v_tile), acc=dP_tile)

        # Load dropout mask and apply
        m_ptrs = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile = tl.load(m_ptrs, mask=combined_mask, other=0)
        dp_masked = dP_tile * m_tile.to(tl.float32) * inv_keep_prob

        # Load P tile
        p_ptrs = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # dS = P * (dP_masked - rowsum)
        dS_tile = p_tile * (dp_masked - rowsum[:, None])

        ds_ptrs = dS_base + sq_offs[:, None] * stride_ds_sq + skv_tile_offs[None, :] * stride_ds_skv
        tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = N_GROUPS  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    d      = HEAD_DIM

    # Transpose dO: [bs, sq, 80, d] -> [bs, 80, sq, d] contiguous, keep BF16
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()  # [bs, 80, sq, d] bfloat16

    # ---- GEMM 2 (BF16): dV = P_dropped_gqa^T @ dO_gqa ----
    # Reshape dO for GQA-aware GEMMs: [bs, 80, sq, d] -> [bs*8, 10*sq, d]
    dO_grouped = dO.view(bs, n_kv_heads, n_groups, seq_q, d)
    dO_gqa = dO_grouped.reshape(bs * n_kv_heads, n_groups * seq_q, d)  # [bs*8, 10*sq, d]

    # P_dropped: [bs, 80, sq, skv] -> [bs*8, 10*sq, skv]
    P_dropped_gqa = attn_weights_dropped.view(bs, n_kv_heads, n_groups, seq_q, seq_kv) \
                                        .reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
    V_gqa = value_states.reshape(bs * n_kv_heads, seq_kv, d)  # [bs*8, skv, d]
    dV_gqa = torch.bmm(P_dropped_gqa.transpose(-2, -1), dO_gqa)  # [bs*8, skv, d] BF16
    dV = dV_gqa.view(bs, n_kv_heads, seq_kv, d).to(torch.bfloat16)

    # ---- Fused Triton kernel: dP = dO @ V^T + softmax backward ----
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    P_attn = attn_weights.contiguous()   # [bs, 80, sq, skv] bfloat16
    dmask  = dropout_mask.contiguous()   # [bs, 80, sq, skv] bool
    V_cont = value_states.contiguous()   # [bs, 8, skv, d] bfloat16

    inv_keep = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Tile sizes: BLOCK_SQ * BLOCK_SKV accumulation, BLOCK_D for the dot product
    # HEAD_DIM=128, use BLOCK_D=128 (fits in registers for BF16 matmul)
    # BLOCK_SQ=16, BLOCK_SKV=64 for good occupancy
    BLOCK_SQ_F  = 16
    BLOCK_SKV_F = 64
    BLOCK_D_F   = 128  # HEAD_DIM, load entire head dim at once

    grid_fused = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ_F))

    fused_softmax_bwd_kernel[grid_fused](
        dO, dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        V_cont, V_cont.stride(0), V_cont.stride(1), V_cont.stride(2), V_cont.stride(3),
        P_attn, P_attn.stride(0), P_attn.stride(1), P_attn.stride(2), P_attn.stride(3),
        dmask,  dmask.stride(0),  dmask.stride(1),  dmask.stride(2),  dmask.stride(3),
        dS,     dS.stride(0),     dS.stride(1),     dS.stride(2),     dS.stride(3),
        bs, n_heads, n_kv_heads, seq_q, seq_kv, d,
        n_groups=n_groups,
        inv_keep_prob=inv_keep,
        BLOCK_SQ=BLOCK_SQ_F, BLOCK_SKV=BLOCK_SKV_F, BLOCK_D=BLOCK_D_F,
    )

    return dS, dV
