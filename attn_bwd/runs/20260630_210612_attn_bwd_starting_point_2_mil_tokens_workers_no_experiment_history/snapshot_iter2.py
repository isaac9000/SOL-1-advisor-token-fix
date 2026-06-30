"""
Optimized attention-backward kernel using Triton fused kernels.

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
def fused_dV_kernel(
    # P_dropped: [bs, 80, sq, skv]
    P_dropped_ptr, stride_pd_bs, stride_pd_h, stride_pd_sq, stride_pd_skv,
    # dO: [bs, 80, sq, d]  (already transposed from [bs,sq,80,d])
    dO_ptr, stride_do_bs, stride_do_h, stride_do_sq, stride_do_d,
    # dV: [bs, 8, skv, d]
    dV_ptr, stride_dv_bs, stride_dv_kvh, stride_dv_skv, stride_dv_d,
    bs, n_kv_heads, n_groups, sq, skv, d,
    BLOCK_SKV: tl.constexpr, BLOCK_SQ: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    Compute dV = sum_{groups} P_dropped^T @ dO, with GQA group reduction fused.
    Grid: (bs * n_kv_heads, cdiv(skv, BLOCK_SKV), cdiv(d, BLOCK_D))
    """
    # Program IDs
    pid_bkv = tl.program_id(0)
    pid_skv = tl.program_id(1)
    pid_d   = tl.program_id(2)

    batch_idx  = pid_bkv // n_kv_heads
    kv_head    = pid_bkv % n_kv_heads

    # Tile offsets
    skv_start = pid_skv * BLOCK_SKV
    d_start   = pid_d   * BLOCK_D

    skv_offs = skv_start + tl.arange(0, BLOCK_SKV)
    d_offs   = d_start   + tl.arange(0, BLOCK_D)
    sq_offs  = tl.arange(0, BLOCK_SQ)

    skv_mask = skv_offs < skv
    d_mask   = d_offs   < d

    # Accumulator for dV tile [BLOCK_SKV, BLOCK_D]
    acc = tl.zeros((BLOCK_SKV, BLOCK_D), dtype=tl.float32)

    # Loop over all 10 query groups for this KV head
    for g in range(n_groups):
        q_head = kv_head * n_groups + g

        # Base pointers for this (batch, q_head)
        P_base = P_dropped_ptr + batch_idx * stride_pd_bs + q_head * stride_pd_h
        dO_base = dO_ptr + batch_idx * stride_do_bs + q_head * stride_do_h

        # Loop over sq dimension in tiles
        num_sq_blocks = tl.cdiv(sq, BLOCK_SQ)
        for sq_tile in range(num_sq_blocks):
            sq_start = sq_tile * BLOCK_SQ
            sq_tile_offs = sq_start + sq_offs
            sq_mask = sq_tile_offs < sq

            # Load P_dropped tile: [BLOCK_SKV, BLOCK_SQ] — we want [skv, sq]
            # P_dropped is [bs, h, sq, skv], so P^T[skv, sq]
            # We need P_dropped[sq, skv] transposed to [skv, sq]
            p_ptrs = P_base + sq_tile_offs[:, None] * stride_pd_sq + skv_offs[None, :] * stride_pd_skv
            # p_tile shape: [BLOCK_SQ, BLOCK_SKV]
            p_tile = tl.load(p_ptrs, mask=sq_mask[:, None] & skv_mask[None, :], other=0.0)
            p_tile = p_tile.to(tl.float32)

            # Load dO tile: [BLOCK_SQ, BLOCK_D]
            do_ptrs = dO_base + sq_tile_offs[:, None] * stride_do_sq + d_offs[None, :] * stride_do_d
            do_tile = tl.load(do_ptrs, mask=sq_mask[:, None] & d_mask[None, :], other=0.0)
            do_tile = do_tile.to(tl.float32)

            # acc += P^T @ dO = [BLOCK_SKV, BLOCK_SQ] @ [BLOCK_SQ, BLOCK_D]
            # p_tile is [BLOCK_SQ, BLOCK_SKV], so we need p_tile^T = [BLOCK_SKV, BLOCK_SQ]
            acc += tl.dot(tl.trans(p_tile), do_tile)

    # Write result
    dV_base = dV_ptr + batch_idx * stride_dv_bs + kv_head * stride_dv_kvh
    out_ptrs = dV_base + skv_offs[:, None] * stride_dv_skv + d_offs[None, :] * stride_dv_d
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=skv_mask[:, None] & d_mask[None, :])


@triton.jit
def fused_dS_kernel(
    # dO: [bs, 80, sq, d]
    dO_ptr, stride_do_bs, stride_do_h, stride_do_sq, stride_do_d,
    # V: [bs, 8, skv, d]  (original, before GQA expand)
    V_ptr, stride_v_bs, stride_v_kvh, stride_v_skv, stride_v_d,
    # P (attn_weights): [bs, 80, sq, skv]
    P_ptr, stride_p_bs, stride_p_h, stride_p_sq, stride_p_skv,
    # dropout_mask: [bs, 80, sq, skv]
    mask_ptr, stride_m_bs, stride_m_h, stride_m_sq, stride_m_skv,
    # dS output: [bs, 80, sq, skv]
    dS_ptr, stride_ds_bs, stride_ds_h, stride_ds_sq, stride_ds_skv,
    bs, n_heads, n_groups, sq, skv, d,
    inv_keep_prob: tl.constexpr,
    BLOCK_SQ: tl.constexpr, BLOCK_SKV: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    Fused: dP = dO @ V^T, apply dropout mask, compute softmax backward dS.
    Grid: (bs * n_heads, cdiv(sq, BLOCK_SQ))
    """
    pid_bh = tl.program_id(0)
    pid_sq = tl.program_id(1)

    batch_idx = pid_bh // n_heads
    head      = pid_bh % n_heads
    kv_head   = head // n_groups

    sq_start = pid_sq * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < sq

    d_offs   = tl.arange(0, BLOCK_D)
    skv_offs = tl.arange(0, BLOCK_SKV)

    dO_base = dO_ptr  + batch_idx * stride_do_bs + head    * stride_do_h
    V_base  = V_ptr   + batch_idx * stride_v_bs  + kv_head * stride_v_kvh
    P_base  = P_ptr   + batch_idx * stride_p_bs  + head    * stride_p_h
    M_base  = mask_ptr + batch_idx * stride_m_bs + head    * stride_m_h
    dS_base = dS_ptr  + batch_idx * stride_ds_bs + head    * stride_ds_h

    # We compute dP = dO @ V^T for this (batch, head) for sq_offs rows x all skv cols
    # Result shape: [BLOCK_SQ, skv] — we tile over skv

    num_skv_blocks = tl.cdiv(skv, BLOCK_SKV)

    # We need to compute rowsum(dP * P) across all skv for softmax bwd
    # Strategy: first pass computes dP tiles and rowsum, second pass writes dS
    # But since skv can be large, do a single pass storing dP and accumulating rowsum

    # Accumulate rowsum of dP*P: [BLOCK_SQ]
    rowsum = tl.zeros((BLOCK_SQ,), dtype=tl.float32)

    # We'll need to store dP temporarily — use a loop with two passes
    # Pass 1: compute dP tiles, accumulate rowsum
    # Pass 2: compute dS and write
    # Since we can't easily store large dP in registers, do two loops over skv

    # ----- Pass 1: compute rowsum(dP * P) -----
    for skv_tile in range(num_skv_blocks):
        skv_start = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask = skv_tile_offs < skv

        # Load dO: [BLOCK_SQ, BLOCK_D]
        do_ptrs = dO_base + sq_offs[:, None] * stride_do_sq + d_offs[None, :] * stride_do_d
        do_tile = tl.load(do_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)

        # Load V: [BLOCK_SKV, BLOCK_D]
        v_ptrs = V_base + skv_tile_offs[:, None] * stride_v_skv + d_offs[None, :] * stride_v_d
        v_tile = tl.load(v_ptrs, mask=skv_mask[:, None], other=0.0).to(tl.float32)

        # dP_tile = dO @ V^T: [BLOCK_SQ, BLOCK_SKV]
        dP_tile = tl.dot(do_tile, tl.trans(v_tile))

        # Load dropout mask and apply
        m_ptrs = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile = tl.load(m_ptrs, mask=sq_mask[:, None] & skv_mask[None, :], other=0)
        dP_tile = dP_tile * m_tile.to(tl.float32) * inv_keep_prob

        # Load P: [BLOCK_SQ, BLOCK_SKV]
        p_ptrs = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile = tl.load(p_ptrs, mask=sq_mask[:, None] & skv_mask[None, :], other=0.0).to(tl.float32)

        # Accumulate rowsum(dP * P)
        rowsum += tl.sum(dP_tile * p_tile, axis=1)

    # ----- Pass 2: compute dS and store -----
    for skv_tile in range(num_skv_blocks):
        skv_start = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask = skv_tile_offs < skv

        # Recompute dP_tile
        do_ptrs = dO_base + sq_offs[:, None] * stride_do_sq + d_offs[None, :] * stride_do_d
        do_tile = tl.load(do_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)

        v_ptrs = V_base + skv_tile_offs[:, None] * stride_v_skv + d_offs[None, :] * stride_v_d
        v_tile = tl.load(v_ptrs, mask=skv_mask[:, None], other=0.0).to(tl.float32)

        dP_tile = tl.dot(do_tile, tl.trans(v_tile))

        m_ptrs = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile = tl.load(m_ptrs, mask=sq_mask[:, None] & skv_mask[None, :], other=0)
        dP_tile = dP_tile * m_tile.to(tl.float32) * inv_keep_prob

        p_ptrs = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile = tl.load(p_ptrs, mask=sq_mask[:, None] & skv_mask[None, :], other=0.0).to(tl.float32)

        # dS = P * (dP - rowsum)
        dS_tile = p_tile * (dP_tile - rowsum[:, None])

        # Store
        ds_ptrs = dS_base + sq_offs[:, None] * stride_ds_sq + skv_tile_offs[None, :] * stride_ds_skv
        tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=sq_mask[:, None] & skv_mask[None, :])


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

    # Transpose dO: [bs, sq, 80, d] -> [bs, 80, sq, d], keep bfloat16
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()  # [bs, 80, sq, d]

    # ---- Compute dV using fused kernel ----
    dV = torch.empty((bs, n_kv_heads, seq_kv, d), dtype=torch.bfloat16, device=dO.device)

    # Make P_dropped contiguous
    P_dropped = attn_weights_dropped.contiguous()  # [bs, 80, sq, skv]

    BLOCK_SKV = 64
    BLOCK_SQ  = 32
    BLOCK_D   = 128  # d=128, fits in one block

    grid_dV = (bs * n_kv_heads,
               triton.cdiv(seq_kv, BLOCK_SKV),
               triton.cdiv(d, BLOCK_D))

    fused_dV_kernel[grid_dV](
        P_dropped, P_dropped.stride(0), P_dropped.stride(1), P_dropped.stride(2), P_dropped.stride(3),
        dO, dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dV, dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        bs, n_kv_heads, n_groups, seq_q, seq_kv, d,
        BLOCK_SKV=BLOCK_SKV, BLOCK_SQ=BLOCK_SQ, BLOCK_D=BLOCK_D,
    )

    # ---- Compute dS using fused kernel ----
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    P_attn = attn_weights.contiguous()      # [bs, 80, sq, skv]
    dmask  = dropout_mask.contiguous()      # [bs, 80, sq, skv]
    V_cont = value_states.contiguous()      # [bs, 8, skv, d]

    inv_keep = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    BLOCK_SQ_DS  = 16
    BLOCK_SKV_DS = 64
    BLOCK_D_DS   = 128

    grid_dS = (bs * n_heads,
               triton.cdiv(seq_q, BLOCK_SQ_DS))

    fused_dS_kernel[grid_dS](
        dO, dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        V_cont, V_cont.stride(0), V_cont.stride(1), V_cont.stride(2), V_cont.stride(3),
        P_attn, P_attn.stride(0), P_attn.stride(1), P_attn.stride(2), P_attn.stride(3),
        dmask, dmask.stride(0), dmask.stride(1), dmask.stride(2), dmask.stride(3),
        dS, dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
        bs, n_heads, n_groups, seq_q, seq_kv, d,
        inv_keep_prob=inv_keep,
        BLOCK_SQ=BLOCK_SQ_DS, BLOCK_SKV=BLOCK_SKV_DS, BLOCK_D=BLOCK_D_DS,
    )

    return dS, dV
