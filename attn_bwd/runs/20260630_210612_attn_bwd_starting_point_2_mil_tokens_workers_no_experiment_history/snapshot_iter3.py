"""
Optimized attention-backward kernel using hybrid cuBLAS + Triton approach.

Strategy:
  1. Use torch.matmul (cuBLAS) for the two large GEMMs:
     - dP_raw = dO @ V_expanded^T   [bs, 80, sq, skv]
     - dV_raw = P_dropped^T @ dO    [bs, 80, skv, d]
  2. Use a lightweight Triton kernel for softmax backward:
     - fuse dropout application + rowsum + dS = P*(dP - rowsum)
     - two passes over skv (pass1: rowsum, pass2: store) — minimal memory
  3. Use a simple Triton kernel for GQA dV reduction:
     - sum dV_raw over the 10 groups to get [bs, 8, skv, d]

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
def softmax_bwd_kernel(
    # dP_raw: [bs, 80, sq, skv] float32
    dP_ptr, stride_dp_bs, stride_dp_h, stride_dp_sq, stride_dp_skv,
    # P (attn_weights): [bs, 80, sq, skv] bfloat16
    P_ptr, stride_p_bs, stride_p_h, stride_p_sq, stride_p_skv,
    # dropout_mask: [bs, 80, sq, skv] bool
    mask_ptr, stride_m_bs, stride_m_h, stride_m_sq, stride_m_skv,
    # dS output: [bs, 80, sq, skv] bfloat16
    dS_ptr, stride_ds_bs, stride_ds_h, stride_ds_sq, stride_ds_skv,
    bs, n_heads, sq, skv,
    inv_keep_prob: tl.constexpr,
    BLOCK_SQ: tl.constexpr, BLOCK_SKV: tl.constexpr,
):
    """
    Single-pass softmax backward with dropout application.
    Grid: (bs * n_heads, cdiv(sq, BLOCK_SQ))
    For each sq tile, iterate over all skv tiles twice:
      Pass 1: accumulate rowsum(dP_masked * P)
      Pass 2: compute and store dS = P * (dP_masked - rowsum)
    The dP values come already computed (from cuBLAS matmul).
    """
    pid_bh = tl.program_id(0)
    pid_sq = tl.program_id(1)

    batch_idx = pid_bh // n_heads
    head      = pid_bh % n_heads

    sq_start = pid_sq * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < sq

    skv_offs = tl.arange(0, BLOCK_SKV)

    dP_base = dP_ptr  + batch_idx * stride_dp_bs + head * stride_dp_h
    P_base  = P_ptr   + batch_idx * stride_p_bs  + head * stride_p_h
    M_base  = mask_ptr + batch_idx * stride_m_bs + head * stride_m_h
    dS_base = dS_ptr  + batch_idx * stride_ds_bs + head * stride_ds_h

    num_skv_blocks = tl.cdiv(skv, BLOCK_SKV)

    # ----- Pass 1: compute rowsum(dP_masked * P) -----
    rowsum = tl.zeros((BLOCK_SQ,), dtype=tl.float32)

    for skv_tile in range(num_skv_blocks):
        skv_start = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask = skv_tile_offs < skv

        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        # Load dP_raw tile: [BLOCK_SQ, BLOCK_SKV]
        dp_ptrs = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
        dp_tile = tl.load(dp_ptrs, mask=combined_mask, other=0.0)  # float32

        # Load dropout mask and apply
        m_ptrs = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile = tl.load(m_ptrs, mask=combined_mask, other=0)
        dp_masked = dp_tile * m_tile.to(tl.float32) * inv_keep_prob

        # Load P: [BLOCK_SQ, BLOCK_SKV]
        p_ptrs = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # Accumulate rowsum(dP_masked * P)
        rowsum += tl.sum(dp_masked * p_tile, axis=1)

    # ----- Pass 2: compute dS and store -----
    for skv_tile in range(num_skv_blocks):
        skv_start = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask = skv_tile_offs < skv

        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dp_ptrs = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
        dp_tile = tl.load(dp_ptrs, mask=combined_mask, other=0.0)

        m_ptrs = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile = tl.load(m_ptrs, mask=combined_mask, other=0)
        dp_masked = dp_tile * m_tile.to(tl.float32) * inv_keep_prob

        p_ptrs = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # dS = P * (dP_masked - rowsum)
        dS_tile = p_tile * (dp_masked - rowsum[:, None])

        ds_ptrs = dS_base + sq_offs[:, None] * stride_ds_sq + skv_tile_offs[None, :] * stride_ds_skv
        tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


@triton.jit
def gqa_dV_reduce_kernel(
    # dV_raw: [bs, 80, skv, d] float32
    dV_raw_ptr, stride_dvr_bs, stride_dvr_h, stride_dvr_skv, stride_dvr_d,
    # dV out: [bs, 8, skv, d] bfloat16
    dV_ptr, stride_dv_bs, stride_dv_kvh, stride_dv_skv, stride_dv_d,
    bs, n_kv_heads, n_groups, skv, d,
    BLOCK_SKV: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    Reduce dV_raw [bs, 80, skv, d] over groups to [bs, 8, skv, d].
    Grid: (bs * n_kv_heads, cdiv(skv, BLOCK_SKV), cdiv(d, BLOCK_D))
    """
    pid_bkv = tl.program_id(0)
    pid_skv = tl.program_id(1)
    pid_d   = tl.program_id(2)

    batch_idx = pid_bkv // n_kv_heads
    kv_head   = pid_bkv % n_kv_heads

    skv_start = pid_skv * BLOCK_SKV
    d_start   = pid_d   * BLOCK_D

    skv_offs = skv_start + tl.arange(0, BLOCK_SKV)
    d_offs   = d_start   + tl.arange(0, BLOCK_D)

    skv_mask = skv_offs < skv
    d_mask   = d_offs   < d
    combined_mask = skv_mask[:, None] & d_mask[None, :]

    acc = tl.zeros((BLOCK_SKV, BLOCK_D), dtype=tl.float32)

    for g in range(n_groups):
        q_head = kv_head * n_groups + g
        base = dV_raw_ptr + batch_idx * stride_dvr_bs + q_head * stride_dvr_h
        ptrs = base + skv_offs[:, None] * stride_dvr_skv + d_offs[None, :] * stride_dvr_d
        tile = tl.load(ptrs, mask=combined_mask, other=0.0)  # float32
        acc += tile

    dV_base = dV_ptr + batch_idx * stride_dv_bs + kv_head * stride_dv_kvh
    out_ptrs = dV_base + skv_offs[:, None] * stride_dv_skv + d_offs[None, :] * stride_dv_d
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=combined_mask)


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

    # Transpose dO: [bs, sq, 80, d] -> [bs, 80, sq, d], contiguous
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()  # [bs, 80, sq, d] bfloat16

    # ---- GEMM 1: dP_raw = dO @ V_expanded^T ----
    # V: [bs, 8, skv, d] -> expand to [bs, 80, skv, d]
    V_exp = value_states.reshape(bs, n_kv_heads, 1, seq_kv, d).expand(
        bs, n_kv_heads, n_groups, seq_kv, d
    ).reshape(bs, n_heads, seq_kv, d).contiguous()  # [bs, 80, skv, d]

    # dO: [bs, 80, sq, d], V_exp: [bs, 80, skv, d]
    # dP_raw = dO @ V_exp^T => [bs, 80, sq, skv]
    dO_f32 = dO.float()
    V_exp_f32 = V_exp.float()
    dP_raw = torch.matmul(dO_f32, V_exp_f32.transpose(-2, -1))  # [bs, 80, sq, skv] float32

    # ---- GEMM 2: dV_raw = P_dropped^T @ dO ----
    # P_dropped: [bs, 80, sq, skv], dO: [bs, 80, sq, d]
    # dV_raw = P_dropped^T @ dO => [bs, 80, skv, d]
    P_dropped_f32 = attn_weights_dropped.float()
    dV_raw = torch.matmul(P_dropped_f32.transpose(-2, -1), dO_f32)  # [bs, 80, skv, d] float32

    # ---- Triton kernel 1: softmax backward ----
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    P_attn = attn_weights.contiguous()   # [bs, 80, sq, skv] bfloat16
    dmask  = dropout_mask.contiguous()   # [bs, 80, sq, skv] bool

    inv_keep = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    BLOCK_SQ_DS  = 16
    BLOCK_SKV_DS = 64

    grid_dS = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ_DS))

    softmax_bwd_kernel[grid_dS](
        dP_raw, dP_raw.stride(0), dP_raw.stride(1), dP_raw.stride(2), dP_raw.stride(3),
        P_attn, P_attn.stride(0), P_attn.stride(1), P_attn.stride(2), P_attn.stride(3),
        dmask, dmask.stride(0), dmask.stride(1), dmask.stride(2), dmask.stride(3),
        dS, dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
        bs, n_heads, seq_q, seq_kv,
        inv_keep_prob=inv_keep,
        BLOCK_SQ=BLOCK_SQ_DS, BLOCK_SKV=BLOCK_SKV_DS,
    )

    # ---- Triton kernel 2: GQA dV reduction ----
    dV = torch.empty((bs, n_kv_heads, seq_kv, d), dtype=torch.bfloat16, device=dO.device)

    BLOCK_SKV = 64
    BLOCK_D   = 128

    grid_dV = (bs * n_kv_heads,
               triton.cdiv(seq_kv, BLOCK_SKV),
               triton.cdiv(d, BLOCK_D))

    gqa_dV_reduce_kernel[grid_dV](
        dV_raw, dV_raw.stride(0), dV_raw.stride(1), dV_raw.stride(2), dV_raw.stride(3),
        dV, dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        bs, n_kv_heads, n_groups, seq_kv, d,
        BLOCK_SKV=BLOCK_SKV, BLOCK_D=BLOCK_D,
    )

    return dS, dV
