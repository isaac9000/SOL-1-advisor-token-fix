"""
Optimized attention-backward kernel using BF16 GEMMs + Triton softmax backward.

Strategy:
  1. Avoid materializing V_exp [bs, 80, skv, d] entirely.
     Instead reshape dO to [bs*8, 10, sq, d] and use expand+contiguous on V
     to [bs*8, 10, skv, d], then flatten to [bs*80, sq/skv, d] for bmm in BF16.

  2. For dP: bmm(dO_bf16, V^T_bf16) in BF16 — maximize B200 tensor core throughput.
     For dV: bmm(P_dropped^T_bf16, dO_bf16) in BF16.

  3. Triton softmax-backward kernel: reads BF16 dP, applies dropout + rowsum
     in float32 internally, stores BF16 dS.

  4. dV: bmm result is [bs*80, skv, d], reshape to [bs, 8, 10, skv, d] and sum dim=2.

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
    # dP: [bs, 80, sq, skv] bfloat16
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
    dP is now BF16 (from BF16 bmm).
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

        # Load dP tile (BF16) and cast to float32
        dp_ptrs = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
        dp_tile = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)  # bf16 -> f32

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
        dp_tile = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        m_ptrs = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile = tl.load(m_ptrs, mask=combined_mask, other=0)
        dp_masked = dp_tile * m_tile.to(tl.float32) * inv_keep_prob

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

    # Transpose dO: [bs, sq, 80, d] -> [bs, 80, sq, d], contiguous, keep BF16
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()  # [bs, 80, sq, d] bfloat16

    # ---- GEMM 1 (BF16): dP = dO @ V^T, avoiding V_exp materialization ----
    # Reshape dO: [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8*10, sq, d]
    dO_grouped = dO.view(bs, n_kv_heads, n_groups, seq_q, d)
    dO_bmm = dO_grouped.reshape(bs * n_kv_heads * n_groups, seq_q, d)  # [bs*80, sq, d]

    # V: [bs, 8, skv, d] -> expand to [bs, 8, 10, skv, d] (no copy) -> [bs*80, skv, d]
    # Use .expand() then .reshape() — expand is zero-copy, reshape after contiguous()
    V_grouped = value_states.unsqueeze(2).expand(bs, n_kv_heads, n_groups, seq_kv, d)
    V_bmm = V_grouped.reshape(bs * n_kv_heads * n_groups, seq_kv, d)  # triggers contiguous copy

    # BF16 bmm: dP = dO_bmm @ V_bmm^T => [bs*80, sq, skv]
    dP_raw = torch.bmm(dO_bmm, V_bmm.transpose(-2, -1))  # BF16, [bs*80, sq, skv]
    dP_raw = dP_raw.view(bs, n_heads, seq_q, seq_kv)  # [bs, 80, sq, skv] BF16

    # ---- GEMM 2 (BF16): dV = P_dropped^T @ dO ----
    # P_dropped: [bs, 80, sq, skv] -> [bs*80, sq, skv]
    P_dropped_bmm = attn_weights_dropped.reshape(bs * n_heads, seq_q, seq_kv)  # [bs*80, sq, skv]
    dO_bmm2 = dO.reshape(bs * n_heads, seq_q, d)  # [bs*80, sq, d]

    # BF16 bmm: dV_raw = P_dropped^T @ dO => [bs*80, skv, d]
    dV_raw = torch.bmm(P_dropped_bmm.transpose(-2, -1), dO_bmm2)  # BF16, [bs*80, skv, d]

    # Reshape and sum over groups: [bs*80, skv, d] -> [bs, 8, 10, skv, d] -> sum -> [bs, 8, skv, d]
    dV = dV_raw.view(bs, n_kv_heads, n_groups, seq_kv, d).sum(dim=2).to(torch.bfloat16)

    # ---- Triton kernel: softmax backward (reads BF16 dP_raw) ----
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

    return dS, dV
