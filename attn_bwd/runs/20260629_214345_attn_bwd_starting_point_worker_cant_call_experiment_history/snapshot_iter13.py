"""
Optimized attention-backward kernel — fused Triton softmax-bwd with inline dO@V^T.

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
def _fused_dS_kernel(
    # dO: [bs, 80, sq, d]  bfloat16 (already transposed)
    dO_ptr,
    # value_states: [bs, 8, skv, d]  bfloat16
    VS_ptr,
    # attn_weights P: [bs, 80, sq, skv]  bfloat16
    P_ptr,
    # dropout_mask: [bs, 80, sq, skv]  bool
    mask_ptr,
    # output dS: [bs, 80, sq, skv]  bfloat16
    dS_ptr,
    # strides for dO [bs, 80, sq, d]
    dO_stride_bs, dO_stride_h, dO_stride_sq, dO_stride_d,
    # strides for VS [bs, 8, skv, d]
    vs_stride_bs, vs_stride_kv, vs_stride_skv, vs_stride_d,
    # strides for P / mask / dS [bs, 80, sq, skv]
    attn_stride_bs, attn_stride_h, attn_stride_sq, attn_stride_skv,
    # problem dimensions
    seq_q: tl.constexpr,
    seq_kv: tl.constexpr,
    n_kv_heads: tl.constexpr,
    n_groups: tl.constexpr,
    scale: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """
    One program per (bs, head, sq_row).
    Loads dO_row [HEAD_DIM] into registers once, then sweeps over seq_kv
    in tiles to compute dP = dO_row @ V^T, accumulate row_sum, then
    second sweep writes dS = P * (dP - row_sum). No dP_dropped materialized.
    """
    pid = tl.program_id(0)

    # Decode program ID -> (batch, head, sq_row)
    n_heads_total = n_kv_heads * n_groups  # 80
    total_per_batch = n_heads_total * seq_q
    bs_idx  = pid // total_per_batch
    rem     = pid % total_per_batch
    h_idx   = rem // seq_q
    sq_idx  = rem % seq_q

    kv_head_idx = h_idx // n_groups  # which KV head (0..7)

    # Base pointers for this row
    dO_row_ptr = (dO_ptr
                  + bs_idx * dO_stride_bs
                  + h_idx  * dO_stride_h
                  + sq_idx * dO_stride_sq)

    vs_base_ptr = (VS_ptr
                   + bs_idx    * vs_stride_bs
                   + kv_head_idx * vs_stride_kv)

    p_row_ptr = (P_ptr
                 + bs_idx * attn_stride_bs
                 + h_idx  * attn_stride_h
                 + sq_idx * attn_stride_sq)

    mask_row_ptr = (mask_ptr
                    + bs_idx * attn_stride_bs
                    + h_idx  * attn_stride_h
                    + sq_idx * attn_stride_sq)

    dS_row_ptr = (dS_ptr
                  + bs_idx * attn_stride_bs
                  + h_idx  * attn_stride_h
                  + sq_idx * attn_stride_sq)

    # Load dO_row into registers: [HEAD_DIM]
    d_offsets = tl.arange(0, HEAD_DIM)
    dO_row = tl.load(dO_row_ptr + d_offsets * dO_stride_d).to(tl.float32)  # [HEAD_DIM]

    # ── First pass: compute row_sum = sum_j(dP_j * P_j) ─────────────────────
    row_sum = tl.zeros([1], dtype=tl.float32)

    for kv_start in tl.range(0, seq_kv, BLOCK_SKV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_SKV)
        kv_mask = kv_offsets < seq_kv

        # Load V tile: [BLOCK_SKV, HEAD_DIM]
        # vs_base_ptr points to [skv, d] for this (bs, kv_head)
        v_tile_ptrs = (vs_base_ptr
                       + kv_offsets[:, None] * vs_stride_skv
                       + d_offsets[None, :] * vs_stride_d)
        v_tile = tl.load(v_tile_ptrs,
                         mask=kv_mask[:, None],
                         other=0.0).to(tl.float32)  # [BLOCK_SKV, HEAD_DIM]

        # dP_tile = dO_row @ V_tile^T  (dot product per kv position)
        # [BLOCK_SKV] = sum over d of dO_row[d] * V_tile[kv, d]
        dp_tile = tl.sum(dO_row[None, :] * v_tile, axis=1)  # [BLOCK_SKV]

        # Load dropout mask and apply scale
        m_tile = tl.load(mask_row_ptr + kv_offsets,
                         mask=kv_mask, other=0).to(tl.float32)
        dp_tile = dp_tile * m_tile * scale

        # Load P tile
        p_tile = tl.load(p_row_ptr + kv_offsets,
                         mask=kv_mask, other=0.0).to(tl.float32)

        row_sum += tl.sum(dp_tile * p_tile, axis=0)

    row_sum_val = tl.sum(row_sum, axis=0)

    # ── Second pass: compute dS = P * (dP - row_sum) and store ──────────────
    for kv_start in tl.range(0, seq_kv, BLOCK_SKV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_SKV)
        kv_mask = kv_offsets < seq_kv

        # Reload V tile
        v_tile_ptrs = (vs_base_ptr
                       + kv_offsets[:, None] * vs_stride_skv
                       + d_offsets[None, :] * vs_stride_d)
        v_tile = tl.load(v_tile_ptrs,
                         mask=kv_mask[:, None],
                         other=0.0).to(tl.float32)

        dp_tile = tl.sum(dO_row[None, :] * v_tile, axis=1)

        m_tile = tl.load(mask_row_ptr + kv_offsets,
                         mask=kv_mask, other=0).to(tl.float32)
        dp_tile = dp_tile * m_tile * scale

        p_tile = tl.load(p_row_ptr + kv_offsets,
                         mask=kv_mask, other=0.0).to(tl.float32)

        ds_tile = p_tile * (dp_tile - row_sum_val)

        tl.store(dS_row_ptr + kv_offsets,
                 ds_tile.to(tl.bfloat16),
                 mask=kv_mask)


def _compute_dV(dO_in, attn_weights_dropped, value_states,
                bs, n_kv, n_g, seq_q, seq_kv, d):
    """
    Compute dV using grouped BMM (cuBLAS).
    dO: [bs, sq, 80, d] -> [bs*8, 10*sq, d]
    attn_weights_dropped: [bs, 80, sq, skv] -> [bs*8, 10*sq, skv]
    dV: [bs, 8, skv, d]
    """
    # Transpose dO: [bs, sq, 80, d] -> [bs, 80, sq, d]
    dO = dO_in.transpose(1, 2).contiguous()  # [bs, 80, sq, d] bfloat16
    dO_reshaped = dO.reshape(bs * n_kv, n_g * seq_q, d)

    Pd_flat = attn_weights_dropped.reshape(bs * n_kv, n_g * seq_q, seq_kv)
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_reshaped)  # [bs*8, skv, d]
    dV = dV_flat.reshape(bs, n_kv, seq_kv, d)
    return dO, dV


_compute_dV_compiled = torch.compile(
    _compute_dV,
    mode="max-autotune",
    fullgraph=True,
)


def custom_kernel(data):
    (dO_in, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = dO_in.shape[0]
    seq_q  = dO_in.shape[1]
    seq_kv = value_states.shape[2]
    n_kv   = NUM_KEY_VALUE_HEADS   # 8
    n_g    = N_GROUPS              # 10
    d      = HEAD_DIM              # 128

    # Compute dV via BMM and also get contiguous dO [bs, 80, sq, d]
    dO_transposed, dV = _compute_dV_compiled(
        dO_in, attn_weights_dropped, value_states,
        bs, n_kv, n_g, seq_q, seq_kv, d
    )

    # ── Fused: compute dS without materializing dP_dropped ───────────────────
    # dO_transposed: [bs, 80, sq, d] bfloat16
    # value_states:  [bs, 8, skv, d] bfloat16
    # attn_weights:  [bs, 80, sq, skv] bfloat16
    # dropout_mask:  [bs, 80, sq, skv] bool
    # Output dS:     [bs, 80, sq, skv] bfloat16

    scale = 1.0 / (1.0 - attention_dropout)

    # Make sure inputs are contiguous
    dO_c   = dO_transposed.contiguous()
    P_c    = attn_weights.contiguous()
    mask_c = dropout_mask.contiguous()
    vs_c   = value_states.contiguous()

    dS = torch.empty_like(P_c)

    n_heads_total = NUM_ATTENTION_HEADS  # 80
    total_rows = bs * n_heads_total * seq_q

    # Choose BLOCK_SKV: power-of-2, at least 64, at most 256
    if seq_kv <= 64:
        BLOCK_SKV = 64
    elif seq_kv <= 128:
        BLOCK_SKV = 128
    elif seq_kv <= 256:
        BLOCK_SKV = 256
    else:
        BLOCK_SKV = 256

    grid = (total_rows,)

    _fused_dS_kernel[grid](
        dO_c, vs_c, P_c, mask_c, dS,
        # dO strides [bs, 80, sq, d]
        dO_c.stride(0), dO_c.stride(1), dO_c.stride(2), dO_c.stride(3),
        # VS strides [bs, 8, skv, d]
        vs_c.stride(0), vs_c.stride(1), vs_c.stride(2), vs_c.stride(3),
        # attn strides [bs, 80, sq, skv]
        P_c.stride(0), P_c.stride(1), P_c.stride(2), P_c.stride(3),
        seq_q=seq_q,
        seq_kv=seq_kv,
        n_kv_heads=n_kv,
        n_groups=n_g,
        scale=scale,
        HEAD_DIM=HEAD_DIM,
        BLOCK_SKV=BLOCK_SKV,
    )

    dV_out = dV.to(torch.bfloat16)

    return dS, dV_out
