"""
Triton-fused attention-backward kernel.

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
# Kernel 1: Single-pass dS = P*(dP - row_sum) where dP = dO @ V^T
# row_sum is precomputed externally (via PyTorch bmm) and passed in.
# Each program handles one (bs_idx, head_idx, sq_tile) block
# Single pass over skv: load V, compute dP tile, apply dropout, compute dS, write.
# ---------------------------------------------------------------------------
@triton.jit
def attn_bwd_ds_kernel(
    # pointers
    dO_ptr,            # [bs, n_heads, sq, d]   float32
    V_ptr,             # [bs, 8, skv, d]         bfloat16  (KV heads, unexpanded)
    P_ptr,             # [bs, n_heads, sq, skv]  bfloat16
    mask_ptr,          # [bs, n_heads, sq, skv]  bool
    row_sum_ptr,       # [bs, n_heads, sq]        float32  (precomputed)
    dS_ptr,            # [bs, n_heads, sq, skv]  bfloat16  (output)
    # dims
    bs, n_heads, sq, skv,
    n_kv_heads, head_dim,
    inv_scale,         # 1/(1-dropout)
    # strides for dO [bs, n_heads, sq, d]
    dO_stride_bs, dO_stride_h, dO_stride_sq, dO_stride_d,
    # strides for V [bs, 8, skv, d]
    V_stride_bs, V_stride_h, V_stride_skv, V_stride_d,
    # strides for P [bs, n_heads, sq, skv]
    P_stride_bs, P_stride_h, P_stride_sq, P_stride_skv,
    # strides for mask [bs, n_heads, sq, skv]
    M_stride_bs, M_stride_h, M_stride_sq, M_stride_skv,
    # strides for row_sum [bs, n_heads, sq]
    RS_stride_bs, RS_stride_h, RS_stride_sq,
    # strides for dS [bs, n_heads, sq, skv]
    dS_stride_bs, dS_stride_h, dS_stride_sq, dS_stride_skv,
    # tile sizes
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    N_GROUPS: tl.constexpr,
):
    pid_bh = tl.program_id(0)   # batch * n_heads
    pid_sq = tl.program_id(1)   # sq tile index

    bs_idx = pid_bh // n_heads
    h_idx  = pid_bh % n_heads
    kv_idx = h_idx // N_GROUPS  # which KV head

    sq_start = pid_sq * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < sq

    d_offs = tl.arange(0, BLOCK_D)

    # Load dO tile [BLOCK_SQ, BLOCK_D]
    dO_base = bs_idx * dO_stride_bs + h_idx * dO_stride_h
    dO = tl.load(
        dO_ptr + dO_base
        + sq_offs[:, None] * dO_stride_sq
        + d_offs[None, :] * dO_stride_d,
        mask=sq_mask[:, None] & (d_offs[None, :] < head_dim),
        other=0.0,
    )  # float32 [BLOCK_SQ, BLOCK_D]

    # Load precomputed row_sum [BLOCK_SQ]
    RS_base = bs_idx * RS_stride_bs + h_idx * RS_stride_h
    row_sum = tl.load(
        row_sum_ptr + RS_base + sq_offs * RS_stride_sq,
        mask=sq_mask,
        other=0.0,
    )  # float32 [BLOCK_SQ]

    V_base  = bs_idx * V_stride_bs  + kv_idx * V_stride_h
    M_base  = bs_idx * M_stride_bs  + h_idx  * M_stride_h
    P_base  = bs_idx * P_stride_bs  + h_idx  * P_stride_h
    dS_base = bs_idx * dS_stride_bs + h_idx  * dS_stride_h

    skv_offs_base = tl.arange(0, BLOCK_SKV)

    # Single pass over skv
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_offs_base
        skv_mask = skv_offs < skv

        # Load V tile [BLOCK_SKV, BLOCK_D]
        V_tile = tl.load(
            V_ptr + V_base
            + skv_offs[:, None] * V_stride_skv
            + d_offs[None, :] * V_stride_d,
            mask=skv_mask[:, None] & (d_offs[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)  # [BLOCK_SKV, BLOCK_D]

        # dP_tile = dO @ V^T  [BLOCK_SQ, BLOCK_SKV]
        dP_tile = tl.dot(dO, tl.trans(V_tile))

        # Load dropout mask [BLOCK_SQ, BLOCK_SKV]
        drop_mask = tl.load(
            mask_ptr + M_base
            + sq_offs[:, None] * M_stride_sq
            + skv_offs[None, :] * M_stride_skv,
            mask=sq_mask[:, None] & skv_mask[None, :],
            other=0,
        )  # bool

        # Dropout backward
        dP_tile = dP_tile * drop_mask.to(tl.float32) * inv_scale

        # Load P tile [BLOCK_SQ, BLOCK_SKV]
        P_tile = tl.load(
            P_ptr + P_base
            + sq_offs[:, None] * P_stride_sq
            + skv_offs[None, :] * P_stride_skv,
            mask=sq_mask[:, None] & skv_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Softmax backward: dS = P * (dP - row_sum)
        dS_tile = P_tile * (dP_tile - row_sum[:, None])

        # Write dS
        tl.store(
            dS_ptr + dS_base
            + sq_offs[:, None] * dS_stride_sq
            + skv_offs[None, :] * dS_stride_skv,
            dS_tile.to(tl.bfloat16),
            mask=sq_mask[:, None] & skv_mask[None, :],
        )


# ---------------------------------------------------------------------------
# Kernel 2: Fused dV with GQA reduction
# Each program handles one (bs_idx, kv_head_idx, skv_tile) block
# Accumulates over 10 GQA groups and all sq positions
# ---------------------------------------------------------------------------
@triton.jit
def attn_bwd_dv_kernel(
    # pointers
    P_drop_ptr,        # [bs, n_heads, sq, skv]   bfloat16
    dO_ptr,            # [bs, n_heads, sq, d]      float32
    dV_ptr,            # [bs, 8, skv, d]            bfloat16  (output)
    # dims
    bs, n_heads, sq, skv, n_kv_heads, head_dim,
    # strides for P_drop [bs, n_heads, sq, skv]
    Pd_stride_bs, Pd_stride_h, Pd_stride_sq, Pd_stride_skv,
    # strides for dO [bs, n_heads, sq, d]
    dO_stride_bs, dO_stride_h, dO_stride_sq, dO_stride_d,
    # strides for dV [bs, 8, skv, d]
    dV_stride_bs, dV_stride_h, dV_stride_skv, dV_stride_d,
    # tile sizes
    BLOCK_SKV: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_D: tl.constexpr,
    N_GROUPS: tl.constexpr,
):
    pid_bkv = tl.program_id(0)   # batch * n_kv_heads
    pid_skv = tl.program_id(1)   # skv tile index

    bs_idx  = pid_bkv // n_kv_heads
    kv_idx  = pid_bkv % n_kv_heads

    skv_start = pid_skv * BLOCK_SKV
    skv_offs  = skv_start + tl.arange(0, BLOCK_SKV)
    skv_mask  = skv_offs < skv

    d_offs = tl.arange(0, BLOCK_D)
    sq_offs_base = tl.arange(0, BLOCK_SQ)

    dV_acc = tl.zeros([BLOCK_SKV, BLOCK_D], dtype=tl.float32)

    for sq_block in range(0, tl.cdiv(sq, BLOCK_SQ)):
        sq_offs = sq_block * BLOCK_SQ + sq_offs_base
        sq_mask = sq_offs < sq

        for g in range(0, N_GROUPS):
            h_idx = kv_idx * N_GROUPS + g

            Pd_base = bs_idx * Pd_stride_bs + h_idx * Pd_stride_h
            P_tile = tl.load(
                P_drop_ptr + Pd_base
                + sq_offs[:, None] * Pd_stride_sq
                + skv_offs[None, :] * Pd_stride_skv,
                mask=sq_mask[:, None] & skv_mask[None, :],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_SQ, BLOCK_SKV]

            dO_base = bs_idx * dO_stride_bs + h_idx * dO_stride_h
            dO_tile = tl.load(
                dO_ptr + dO_base
                + sq_offs[:, None] * dO_stride_sq
                + d_offs[None, :] * dO_stride_d,
                mask=sq_mask[:, None] & (d_offs[None, :] < head_dim),
                other=0.0,
            )  # float32 [BLOCK_SQ, BLOCK_D]

            dV_acc += tl.dot(tl.trans(P_tile), dO_tile)

    dV_base = bs_idx * dV_stride_bs + kv_idx * dV_stride_h
    tl.store(
        dV_ptr + dV_base
        + skv_offs[:, None] * dV_stride_skv
        + d_offs[None, :] * dV_stride_d,
        dV_acc.to(tl.bfloat16),
        mask=skv_mask[:, None] & (d_offs[None, :] < head_dim),
    )


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = N_GROUPS  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # Transpose dO to [bs, n_heads, sq, d] in float32 (contiguous)
    dO = grad_attn_output.transpose(1, 2).contiguous().to(torch.float32)

    # Ensure inputs are contiguous
    attn_weights_c         = attn_weights.contiguous()
    attn_weights_dropped_c = attn_weights_dropped.contiguous()
    value_states_c         = value_states.contiguous()
    dropout_mask_c         = dropout_mask.contiguous()

    inv_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # ---------------------------------------------------------------------------
    # Precompute row_sum = sum_skv(dP * P) using PyTorch BMM
    # This avoids a double-pass in the Triton kernel for dS.
    # dP = (dO @ V_expanded^T) * dropout_mask * inv_scale
    # row_sum = sum_skv(dP * P)  [bs, n_heads, sq]
    # ---------------------------------------------------------------------------
    # GQA expand value_states: [bs,8,skv,d] -> [bs,80,skv,d]
    vs_exp = value_states_c[:, :, None, :, :].expand(
        bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM
    ).reshape(bs, n_heads, seq_kv, HEAD_DIM)  # bfloat16

    # dP_dropped = dO @ vs_exp^T: [bs,80,sq,d] @ [bs,80,d,skv] -> [bs,80,sq,skv]
    dP_dropped = torch.bmm(
        dO.reshape(bs * n_heads, seq_q, HEAD_DIM),
        vs_exp.to(torch.float32).reshape(bs * n_heads, seq_kv, HEAD_DIM).transpose(1, 2)
    ).reshape(bs, n_heads, seq_q, seq_kv)  # float32

    # Apply dropout
    dP = dP_dropped * dropout_mask_c.to(torch.float32) * inv_scale

    # row_sum = sum_skv(dP * P) [bs, 80, sq]
    P_f32 = attn_weights_c.float()  # [bs,80,sq,skv]
    row_sum = (dP * P_f32).sum(dim=-1)  # [bs, 80, sq] float32

    # Make contiguous for Triton
    row_sum = row_sum.contiguous()

    # Output tensors
    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=dO.device)
    dV = torch.empty(bs, n_kv_heads, seq_kv, HEAD_DIM, dtype=torch.bfloat16, device=dO.device)

    # Tile sizes for kernel 1 (dS)
    BLOCK_SQ_DS  = 32
    BLOCK_SKV_DS = 128
    BLOCK_D_DS   = 128

    grid_ds = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ_DS))

    # Single-pass dS kernel using precomputed row_sum
    attn_bwd_ds_kernel[grid_ds](
        dO, value_states_c, attn_weights_c, dropout_mask_c, row_sum, dS,
        bs, n_heads, seq_q, seq_kv,
        n_kv_heads, HEAD_DIM,
        inv_scale,
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        value_states_c.stride(0), value_states_c.stride(1),
        value_states_c.stride(2), value_states_c.stride(3),
        attn_weights_c.stride(0), attn_weights_c.stride(1),
        attn_weights_c.stride(2), attn_weights_c.stride(3),
        dropout_mask_c.stride(0), dropout_mask_c.stride(1),
        dropout_mask_c.stride(2), dropout_mask_c.stride(3),
        row_sum.stride(0), row_sum.stride(1), row_sum.stride(2),
        dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
        BLOCK_SQ=BLOCK_SQ_DS,
        BLOCK_SKV=BLOCK_SKV_DS,
        BLOCK_D=BLOCK_D_DS,
        N_GROUPS=n_groups,
        num_warps=8,
        num_stages=3,
    )

    # Tile sizes for kernel 2 (dV)
    BLOCK_SKV_DV = 64
    BLOCK_SQ_DV  = 64
    BLOCK_D_DV   = 128

    grid_dv = (bs * n_kv_heads, triton.cdiv(seq_kv, BLOCK_SKV_DV))

    attn_bwd_dv_kernel[grid_dv](
        attn_weights_dropped_c, dO, dV,
        bs, n_heads, seq_q, seq_kv, n_kv_heads, HEAD_DIM,
        attn_weights_dropped_c.stride(0), attn_weights_dropped_c.stride(1),
        attn_weights_dropped_c.stride(2), attn_weights_dropped_c.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        BLOCK_SKV=BLOCK_SKV_DV,
        BLOCK_SQ=BLOCK_SQ_DV,
        BLOCK_D=BLOCK_D_DV,
        N_GROUPS=n_groups,
        num_warps=8,
        num_stages=3,
    )

    return dS, dV
