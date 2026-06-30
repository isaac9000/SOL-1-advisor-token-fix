"""
Attention backward: Fused BMM+softmax-backward Triton kernel (no dP_raw materialization)
+ GQA-native cuBLAS batched GEMM for dV.

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
# Fused kernel: dO @ V^T -> dP -> apply dropout mask -> softmax backward -> dS
#
# Each program handles ONE (bs, head, sq_row) triple.
# We sweep over skv in blocks, computing dot products in registers.
# Two-pass approach: pass 1 computes row_sum, pass 2 writes dS.
# V is accessed in GQA mode: kv_head = head // 10
# ---------------------------------------------------------------------------
@triton.jit
def fused_bwd_dS_kernel(
    dO_ptr,      # [bs, 80, sq, 128]  bfloat16  (contiguous, transposed from input)
    V_ptr,       # [bs,  8, skv, 128] bfloat16
    P_ptr,       # [bs, 80, sq, skv]  bfloat16
    mask_ptr,    # [bs, 80, sq, skv]  bool
    dS_ptr,      # [bs, 80, sq, skv]  bfloat16  (output)
    inv_scale,   # scalar float32
    bs, n_heads, n_kv_heads, n_groups,
    sq, skv, head_dim,
    # strides for dO [bs, 80, sq, 128]
    dO_stride_bs, dO_stride_h, dO_stride_sq, dO_stride_d,
    # strides for V [bs, 8, skv, 128]
    V_stride_bs, V_stride_kvh, V_stride_skv, V_stride_d,
    # strides for P, mask, dS [bs, 80, sq, skv]
    S_stride_bs, S_stride_h, S_stride_sq, S_stride_skv,
    BLOCK_SKV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Each program handles one row: (bs_idx, head_idx, sq_row).
    Pass 1: sweep skv blocks, compute dP = dO @ V[kv_head]^T, apply dropout,
            accumulate row_sum = sum(dP * P).
    Pass 2: sweep skv blocks again, compute dS = P*(dP - row_sum), store.
    """
    pid = tl.program_id(0)

    # Decode (bs, head, sq) from flat pid
    n_rows_per_bs = n_heads * sq
    bs_idx  = pid // n_rows_per_bs
    rem     = pid % n_rows_per_bs
    h_idx   = rem // sq
    sq_idx  = rem % sq

    kv_idx  = h_idx // n_groups   # GQA: which KV head

    # Base pointer offsets for this row
    dO_base = bs_idx * dO_stride_bs + h_idx * dO_stride_h + sq_idx * dO_stride_sq
    V_base  = bs_idx * V_stride_bs  + kv_idx * V_stride_kvh
    S_base  = bs_idx * S_stride_bs  + h_idx  * S_stride_h  + sq_idx * S_stride_sq

    # Load the dO row: [HEAD_DIM]
    d_offs  = tl.arange(0, HEAD_DIM)
    dO_row  = tl.load(dO_ptr + dO_base + d_offs * dO_stride_d).to(tl.float32)
    # dO_row: [HEAD_DIM]  float32

    skv_arange = tl.arange(0, BLOCK_SKV)

    # ---- Pass 1: accumulate row_sum ----
    row_sum = 0.0

    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        # Load V block: [BLOCK_SKV, HEAD_DIM]
        V_tile = tl.load(
            V_ptr + V_base + skv_offs[:, None] * V_stride_skv + d_offs[None, :] * V_stride_d,
            mask=skv_mask[:, None],
            other=0.0,
        ).to(tl.float32)  # [BLOCK_SKV, HEAD_DIM]

        # dP_tile = dO_row @ V_tile^T = V_tile @ dO_row  (dot products)
        # Each element: dot(dO_row, V_tile[j,:]) for j in BLOCK_SKV
        dP_tile = tl.sum(V_tile * dO_row[None, :], axis=1)  # [BLOCK_SKV]

        # Apply dropout mask and scaling
        drop = tl.load(
            mask_ptr + S_base + skv_offs * S_stride_skv,
            mask=skv_mask, other=0,
        )
        dP_tile = dP_tile * drop.to(tl.float32) * inv_scale

        # Load P tile
        P_tile = tl.load(
            P_ptr + S_base + skv_offs * S_stride_skv,
            mask=skv_mask, other=0.0,
        ).to(tl.float32)

        row_sum = row_sum + tl.sum(dP_tile * P_tile, axis=0)

    # ---- Pass 2: compute and store dS ----
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        # Recompute V block
        V_tile = tl.load(
            V_ptr + V_base + skv_offs[:, None] * V_stride_skv + d_offs[None, :] * V_stride_d,
            mask=skv_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        dP_tile = tl.sum(V_tile * dO_row[None, :], axis=1)

        drop = tl.load(
            mask_ptr + S_base + skv_offs * S_stride_skv,
            mask=skv_mask, other=0,
        )
        dP_tile = dP_tile * drop.to(tl.float32) * inv_scale

        P_tile = tl.load(
            P_ptr + S_base + skv_offs * S_stride_skv,
            mask=skv_mask, other=0.0,
        ).to(tl.float32)

        dS_tile = P_tile * (dP_tile - row_sum)

        tl.store(
            dS_ptr + S_base + skv_offs * S_stride_skv,
            dS_tile.to(tl.bfloat16),
            mask=skv_mask,
        )


# Single-pass variant for small skv (fits in BLOCK_SKV registers)
@triton.jit
def fused_bwd_dS_kernel_single(
    dO_ptr,
    V_ptr,
    P_ptr,
    mask_ptr,
    dS_ptr,
    inv_scale,
    bs, n_heads, n_kv_heads, n_groups,
    sq, skv, head_dim,
    dO_stride_bs, dO_stride_h, dO_stride_sq, dO_stride_d,
    V_stride_bs, V_stride_kvh, V_stride_skv, V_stride_d,
    S_stride_bs, S_stride_h, S_stride_sq, S_stride_skv,
    BLOCK_SKV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Single-pass variant: load all skv at once when BLOCK_SKV >= skv.
    """
    pid = tl.program_id(0)

    n_rows_per_bs = n_heads * sq
    bs_idx  = pid // n_rows_per_bs
    rem     = pid % n_rows_per_bs
    h_idx   = rem // sq
    sq_idx  = rem % sq
    kv_idx  = h_idx // n_groups

    dO_base = bs_idx * dO_stride_bs + h_idx * dO_stride_h + sq_idx * dO_stride_sq
    V_base  = bs_idx * V_stride_bs  + kv_idx * V_stride_kvh
    S_base  = bs_idx * S_stride_bs  + h_idx  * S_stride_h  + sq_idx * S_stride_sq

    d_offs     = tl.arange(0, HEAD_DIM)
    skv_offs   = tl.arange(0, BLOCK_SKV)
    skv_mask   = skv_offs < skv

    # Load dO row
    dO_row = tl.load(dO_ptr + dO_base + d_offs * dO_stride_d).to(tl.float32)

    # Load V tile [BLOCK_SKV, HEAD_DIM]
    V_tile = tl.load(
        V_ptr + V_base + skv_offs[:, None] * V_stride_skv + d_offs[None, :] * V_stride_d,
        mask=skv_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    # dP = V_tile @ dO_row
    dP_tile = tl.sum(V_tile * dO_row[None, :], axis=1)

    # Apply dropout
    drop = tl.load(
        mask_ptr + S_base + skv_offs * S_stride_skv,
        mask=skv_mask, other=0,
    )
    dP_tile = dP_tile * drop.to(tl.float32) * inv_scale

    # Load P tile
    P_tile = tl.load(
        P_ptr + S_base + skv_offs * S_stride_skv,
        mask=skv_mask, other=0.0,
    ).to(tl.float32)

    # row_sum and dS in one pass
    row_sum = tl.sum(dP_tile * P_tile, axis=0)
    dS_tile = P_tile * (dP_tile - row_sum)

    tl.store(
        dS_ptr + S_base + skv_offs * S_stride_skv,
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

    device = grad_attn_output.device

    # ----------------------------------------------------------------
    # Step 1: Transpose dO from [bs, sq, 80, 128] -> [bs, 80, sq, 128]
    # contiguous so the fused kernel can stride into it cleanly
    # ----------------------------------------------------------------
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()
    # dO: [bs, 80, sq, 128]  bfloat16

    # ----------------------------------------------------------------
    # Step 2: Fused kernel: dO @ V^T (GQA) + dropout + softmax bwd -> dS
    # Eliminates the large float32 dP_raw intermediate tensor
    # ----------------------------------------------------------------
    inv_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    P    = attn_weights        # [bs, 80, sq, skv] bfloat16
    mask = dropout_mask        # [bs, 80, sq, skv] bool
    V    = value_states        # [bs,  8, skv, 128] bfloat16

    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=device)

    grid = (bs * n_heads * seq_q,)

    HEAD_DIM_C = HEAD_DIM  # constexpr 128

    # Choose BLOCK_SKV based on seq_kv
    if seq_kv <= 256:
        BLOCK_SKV = 256
        single_pass = True
        num_warps = 4
    elif seq_kv <= 512:
        BLOCK_SKV = 512
        single_pass = True
        num_warps = 8
    elif seq_kv <= 1024:
        BLOCK_SKV = 1024
        single_pass = True
        num_warps = 16
    else:
        BLOCK_SKV = 512
        single_pass = False
        num_warps = 8

    if single_pass:
        fused_bwd_dS_kernel_single[grid](
            dO, V, P, mask, dS,
            inv_scale,
            bs, n_heads, n_kv_heads, n_groups,
            seq_q, seq_kv, HEAD_DIM_C,
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            V.stride(0),  V.stride(1),  V.stride(2),  V.stride(3),
            P.stride(0),  P.stride(1),  P.stride(2),  P.stride(3),
            BLOCK_SKV=BLOCK_SKV,
            HEAD_DIM=HEAD_DIM_C,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        fused_bwd_dS_kernel[grid](
            dO, V, P, mask, dS,
            inv_scale,
            bs, n_heads, n_kv_heads, n_groups,
            seq_q, seq_kv, HEAD_DIM_C,
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            V.stride(0),  V.stride(1),  V.stride(2),  V.stride(3),
            P.stride(0),  P.stride(1),  P.stride(2),  P.stride(3),
            BLOCK_SKV=BLOCK_SKV,
            HEAD_DIM=HEAD_DIM_C,
            num_warps=num_warps,
            num_stages=2,
        )

    # ----------------------------------------------------------------
    # Step 3: dV = P_drop_grouped^T @ dO_grouped  -> [bs, 8, skv, 128]
    # GQA-native: reshape to [bs, 8, 10*sq, ...], use cuBLAS BMM
    # ----------------------------------------------------------------
    dO_grouped = dO.reshape(bs, n_kv_heads, n_groups * seq_q, HEAD_DIM)
    # [bs, 8, 10*sq, 128]

    P_drop_grouped = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups * seq_q, seq_kv)
    # [bs, 8, 10*sq, skv]  bfloat16

    dV = torch.matmul(P_drop_grouped.transpose(-2, -1), dO_grouped).to(torch.bfloat16)
    # [bs, 8, skv, 128]

    return dS, dV
