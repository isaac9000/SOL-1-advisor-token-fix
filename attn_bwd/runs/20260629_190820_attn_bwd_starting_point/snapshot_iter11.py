"""
Optimized attention-backward kernel using cuBLAS BMMs +
a two-phase Triton kernel for fused softmax-backward + dropout.

Key optimization: restructure BMMs to use batch=bs*8 with large 10*sq matrix
dimensions, eliminating the group-reduction step. The dO contiguous call is
done once and shared across both BMMs.

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
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10


@triton.jit
def softmax_bwd_phase1_kernel(
    dP_dropped_ptr,    # [B80, sq, skv]  bfloat16
    dropout_mask_ptr,  # [B80, sq, skv]  bool (uint8)
    P_ptr,             # [B80, sq, skv]  bfloat16
    row_sum_ptr,       # [B80*sq]        float32  (output)
    skv: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    N_TILES: tl.constexpr,
):
    """
    2D grid: axis0 = row_id (B80*sq), axis1 = tile_id (0..N_TILES-1)
    Each program accumulates partial sum for its tile, then atomically adds to row_sum.
    """
    row_id  = tl.program_id(0)
    tile_id = tl.program_id(1)

    row_offset = row_id * skv
    start = tile_id * BLOCK_SKV
    kv_ids = start + tl.arange(0, BLOCK_SKV)
    mask = kv_ids < skv
    offsets = row_offset + kv_ids

    dp_dropped = tl.load(dP_dropped_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dmask      = tl.load(dropout_mask_ptr + offsets, mask=mask, other=0).to(tl.float32)
    p          = tl.load(P_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    dp = dp_dropped * dmask * scale
    partial = tl.sum(dp * p, axis=0)

    tl.atomic_add(row_sum_ptr + row_id, partial)


@triton.jit
def softmax_bwd_phase2_kernel(
    dP_dropped_ptr,    # [B80, sq, skv]  bfloat16
    dropout_mask_ptr,  # [B80, sq, skv]  bool (uint8)
    P_ptr,             # [B80, sq, skv]  bfloat16
    row_sum_ptr,       # [B80*sq]        float32
    dS_ptr,            # [B80, sq, skv]  bfloat16  (output)
    skv: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """
    2D grid: axis0 = row_id (B80*sq), axis1 = tile_id
    Each program reads the row_sum and writes its tile of dS.
    """
    row_id  = tl.program_id(0)
    tile_id = tl.program_id(1)

    row_offset = row_id * skv
    start = tile_id * BLOCK_SKV
    kv_ids = start + tl.arange(0, BLOCK_SKV)
    mask = kv_ids < skv
    offsets = row_offset + kv_ids

    dp_dropped = tl.load(dP_dropped_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dmask      = tl.load(dropout_mask_ptr + offsets, mask=mask, other=0).to(tl.float32)
    p          = tl.load(P_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rs         = tl.load(row_sum_ptr + row_id)

    dp = dp_dropped * dmask * scale
    ds = p * (dp - rs)

    tl.store(dS_ptr + offsets, ds.to(tl.bfloat16), mask=mask)


def fused_softmax_bwd(dP_dropped, dropout_mask, attn_weights, attention_dropout):
    """
    Two-phase fused softmax backward + dropout scaling.
    All inputs/output shaped [B80, sq, skv] and CONTIGUOUS.
    """
    B80, sq, skv = dP_dropped.shape
    dS = torch.empty_like(dP_dropped)

    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Choose tile block size for skv dimension (power of 2)
    if skv <= 64:
        BLOCK_SKV = 64
    elif skv <= 128:
        BLOCK_SKV = 128
    elif skv <= 256:
        BLOCK_SKV = 256
    elif skv <= 512:
        BLOCK_SKV = 512
    elif skv <= 1024:
        BLOCK_SKV = 1024
    else:
        BLOCK_SKV = 2048

    total_rows = B80 * sq
    import math
    n_tiles = math.ceil(skv / BLOCK_SKV)

    # Phase 1: accumulate row sums via atomic add
    row_sum = torch.zeros(total_rows, dtype=torch.float32, device=dP_dropped.device)

    grid1 = (total_rows, n_tiles)
    softmax_bwd_phase1_kernel[grid1](
        dP_dropped,
        dropout_mask,
        attn_weights,
        row_sum,
        skv=skv,
        scale=scale,
        BLOCK_SKV=BLOCK_SKV,
        N_TILES=n_tiles,
    )

    # Phase 2: write dS using the computed row sums
    grid2 = (total_rows, n_tiles)
    softmax_bwd_phase2_kernel[grid2](
        dP_dropped,
        dropout_mask,
        attn_weights,
        row_sum,
        dS,
        skv=skv,
        scale=scale,
        BLOCK_SKV=BLOCK_SKV,
    )

    return dS


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs = grad_attn_output.shape[0]
    seq_q = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, sq, 80, d] -> [bs, 80, sq, d] contiguous
    # Done ONCE, reused for both dP and dV BMMs
    dO = grad_attn_output.transpose(1, 2).contiguous()  # bf16, [bs, 80, sq, d]

    # ------------------------------------------------------------------ #
    #  Compute dP_dropped = dO @ V^T  WITHOUT materializing expanded V
    #
    #  Grouped BMM with batch = bs*8:
    #  dO: [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    #  V:  [bs, 8, skv, d] -> [bs*8, d, skv]
    #  BMM: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
    #  reshape -> [bs*80, sq, skv]
    # ------------------------------------------------------------------ #
    dO_grouped = dO.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM)
    dO_for_dP = dO_grouped.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS * seq_q, HEAD_DIM)

    V_flat_t = value_states.reshape(bs * NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM).transpose(-2, -1).contiguous()

    # BMM output is [B8, 10*sq, skv] - contiguous
    dP_dropped_grouped = torch.bmm(dO_for_dP, V_flat_t)

    # Reshape to [B80, sq, skv] - view (no copy) since memory is contiguous
    dP_dropped_flat = dP_dropped_grouped.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv) \
                                        .reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # Flatten attn_weights and dropout_mask to [B80, sq, skv]
    dropout_mask_flat = dropout_mask.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv)
    attn_weights_flat = attn_weights.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # Ensure contiguous for Triton kernel
    if not dP_dropped_flat.is_contiguous():
        dP_dropped_flat = dP_dropped_flat.contiguous()
    if not dropout_mask_flat.is_contiguous():
        dropout_mask_flat = dropout_mask_flat.contiguous()
    if not attn_weights_flat.is_contiguous():
        attn_weights_flat = attn_weights_flat.contiguous()

    dS_flat = fused_softmax_bwd(dP_dropped_flat, dropout_mask_flat, attn_weights_flat, attention_dropout)
    dS = dS_flat.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ------------------------------------------------------------------ #
    #  Compute dV = attn_weights_dropped^T @ dO  with batch=bs*8
    #
    #  Key: merge 10 groups into matrix dimension, giving large 10*sq rows
    #  attn_weights_dropped: [bs, 80, sq, skv]
    #    -> [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv] -> T -> [bs*8, skv, 10*sq]
    #  dO: [bs*8, 10*sq, d]  (from above)
    #  BMM: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]  (sum over groups)
    # ------------------------------------------------------------------ #
    # Reshape attn_weights_dropped to [B8, 10*sq, skv]
    aw_dropped_for_dV = attn_weights_dropped.reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv
    ).reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS * seq_q, seq_kv)
    # Transpose to [B8, skv, 10*sq]
    aw_dropped_t = aw_dropped_for_dV.transpose(-2, -1).contiguous()  # [B8, skv, 10*sq]

    # dO for dV: [bs*8, 10*sq, d] - same reshape as dO_for_dP
    # dO_for_dP is already this shape; check if contiguous
    dO_for_dV = dO_for_dP if dO_for_dP.is_contiguous() else dO_for_dP.contiguous()

    # Single BMM: [B8, skv, 10*sq] @ [B8, 10*sq, d] -> [B8, skv, d]
    dV_flat = torch.bmm(aw_dropped_t, dO_for_dV)  # [B8, skv, d]  bf16

    dV = dV_flat.reshape(bs, NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV
