"""
Optimized attention-backward kernel.

Strategy:
- Use cuBLAS BMMs for dP and dV (fast, uses tensor cores)
- Use Triton for the softmax-bwd pass (operates on materialized dP)
- dO transpose is done once, contiguous copy shared between dP and dV BMMs
- All BMM inputs are made explicitly contiguous to hit cuBLAS fast paths

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
def _softmax_bwd_kernel(
    # dP_dropped: [bs, 80, sq, skv] float32 (already dropout-masked and scaled)
    dP_ptr,
    # P: [bs, 80, sq, skv] bfloat16
    P_ptr,
    # dS output: [bs, 80, sq, skv] bfloat16
    dS_ptr,
    # strides for [bs, 80, sq, skv]
    stride_bs, stride_h, stride_sq, stride_skv,
    # problem dims
    seq_kv,
    BLOCK_SKV: tl.constexpr,
    BLOCKS_PER_ROW: tl.constexpr,
):
    """
    2D grid: axis-0 = row index (bs*80*sq), axis-1 = block within row.
    Each program handles a chunk of seq_kv elements.
    We do a two-phase reduction:
      Phase 1: each block computes partial sum of dP*P into atomic global accumulation
      Phase 2 (separate kernel): finish dS = P*(dP - row_sum)
    
    Actually for simplicity: use 1D grid with one block per row, but process
    BLOCK_SKV elements per iteration. The 2D parallelism is over rows, not within.
    With rows = bs*80*sq and each row doing seq_kv work, the GPU should be well occupied.
    """
    row_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)

    # Decode row_idx -> (bs, h, sq)
    # stride_sq = seq_kv, stride_h = sq*seq_kv, stride_bs = 80*sq*seq_kv
    # But we work in terms of element offsets
    row_base = (row_idx * stride_sq)  # offset for this row in the flat (bs*80*sq) dimension

    # This block handles [chunk_idx*BLOCK_SKV : (chunk_idx+1)*BLOCK_SKV]
    kv_start = chunk_idx * BLOCK_SKV
    kv_offsets = kv_start + tl.arange(0, BLOCK_SKV)
    kv_mask = kv_offsets < seq_kv

    dP_row_base = dP_ptr + row_base
    P_row_base = P_ptr + row_base
    dS_row_base = dS_ptr + row_base

    dP_tile = tl.load(dP_row_base + kv_offsets, mask=kv_mask, other=0.0)
    p_tile = tl.load(P_row_base + kv_offsets, mask=kv_mask, other=0.0).to(tl.float32)

    # Partial sum for this chunk
    partial_sum = tl.sum(dP_tile * p_tile, axis=0)

    # We need the full row sum — but with 2D grid we can't easily share.
    # Fall back to 1D: only use this when BLOCKS_PER_ROW == 1
    # (handled by caller choosing grid size appropriately)
    # For BLOCKS_PER_ROW > 1, we need a scratch buffer — see below.
    ds_tile = p_tile * (dP_tile - partial_sum)
    tl.store(dS_row_base + kv_offsets, ds_tile.to(tl.bfloat16), mask=kv_mask)


@triton.jit
def _softmax_bwd_1d_kernel(
    # dP_dropped: [N_rows, seq_kv] float32
    dP_ptr,
    # P: [N_rows, seq_kv] bfloat16
    P_ptr,
    # dS output: [N_rows, seq_kv] bfloat16
    dS_ptr,
    # problem dims
    seq_kv,
    BLOCK_SKV: tl.constexpr,
):
    """
    1D grid: one program per row (bs*80*sq).
    Computes dS = P * (dP - sum(dP*P)) in two passes over seq_kv.
    """
    row_idx = tl.program_id(0)
    row_off = row_idx * seq_kv

    # Pass 1: compute row_sum = sum_j(dP_j * P_j)
    row_sum = tl.zeros([1], dtype=tl.float32)
    for kv_start in tl.range(0, seq_kv, BLOCK_SKV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_SKV)
        kv_mask = kv_offsets < seq_kv
        dp = tl.load(dP_ptr + row_off + kv_offsets, mask=kv_mask, other=0.0)
        p  = tl.load(P_ptr  + row_off + kv_offsets, mask=kv_mask, other=0.0).to(tl.float32)
        row_sum += tl.sum(dp * p, axis=0)
    row_sum_val = tl.sum(row_sum, axis=0)

    # Pass 2: write dS
    for kv_start in tl.range(0, seq_kv, BLOCK_SKV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_SKV)
        kv_mask = kv_offsets < seq_kv
        dp = tl.load(dP_ptr + row_off + kv_offsets, mask=kv_mask, other=0.0)
        p  = tl.load(P_ptr  + row_off + kv_offsets, mask=kv_mask, other=0.0).to(tl.float32)
        ds = p * (dp - row_sum_val)
        tl.store(dS_ptr + row_off + kv_offsets, ds.to(tl.bfloat16), mask=kv_mask)


@triton.jit  
def _softmax_bwd_2d_phase1_kernel(
    # dP: [N_rows, seq_kv] float32
    dP_ptr,
    # P: [N_rows, seq_kv] bfloat16
    P_ptr,
    # partial sums scratch: [N_rows, BLOCKS_PER_ROW] float32
    scratch_ptr,
    # problem dims
    seq_kv,
    BLOCKS_PER_ROW: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """Phase 1: each block computes partial sum(dP*P) for its chunk of seq_kv."""
    row_idx   = tl.program_id(0)
    chunk_idx = tl.program_id(1)

    kv_start  = chunk_idx * BLOCK_SKV
    kv_offsets = kv_start + tl.arange(0, BLOCK_SKV)
    kv_mask   = kv_offsets < seq_kv

    row_off = row_idx * seq_kv
    dp = tl.load(dP_ptr + row_off + kv_offsets, mask=kv_mask, other=0.0)
    p  = tl.load(P_ptr  + row_off + kv_offsets, mask=kv_mask, other=0.0).to(tl.float32)

    partial = tl.sum(dp * p, axis=0)
    tl.store(scratch_ptr + row_idx * BLOCKS_PER_ROW + chunk_idx, partial)


@triton.jit
def _softmax_bwd_2d_phase2_kernel(
    # dP: [N_rows, seq_kv] float32
    dP_ptr,
    # P: [N_rows, seq_kv] bfloat16
    P_ptr,
    # partial sums: [N_rows, BLOCKS_PER_ROW] float32
    scratch_ptr,
    # dS output: [N_rows, seq_kv] bfloat16
    dS_ptr,
    # problem dims
    seq_kv,
    BLOCKS_PER_ROW: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """Phase 2: reduce partial sums -> row_sum, then write dS = P*(dP - row_sum)."""
    row_idx   = tl.program_id(0)
    chunk_idx = tl.program_id(1)

    # Load all partial sums for this row and reduce
    partial_offsets = tl.arange(0, BLOCKS_PER_ROW)
    partials = tl.load(scratch_ptr + row_idx * BLOCKS_PER_ROW + partial_offsets)
    row_sum = tl.sum(partials, axis=0)

    # Write dS for this chunk
    kv_start   = chunk_idx * BLOCK_SKV
    kv_offsets = kv_start + tl.arange(0, BLOCK_SKV)
    kv_mask    = kv_offsets < seq_kv

    row_off = row_idx * seq_kv
    dp = tl.load(dP_ptr + row_off + kv_offsets, mask=kv_mask, other=0.0)
    p  = tl.load(P_ptr  + row_off + kv_offsets, mask=kv_mask, other=0.0).to(tl.float32)

    ds = p * (dp - row_sum)
    tl.store(dS_ptr + row_off + kv_offsets, ds.to(tl.bfloat16), mask=kv_mask)


def custom_kernel(data):
    (dO_in, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = dO_in.shape[0]
    seq_q  = dO_in.shape[1]
    seq_kv = value_states.shape[2]
    n_kv   = NUM_KEY_VALUE_HEADS   # 8
    n_g    = N_GROUPS              # 10
    n_h    = NUM_ATTENTION_HEADS   # 80
    d      = HEAD_DIM              # 128

    scale = 1.0 / (1.0 - attention_dropout)

    # ── Step 1: Transpose dO once ─────────────────────────────────────────────
    # dO_in: [bs, sq, 80, d] -> dO: [bs, 80, sq, d]
    dO = dO_in.transpose(1, 2).contiguous()  # [bs, 80, sq, d] bfloat16

    # ── Step 2: Compute dP = dO @ V^T via large 2D matmul ────────────────────
    # GQA: value_states [bs, 8, skv, d], dO [bs, 80, sq, d]
    # Reshape for large 2D matmul to maximize GEMM tile utilization on B200:
    #   dO reshaped to [bs*80*sq, d]
    #   value_states repeated for 10 groups -> [bs*80, d, skv] arranged as [bs*80*sq, d] @ [d, skv]
    # We use the GQA bmm trick: [bs*8, 10*sq, d] @ [bs*8, d, skv]
    # Make both sides contiguous before bmm for cuBLAS fast path
    dO_flat = dO.reshape(bs * n_kv, n_g * seq_q, d)  # [bs*8, 10*sq, d] — already contiguous
    # value_states: [bs, 8, skv, d] -> [bs*8, skv, d] -> transpose to [bs*8, d, skv] contiguous
    vs_flat = value_states.reshape(bs * n_kv, seq_kv, d)
    # Make the transposed V contiguous so cuBLAS sees a proper row-major matrix
    vs_T = vs_flat.transpose(-2, -1).contiguous()  # [bs*8, d, skv] contiguous
    # dP_flat: [bs*8, 10*sq, skv]
    dP_flat = torch.bmm(dO_flat, vs_T)  # bfloat16
    # Reshape to [bs, 80, sq, skv]
    dP = dP_flat.reshape(bs, n_h, seq_q, seq_kv)

    # ── Step 3: Apply dropout mask and scale ─────────────────────────────────
    # dP_dropped = dP * mask / (1 - p_drop)
    # Use float32 for accuracy
    dP_dropped_f32 = dP.float() * dropout_mask.float() * scale  # [bs, 80, sq, skv] float32

    # Make contiguous flat views for Triton
    dP_flat_f32 = dP_dropped_f32.contiguous().reshape(-1, seq_kv)  # [N_rows, skv]
    P_flat      = attn_weights.contiguous().reshape(-1, seq_kv)     # [N_rows, skv] bfloat16
    N_rows = bs * n_h * seq_q

    dS_flat = torch.empty((N_rows, seq_kv), dtype=torch.bfloat16, device=dO.device)

    # ── Step 4: Softmax backward via 2D Triton kernel ────────────────────────
    # Choose block size and parallelism strategy
    if seq_kv <= 512:
        # 1D kernel: one block per row, all seq_kv fits in a few iterations
        BLOCK_SKV = min(512, triton.next_power_of_2(seq_kv))
        grid_1d = (N_rows,)
        _softmax_bwd_1d_kernel[grid_1d](
            dP_flat_f32, P_flat, dS_flat,
            seq_kv,
            BLOCK_SKV=BLOCK_SKV,
        )
    else:
        # 2D kernel: multiple blocks per row to increase parallelism
        BLOCK_SKV = 512
        BLOCKS_PER_ROW = triton.cdiv(seq_kv, BLOCK_SKV)
        # Round BLOCKS_PER_ROW up to next power of 2 (for the reduction)
        BLOCKS_PER_ROW_POW2 = triton.next_power_of_2(BLOCKS_PER_ROW)

        scratch = torch.empty((N_rows, BLOCKS_PER_ROW_POW2),
                               dtype=torch.float32, device=dO.device)

        grid_2d = (N_rows, BLOCKS_PER_ROW_POW2)

        _softmax_bwd_2d_phase1_kernel[grid_2d](
            dP_flat_f32, P_flat, scratch,
            seq_kv,
            BLOCKS_PER_ROW=BLOCKS_PER_ROW_POW2,
            BLOCK_SKV=BLOCK_SKV,
        )
        _softmax_bwd_2d_phase2_kernel[grid_2d](
            dP_flat_f32, P_flat, scratch, dS_flat,
            seq_kv,
            BLOCKS_PER_ROW=BLOCKS_PER_ROW_POW2,
            BLOCK_SKV=BLOCK_SKV,
        )

    dS = dS_flat.reshape(bs, n_h, seq_q, seq_kv)

    # ── Step 5: Compute dV via BMM ────────────────────────────────────────────
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs*8, 10*sq, skv]
    Pd_flat = attn_weights_dropped.reshape(bs * n_kv, n_g * seq_q, seq_kv)
    # Make the transposed Pd contiguous so cuBLAS sees a proper row-major matrix
    # Pd^T: [bs*8, skv, 10*sq] contiguous
    Pd_T = Pd_flat.transpose(-2, -1).contiguous()  # [bs*8, skv, 10*sq]
    dO_for_dV = dO.reshape(bs * n_kv, n_g * seq_q, d)  # same as dO_flat, already contiguous
    # dV_flat = Pd^T @ dO: [bs*8, skv, d]
    dV_flat = torch.bmm(Pd_T, dO_for_dV)
    dV = dV_flat.reshape(bs, n_kv, seq_kv, d).to(torch.bfloat16)

    return dS, dV
