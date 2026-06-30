"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- Keep both BMMs as bfloat16 torch.matmul (cuBLAS-optimized)
- Two-phase Triton kernel for elementwise dropout-bwd + softmax-bwd:
  * Phase 1 (2D grid): each (row, kv_tile) program computes partial dot products
    and writes partial dS tiles scaled by a placeholder; then stores partial sums
    into a buffer. When seq_kv fits in one tile: single-pass (no phase 2 needed).
  * Phase 2: reduce partial dot sums and re-scale dS tiles (only for multi-tile rows)
  * Parallelizes along seq_kv dimension, eliminating serial bottleneck for large seq_kv
- BMM2 fused with GQA reduction: single large GEMM with K = 10*sq

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


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: Phase 1 of 2D-parallel softmax-bwd
#
# Grid: (total_rows, num_kv_tiles)
# Each program (row_idx, tile_idx) handles BLOCK_KV elements of seq_kv.
#
# For single-pass (num_kv_tiles == 1):
#   - Compute full dot, write final dS immediately
#
# For multi-tile:
#   - Compute partial dot for this tile -> store in partial_dot[row_idx, tile_idx]
#   - Store partial dS values (without the dot correction) into dS_partial buffer
#     Actually we store "P * dP" terms and "P_vals" separately — but that's
#     memory-heavy. Better: store partial dot sums, then phase 2 does a
#     correction pass writing the final dS.
#
# Phase 1: compute partial dots + store (dP_vals, P_vals) info — but storing
# all those is too expensive. Instead, phase 2 just re-reads dP_dropped+mask+P
# (same memory reads but parallelized) and writes final dS with known dot.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _softmax_bwd_phase1_kernel(
    dP_dropped_ptr,    # [total_rows, seq_kv]  bfloat16
    attn_weights_ptr,  # [total_rows, seq_kv]  bfloat16
    dropout_mask_ptr,  # [total_rows, seq_kv]  bool (uint8)
    partial_dot_ptr,   # [total_rows, num_kv_tiles]  float32  (output)
    seq_kv,            # runtime int
    inv_keep_prob,     # runtime float32
    BLOCK_KV: tl.constexpr,
):
    row_idx  = tl.program_id(0)
    tile_idx = tl.program_id(1)

    blk_start = tile_idx * BLOCK_KV
    offs = blk_start + tl.arange(0, BLOCK_KV)
    valid = offs < seq_kv

    base = row_idx * seq_kv

    dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
    dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
    P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

    dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
    partial_dot = tl.sum(P_vals * dP_vals, axis=0)

    # Store partial dot sum
    num_kv_tiles = tl.num_programs(1)
    tl.store(partial_dot_ptr + row_idx * num_kv_tiles + tile_idx, partial_dot)


@triton.jit
def _softmax_bwd_phase2_kernel(
    dP_dropped_ptr,    # [total_rows, seq_kv]  bfloat16
    attn_weights_ptr,  # [total_rows, seq_kv]  bfloat16
    dropout_mask_ptr,  # [total_rows, seq_kv]  bool (uint8)
    partial_dot_ptr,   # [total_rows, num_kv_tiles]  float32
    dS_ptr,            # [total_rows, seq_kv]  bfloat16  (output)
    seq_kv,            # runtime int
    inv_keep_prob,     # runtime float32
    num_kv_tiles,      # runtime int
    BLOCK_KV: tl.constexpr,
):
    row_idx  = tl.program_id(0)
    tile_idx = tl.program_id(1)

    # Reduce partial dots for this row
    dot = tl.zeros([1], dtype=tl.float32)
    for t in tl.range(0, num_kv_tiles):
        dot += tl.load(partial_dot_ptr + row_idx * num_kv_tiles + t)

    blk_start = tile_idx * BLOCK_KV
    offs = blk_start + tl.arange(0, BLOCK_KV)
    valid = offs < seq_kv

    base = row_idx * seq_kv

    dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
    dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
    P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

    dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
    dS_vals = P_vals * (dP_vals - dot)
    tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: single-pass for small seq_kv (fits in BLOCK_KV)
# Grid: (total_rows,)
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _softmax_bwd_singlepass_kernel(
    dP_dropped_ptr,    # [total_rows, seq_kv]  bfloat16
    attn_weights_ptr,  # [total_rows, seq_kv]  bfloat16
    dropout_mask_ptr,  # [total_rows, seq_kv]  bool (uint8)
    dS_ptr,            # [total_rows, seq_kv]  bfloat16  (output)
    seq_kv,            # runtime int
    inv_keep_prob,     # runtime float32
    BLOCK_KV: tl.constexpr,
):
    row_idx = tl.program_id(0)
    base = row_idx * seq_kv

    offs = tl.arange(0, BLOCK_KV)
    valid = offs < seq_kv

    dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
    dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
    P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

    dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
    dot = tl.sum(P_vals * dP_vals, axis=0)
    dS_vals = P_vals * (dP_vals - dot)
    tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # ── Step 1: Transpose grad and reshape for grouped computation ───────────
    # [bs, sq, 80, d] -> [bs, 80, sq, d] (contiguous bfloat16)
    dO = grad_attn_output.transpose(1, 2).contiguous()
    # [bs, 80, sq, d] -> [bs, 8, 10, sq, d]
    dO_grouped = dO.reshape(bs, n_kv_heads, n_groups, seq_q, HEAD_DIM)

    # ── Step 2: BMM1 — compute dP_dropped (bfloat16 matmul) ─────────────────
    # vs_T: [bs, 8, 1, d, skv]  (bfloat16)
    vs_T = value_states.transpose(-2, -1).unsqueeze(2)
    # dP_dropped_grouped: [bs, 8, 10, sq, skv]  bfloat16
    dP_dropped_grouped = torch.matmul(dO_grouped, vs_T)
    # Reshape to [bs, 80, sq, skv] — zero-copy view
    dP_dropped = dP_dropped_grouped.reshape(bs, n_heads, seq_q, seq_kv)

    # ── Step 3: Triton kernel — dropout bwd + softmax bwd ────────────────────
    total_rows = bs * n_heads * seq_q
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    # Flatten to [total_rows, seq_kv]
    dP_dropped_flat   = dP_dropped.reshape(total_rows, seq_kv)
    if not dP_dropped_flat.is_contiguous():
        dP_dropped_flat = dP_dropped_flat.contiguous()
    attn_weights_flat = attn_weights.reshape(total_rows, seq_kv)
    if not attn_weights_flat.is_contiguous():
        attn_weights_flat = attn_weights_flat.contiguous()
    dropout_mask_flat = dropout_mask.reshape(total_rows, seq_kv)
    if not dropout_mask_flat.is_contiguous():
        dropout_mask_flat = dropout_mask_flat.contiguous()
    dS_flat = dS.reshape(total_rows, seq_kv)

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # Choose tile size: use 1024 as standard block size for good occupancy
    BLOCK_KV = 1024

    if seq_kv <= BLOCK_KV:
        # Single-pass: fits in one tile
        BLK = triton.next_power_of_2(seq_kv)
        _softmax_bwd_singlepass_kernel[(total_rows,)](
            dP_dropped_flat,
            attn_weights_flat,
            dropout_mask_flat,
            dS_flat,
            seq_kv,
            inv_keep_prob,
            BLOCK_KV=BLK,
        )
    else:
        # Two-phase 2D parallel: each row split into multiple tiles
        num_kv_tiles = triton.cdiv(seq_kv, BLOCK_KV)
        partial_dot = torch.empty((total_rows, num_kv_tiles), dtype=torch.float32, device=dO.device)

        # Phase 1: compute partial dot sums
        _softmax_bwd_phase1_kernel[(total_rows, num_kv_tiles)](
            dP_dropped_flat,
            attn_weights_flat,
            dropout_mask_flat,
            partial_dot,
            seq_kv,
            inv_keep_prob,
            BLOCK_KV=BLOCK_KV,
        )

        # Phase 2: reduce partial dots, write final dS
        _softmax_bwd_phase2_kernel[(total_rows, num_kv_tiles)](
            dP_dropped_flat,
            attn_weights_flat,
            dropout_mask_flat,
            partial_dot,
            dS_flat,
            seq_kv,
            inv_keep_prob,
            num_kv_tiles,
            BLOCK_KV=BLOCK_KV,
        )

    # ── Step 4: Fused BMM2 + GQA reduction — single large GEMM ──────────────
    #
    # P_dropped: [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv] -> [bs*8, skv, 10*sq]
    # dO:        [bs, 8, 10, sq, d]   -> [bs*8, 10*sq, d]
    # result:    [bs*8, skv, d]

    P_dropped_2d = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    if not P_dropped_2d.is_contiguous():
        P_dropped_2d = P_dropped_2d.contiguous()
    P_dropped_2d_T = P_dropped_2d.transpose(-2, -1)

    dO_2d = dO_grouped.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)
    if not dO_2d.is_contiguous():
        dO_2d = dO_2d.contiguous()

    # Single BMM: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
    dV_flat = torch.bmm(P_dropped_2d_T, dO_2d)

    # Reshape to final output shape [bs, 8, skv, d]
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV
