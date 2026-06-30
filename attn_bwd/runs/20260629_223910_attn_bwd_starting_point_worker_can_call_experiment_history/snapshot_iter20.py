"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- Fused BMM1 + softmax-bwd Triton kernel: eliminates materialization of the
  large [bs, 80, sq, skv] intermediate dP_dropped tensor.
  Each program handles one (batch, head_group, query_row) entry:
  1. Loads dO_row [128] into registers
  2. Tiles over seq_kv: for each kv-tile loads V-rows and computes partial dot
  3. Simultaneously loads attn_weights and dropout_mask
  4. Accumulates dot = sum(P * dP) across the full seq_kv dimension
  5. Second pass (or single pass for small seq_kv): writes dS = P*(dP - dot)
- BMM2 fused with GQA reduction runs concurrently on stream B:
  [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
- Custom Triton transpose kernel for dO
- Dual-stream pipelining

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
# Triton kernel: tiled transpose [bs, sq, n_heads, d] -> [bs, n_heads, sq, d]
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _transpose_sq_heads_kernel(
    src_ptr,
    dst_ptr,
    bs,
    sq,
    n_heads,
    HEAD_DIM: tl.constexpr,
    TILE_SQ: tl.constexpr,
    TILE_H: tl.constexpr,
):
    num_tiles_sq = tl.cdiv(sq, TILE_SQ)
    num_tiles_h  = tl.cdiv(n_heads, TILE_H)

    pid = tl.program_id(0)
    tile_per_batch = num_tiles_sq * num_tiles_h
    batch_idx  = pid // tile_per_batch
    tile_idx   = pid % tile_per_batch
    tile_sq_idx = tile_idx % num_tiles_sq
    tile_h_idx  = tile_idx // num_tiles_sq

    sq_start = tile_sq_idx * TILE_SQ
    h_start  = tile_h_idx  * TILE_H

    offs_sq = sq_start + tl.arange(0, TILE_SQ)
    offs_h  = h_start  + tl.arange(0, TILE_H)
    offs_d  = tl.arange(0, HEAD_DIM)

    valid_sq = offs_sq < sq
    valid_h  = offs_h  < n_heads

    src_base = batch_idx * sq * n_heads * HEAD_DIM
    src_offsets = (src_base
                   + offs_sq[:, None, None] * (n_heads * HEAD_DIM)
                   + offs_h[None, :, None] * HEAD_DIM
                   + offs_d[None, None, :])
    valid_mask = (valid_sq[:, None, None] & valid_h[None, :, None])

    vals = tl.load(src_ptr + src_offsets, mask=valid_mask, other=0.0)

    dst_base = batch_idx * n_heads * sq * HEAD_DIM
    dst_offsets = (dst_base
                   + offs_h[None, :, None] * (sq * HEAD_DIM)
                   + offs_sq[:, None, None] * HEAD_DIM
                   + offs_d[None, None, :])

    tl.store(dst_ptr + dst_offsets, vals, mask=valid_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Fused BMM1 + softmax-bwd Triton kernel
#
# Each program handles one query row: (batch_kv_idx, query_row_in_group)
# where batch_kv_idx indexes into [bs*8] and query_row indexes into [10*sq].
#
# The kernel:
# 1. Loads dO_row [HEAD_DIM] into registers
# 2. Tiles over seq_kv computing dP[j] = dot(dO_row, V[j]) for each kv position
#    (held as a block of BLOCK_KV values at a time)
# 3. Applies dropout mask scaling to get dP (undropped)
# 4. Reads attn_weights P[j] and accumulates dot_sum = sum(P * dP)
# 5. Second pass: writes dS[j] = P[j] * (dP[j] - dot_sum)
#
# Grid: (bs * n_kv_heads, n_groups * seq_q)  = 2D grid
# Each program processes one (batch_kv, query_position) pair.
#
# Layout of inputs:
#   dO_2d:      [bs*8, 10*sq, 128]   — row = batch_kv * (10*sq) + q_local
#   V:          [bs*8, skv, 128]     — V[batch_kv, kv_pos, :]
#   P_w:        [bs, 80, sq, skv]    — accessed as [b, h, q, :]
#   P_dropped:  [bs, 80, sq, skv]    — accessed as [b, h, q, :]
#   mask:       [bs, 80, sq, skv]    — accessed as [b, h, q, :]
#   dS_out:     [bs, 80, sq, skv]    — output
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _fused_bmm1_softmax_bwd_kernel(
    dO_ptr,            # [bs*8, 10*sq, 128]   bfloat16  (contiguous)
    V_ptr,             # [bs*8, skv, 128]      bfloat16  (contiguous)
    attn_weights_ptr,  # [bs, 80, sq, skv]     bfloat16
    dropout_mask_ptr,  # [bs, 80, sq, skv]     bool (uint8)
    dS_ptr,            # [bs, 80, sq, skv]     bfloat16  (output)
    bs,                # batch size
    seq_q,             # seq_q
    seq_kv,            # seq_kv
    n_groups,          # 10
    n_kv_heads,        # 8
    inv_keep_prob,     # float32
    stride_aw_bs,      # strides for attn_weights [bs, 80, sq, skv]
    stride_aw_h,
    stride_aw_sq,
    stride_aw_skv,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,  # 128
    SINGLE_PASS: tl.constexpr,
):
    # Program indices
    batch_kv_idx = tl.program_id(0)   # in [0, bs*8)
    q_local_idx  = tl.program_id(1)   # in [0, 10*sq)

    # Decode batch and head indices
    b_idx    = batch_kv_idx // n_kv_heads
    kv_head  = batch_kv_idx % n_kv_heads
    g_idx    = q_local_idx // seq_q      # group index [0, 10)
    q_idx    = q_local_idx % seq_q       # query position [0, sq)

    # Head index in full [80] head space
    h_idx    = kv_head * n_groups + g_idx

    # ── Load dO_row [HEAD_DIM] into registers ──
    dO_row_offset = batch_kv_idx * (n_groups * seq_q * HEAD_DIM) + q_local_idx * HEAD_DIM
    offs_d = tl.arange(0, HEAD_DIM)
    dO_row = tl.load(dO_ptr + dO_row_offset + offs_d).to(tl.float32)  # [HEAD_DIM]

    # ── Base pointer for V[batch_kv_idx, :, :] ──
    V_batch_base = batch_kv_idx * seq_kv * HEAD_DIM

    # ── Base pointer for attn_weights / mask / dS for this (b, h, q) row ──
    aw_row_base = (b_idx * stride_aw_bs
                   + h_idx * stride_aw_h
                   + q_idx * stride_aw_sq)

    if SINGLE_PASS:
        # Single pass: seq_kv <= BLOCK_KV, everything fits in one tile
        offs_kv = tl.arange(0, BLOCK_KV)
        valid_kv = offs_kv < seq_kv

        # Load V tile: [BLOCK_KV, HEAD_DIM]
        V_offsets = (V_batch_base
                     + offs_kv[:, None] * HEAD_DIM
                     + offs_d[None, :])
        V_tile = tl.load(V_ptr + V_offsets, mask=valid_kv[:, None], other=0.0).to(tl.float32)

        # Compute dP_dropped[kv] = dot(dO_row, V[kv]) for each kv position
        # dO_row: [HEAD_DIM], V_tile: [BLOCK_KV, HEAD_DIM]
        dP_dropped_tile = tl.sum(dO_row[None, :] * V_tile, axis=1)  # [BLOCK_KV]

        # Load dropout mask and apply scaling
        mask_offsets = aw_row_base + offs_kv * stride_aw_skv
        dmask_vals = tl.load(dropout_mask_ptr + mask_offsets, mask=valid_kv, other=0).to(tl.int1)
        dP_tile = tl.where(dmask_vals, dP_dropped_tile * inv_keep_prob, 0.0)

        # Load attn_weights P
        P_vals = tl.load(attn_weights_ptr + mask_offsets, mask=valid_kv, other=0.0).to(tl.float32)

        # Compute dot = sum(P * dP)
        dot = tl.sum(P_vals * dP_tile, axis=0)

        # Compute and store dS = P * (dP - dot)
        dS_vals = P_vals * (dP_tile - dot)
        tl.store(dS_ptr + mask_offsets, dS_vals.to(tl.bfloat16), mask=valid_kv)

    else:
        # Two-pass: seq_kv > BLOCK_KV
        # Pass 1: compute dot = sum(P * dP)
        dot = tl.zeros([1], dtype=tl.float32)
        for kv_start in tl.range(0, seq_kv, BLOCK_KV):
            offs_kv = kv_start + tl.arange(0, BLOCK_KV)
            valid_kv = offs_kv < seq_kv

            V_offsets = (V_batch_base
                         + offs_kv[:, None] * HEAD_DIM
                         + offs_d[None, :])
            V_tile = tl.load(V_ptr + V_offsets, mask=valid_kv[:, None], other=0.0).to(tl.float32)
            dP_dropped_tile = tl.sum(dO_row[None, :] * V_tile, axis=1)

            mask_offsets = aw_row_base + offs_kv * stride_aw_skv
            dmask_vals = tl.load(dropout_mask_ptr + mask_offsets, mask=valid_kv, other=0).to(tl.int1)
            dP_tile = tl.where(dmask_vals, dP_dropped_tile * inv_keep_prob, 0.0)

            P_vals = tl.load(attn_weights_ptr + mask_offsets, mask=valid_kv, other=0.0).to(tl.float32)
            dot += tl.sum(P_vals * dP_tile, axis=0)

        # Pass 2: compute and store dS
        for kv_start in tl.range(0, seq_kv, BLOCK_KV):
            offs_kv = kv_start + tl.arange(0, BLOCK_KV)
            valid_kv = offs_kv < seq_kv

            V_offsets = (V_batch_base
                         + offs_kv[:, None] * HEAD_DIM
                         + offs_d[None, :])
            V_tile = tl.load(V_ptr + V_offsets, mask=valid_kv[:, None], other=0.0).to(tl.float32)
            dP_dropped_tile = tl.sum(dO_row[None, :] * V_tile, axis=1)

            mask_offsets = aw_row_base + offs_kv * stride_aw_skv
            dmask_vals = tl.load(dropout_mask_ptr + mask_offsets, mask=valid_kv, other=0).to(tl.int1)
            dP_tile = tl.where(dmask_vals, dP_dropped_tile * inv_keep_prob, 0.0)

            P_vals = tl.load(attn_weights_ptr + mask_offsets, mask=valid_kv, other=0.0).to(tl.float32)
            dS_vals = P_vals * (dP_tile - dot)
            tl.store(dS_ptr + mask_offsets, dS_vals.to(tl.bfloat16), mask=valid_kv)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    device = grad_attn_output.device

    # ── Step 1: Custom tiled transpose [bs, sq, 80, d] -> [bs, 80, sq, d] ────
    dO = torch.empty((bs, n_heads, seq_q, HEAD_DIM), dtype=torch.bfloat16, device=device)

    TILE_SQ = 8
    TILE_H  = 8
    num_tiles_sq = triton.cdiv(seq_q, TILE_SQ)
    num_tiles_h  = triton.cdiv(n_heads, TILE_H)
    transpose_grid = bs * num_tiles_sq * num_tiles_h

    _transpose_sq_heads_kernel[(transpose_grid,)](
        grad_attn_output,
        dO,
        bs, seq_q, n_heads,
        HEAD_DIM=HEAD_DIM,
        TILE_SQ=TILE_SQ,
        TILE_H=TILE_H,
    )

    # dO is now [bs, 80, sq, d], contiguous
    # Reshape to [bs*8, 10*sq, d] for BMMs
    dO_2d = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # ── Step 2: Prepare V for the fused kernel ────────────────────────────────
    # V_2d: [bs*8, skv, 128] — contiguous for efficient tiled loading
    V_2d = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    if not V_2d.is_contiguous():
        V_2d = V_2d.contiguous()

    # ── Step 3: Prepare BMM2 inputs ───────────────────────────────────────────
    P_dropped_2d = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    P_dropped_2d_T = P_dropped_2d.transpose(-2, -1)

    # Allocate outputs
    dV_flat = torch.empty((bs * n_kv_heads, seq_kv, HEAD_DIM),
                           dtype=torch.bfloat16, device=device)
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=device)

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # ── Step 4: Launch BMM2 on stream B concurrently ──────────────────────────
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    default_stream = torch.cuda.current_stream()
    start_event = torch.cuda.Event()
    start_event.record(default_stream)

    # Stream B: BMM2 (dV computation) — runs concurrently with fused kernel on A
    with torch.cuda.stream(stream_b):
        stream_b.wait_event(start_event)
        torch.bmm(P_dropped_2d_T, dO_2d, out=dV_flat)

    # ── Step 5: Fused BMM1 + softmax-bwd on stream A ─────────────────────────
    # Grid: (bs*8, 10*sq)
    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 4096)
    SINGLE_PASS = (seq_kv <= BLOCK_KV)

    # Strides for attn_weights [bs, 80, sq, skv]
    stride_aw_bs  = attn_weights.stride(0)
    stride_aw_h   = attn_weights.stride(1)
    stride_aw_sq  = attn_weights.stride(2)
    stride_aw_skv = attn_weights.stride(3)

    grid = (bs * n_kv_heads, n_groups * seq_q)

    with torch.cuda.stream(stream_a):
        stream_a.wait_event(start_event)
        _fused_bmm1_softmax_bwd_kernel[grid](
            dO_2d,
            V_2d,
            attn_weights,
            dropout_mask,
            dS,
            bs, seq_q, seq_kv,
            n_groups, n_kv_heads,
            inv_keep_prob,
            stride_aw_bs, stride_aw_h, stride_aw_sq, stride_aw_skv,
            BLOCK_KV=BLOCK_KV,
            HEAD_DIM=HEAD_DIM,
            SINGLE_PASS=SINGLE_PASS,
            num_warps=4,
        )

    # ── Step 6: Sync both streams back to the default stream ─────────────────
    event_a = torch.cuda.Event()
    event_b = torch.cuda.Event()
    event_a.record(stream_a)
    event_b.record(stream_b)

    default_stream.wait_event(event_a)
    default_stream.wait_event(event_b)

    # ── Step 7: Reshape outputs ───────────────────────────────────────────────
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV
