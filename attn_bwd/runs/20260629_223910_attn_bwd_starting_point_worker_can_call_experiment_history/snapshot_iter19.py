"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- Both BMMs as clean 3D batched GEMMs (cuBLAS-optimized, no broadcasting)
- BMM1 restructured: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
  then reshape to [bs, 80, sq, skv] — same K-merging trick as BMM2
- BMM2 fused with GQA reduction: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
- dO_2d [bs*8, 10*sq, d] reused across both BMMs (computed once)
- Dual-stream pipelining: BMM1 on stream A, BMM2 on stream B (launched concurrently)
  Triton softmax-bwd runs on stream A after BMM1 (overlaps with BMM2 on stream B)
  Final sync waits for both streams to complete
- Row-batched Triton kernel for elementwise dropout-bwd + softmax-bwd
- Custom Triton transpose kernel: reads grad_attn_output [bs, sq, 80, d] natively
  and writes transposed [bs, 80, sq, d] using tiled 2D access pattern for
  coalesced reads and writes (avoids non-coalesced generic PyTorch transpose)

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
#
# The transpose swaps dims 1 and 2: (sq <-> n_heads).
# The head dimension (d=128) is kept as a contiguous inner dimension and
# is NOT transposed — we move entire head vectors of size d.
#
# Input layout:  [bs, sq, n_heads, d]   strides: (sq*n_heads*d, n_heads*d, d, 1)
# Output layout: [bs, n_heads, sq, d]   strides: (n_heads*sq*d, sq*d, d, 1)
#
# Grid: (bs * cdiv(sq, TILE_SQ) * cdiv(n_heads, TILE_H),)
# Each program handles a TILE_SQ x TILE_H tile of (sq, n_heads) for one batch.
# Within the tile, all HEAD_DIM elements are processed in one shot.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _transpose_sq_heads_kernel(
    src_ptr,              # [bs, sq, n_heads, d]  bfloat16
    dst_ptr,              # [bs, n_heads, sq, d]  bfloat16
    bs,                   # batch size
    sq,                   # seq_q
    n_heads,              # 80
    HEAD_DIM: tl.constexpr,  # 128
    TILE_SQ: tl.constexpr,   # tile size over sq dimension
    TILE_H: tl.constexpr,    # tile size over n_heads dimension
):
    # Compute grid dimensions
    num_tiles_sq = tl.cdiv(sq, TILE_SQ)
    num_tiles_h  = tl.cdiv(n_heads, TILE_H)

    pid = tl.program_id(0)
    # Decompose pid into (batch_idx, tile_h_idx, tile_sq_idx)
    tile_per_batch = num_tiles_sq * num_tiles_h
    batch_idx  = pid // tile_per_batch
    tile_idx   = pid % tile_per_batch
    tile_sq_idx = tile_idx % num_tiles_sq
    tile_h_idx  = tile_idx // num_tiles_sq

    sq_start = tile_sq_idx * TILE_SQ
    h_start  = tile_h_idx  * TILE_H

    offs_sq = sq_start + tl.arange(0, TILE_SQ)   # [TILE_SQ]
    offs_h  = h_start  + tl.arange(0, TILE_H)    # [TILE_H]
    offs_d  = tl.arange(0, HEAD_DIM)              # [HEAD_DIM]

    valid_sq = offs_sq < sq      # [TILE_SQ]
    valid_h  = offs_h  < n_heads # [TILE_H]

    # Src strides: [bs, sq, n_heads, d]
    #   flat index = batch_idx * sq * n_heads * d
    #              + offs_sq[:, None] * n_heads * d
    #              + offs_h[None, :] * d
    #              + offs_d (broadcast over both sq and h)
    # Load: shape [TILE_SQ, TILE_H, HEAD_DIM]
    src_base = batch_idx * sq * n_heads * HEAD_DIM
    src_offsets = (src_base
                   + offs_sq[:, None, None] * (n_heads * HEAD_DIM)
                   + offs_h[None, :, None] * HEAD_DIM
                   + offs_d[None, None, :])
    valid_mask = (valid_sq[:, None, None] & valid_h[None, :, None])

    vals = tl.load(src_ptr + src_offsets, mask=valid_mask, other=0.0)

    # Dst strides: [bs, n_heads, sq, d]
    #   flat index = batch_idx * n_heads * sq * d
    #              + offs_h[None, :] * sq * d
    #              + offs_sq[:, None] * d
    #              + offs_d
    dst_base = batch_idx * n_heads * sq * HEAD_DIM
    dst_offsets = (dst_base
                   + offs_h[None, :, None] * (sq * HEAD_DIM)
                   + offs_sq[:, None, None] * HEAD_DIM
                   + offs_d[None, None, :])

    tl.store(dst_ptr + dst_offsets, vals, mask=valid_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: row-batched fused dropout-bwd + softmax-bwd
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _softmax_bwd_kernel(
    dP_dropped_ptr,    # [total_rows, seq_kv]  bfloat16
    attn_weights_ptr,  # [total_rows, seq_kv]  bfloat16
    dropout_mask_ptr,  # [total_rows, seq_kv]  bool (uint8)
    dS_ptr,            # [total_rows, seq_kv]  bfloat16  (output)
    total_rows,        # runtime int
    seq_kv,            # runtime int
    inv_keep_prob,     # runtime float32
    BLOCK_KV: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_CTA

    for i in tl.static_range(ROWS_PER_CTA):
        row_idx = row_start + i
        if row_idx < total_rows:
            base = row_idx * seq_kv

            if SINGLE_PASS:
                offs = tl.arange(0, BLOCK_KV)
                valid = offs < seq_kv

                dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                dot = tl.sum(P_vals * dP_vals, axis=0)
                dS_vals = P_vals * (dP_vals - dot)
                tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)
            else:
                dot = tl.zeros([1], dtype=tl.float32)
                for blk_start in tl.range(0, seq_kv, BLOCK_KV):
                    offs = blk_start + tl.arange(0, BLOCK_KV)
                    valid = offs < seq_kv

                    dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                    dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                    P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                    dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                    dot += tl.sum(P_vals * dP_vals, axis=0)

                for blk_start in tl.range(0, seq_kv, BLOCK_KV):
                    offs = blk_start + tl.arange(0, BLOCK_KV)
                    valid = offs < seq_kv

                    dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                    dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                    P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                    dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
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
    device = grad_attn_output.device

    # ── Step 1: Custom tiled transpose [bs, sq, 80, d] -> [bs, 80, sq, d] ────
    # Use a Triton kernel instead of PyTorch's generic transpose+contiguous.
    # The kernel uses TILE_SQ x TILE_H tiles over (sq, n_heads) for each batch,
    # processing HEAD_DIM elements in one shot per tile element.
    # This avoids the inefficient strided memory access pattern of the generic copy.

    dO = torch.empty((bs, n_heads, seq_q, HEAD_DIM), dtype=torch.bfloat16, device=device)

    # Tile sizes: TILE_SQ=8, TILE_H=8 -> each CTA handles 8*8=64 head-vectors
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
    # Reshape to [bs*8, 10*sq, d] for both BMMs
    dO_2d = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # ── Step 2: Prepare BMM inputs ────────────────────────────────────────────

    # BMM1 input: vs_T_2d [bs*8, d, skv]
    vs_T_2d = value_states.transpose(-2, -1).reshape(bs * n_kv_heads, HEAD_DIM, seq_kv)
    if not vs_T_2d.is_contiguous():
        vs_T_2d = vs_T_2d.contiguous()

    # BMM2 inputs: P_dropped_2d_T [bs*8, skv, 10*sq]
    P_dropped_2d = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    P_dropped_2d_T = P_dropped_2d.transpose(-2, -1)

    # Flatten attn_weights and dropout_mask for Triton kernel
    total_rows = bs * n_heads * seq_q
    attn_weights_flat = attn_weights.reshape(total_rows, seq_kv)
    dropout_mask_flat = dropout_mask.reshape(total_rows, seq_kv)

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 16384)
    SINGLE_PASS = (seq_kv <= BLOCK_KV)
    ROWS_PER_CTA = 4
    grid_size = triton.cdiv(total_rows, ROWS_PER_CTA)

    # Allocate output tensors
    dP_dropped_2d = torch.empty((bs * n_kv_heads, n_groups * seq_q, seq_kv),
                                 dtype=torch.bfloat16, device=device)
    dV_flat = torch.empty((bs * n_kv_heads, seq_kv, HEAD_DIM),
                           dtype=torch.bfloat16, device=device)
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=device)

    # ── Step 3: Launch BMM1 on stream A, BMM2 on stream B concurrently ───────
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    default_stream = torch.cuda.current_stream()
    start_event = torch.cuda.Event()
    start_event.record(default_stream)

    # Stream A: BMM1
    with torch.cuda.stream(stream_a):
        stream_a.wait_event(start_event)
        torch.bmm(dO_2d, vs_T_2d, out=dP_dropped_2d)

    # Stream B: BMM2
    with torch.cuda.stream(stream_b):
        stream_b.wait_event(start_event)
        torch.bmm(P_dropped_2d_T, dO_2d, out=dV_flat)

    # ── Step 4: After BMM1 completes, run Triton softmax-bwd on stream A ─────
    dS_flat = dS.reshape(total_rows, seq_kv)
    dP_dropped_flat = dP_dropped_2d.reshape(total_rows, seq_kv)

    with torch.cuda.stream(stream_a):
        _softmax_bwd_kernel[(grid_size,)](
            dP_dropped_flat,
            attn_weights_flat,
            dropout_mask_flat,
            dS_flat,
            total_rows,
            seq_kv,
            inv_keep_prob,
            BLOCK_KV=BLOCK_KV,
            SINGLE_PASS=SINGLE_PASS,
            ROWS_PER_CTA=ROWS_PER_CTA,
            num_warps=4,
        )

    # ── Step 5: Sync both streams back to the default stream ─────────────────
    event_a = torch.cuda.Event()
    event_b = torch.cuda.Event()
    event_a.record(stream_a)
    event_b.record(stream_b)

    default_stream.wait_event(event_a)
    default_stream.wait_event(event_b)

    # ── Step 6: Reshape outputs ───────────────────────────────────────────────
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV
