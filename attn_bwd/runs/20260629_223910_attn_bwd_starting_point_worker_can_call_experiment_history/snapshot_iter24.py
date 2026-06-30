"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- Custom Triton transpose kernel for dO: [bs, sq, 80, d] -> [bs, 80, sq, d]
- Custom Triton transpose kernel for vs_T: [bs, 8, skv, d] -> [bs*8, d, skv]
  (replaces .transpose().reshape().contiguous() which materializes a strided copy)
- Both BMMs as clean 3D batched GEMMs (cuBLAS-optimized, no broadcasting)
- BMM1: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
- BMM2 fused with GQA: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
- Dual-stream pipelining with module-level cached streams/events
  (eliminates Python object allocation overhead on every call)
- Row-batched Triton softmax-bwd kernel on stream A after BMM1
  (overlaps with BMM2 on stream B)
- Persistent buffer cache: pre-allocated tensors keyed by (bs, seq_q, seq_kv)
  to eliminate torch.empty() allocation overhead on the hot path
- Shape-adaptive dispatch: skip multi-stream overhead for small workloads
"""

import torch
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128

# ─────────────────────────────────────────────────────────────────────────────
# Module-level cached CUDA streams and events
# Created once at import/first-call time, reused on every subsequent call
# ─────────────────────────────────────────────────────────────────────────────
_stream_a = None
_stream_b = None
_start_event = None
_event_a = None
_event_b = None


def _ensure_streams():
    global _stream_a, _stream_b, _start_event, _event_a, _event_b
    if _stream_a is None:
        _stream_a = torch.cuda.Stream()
        _stream_b = torch.cuda.Stream()
        _start_event = torch.cuda.Event()
        _event_a = torch.cuda.Event()
        _event_b = torch.cuda.Event()


# ─────────────────────────────────────────────────────────────────────────────
# Persistent buffer cache: keyed by (bs, seq_q, seq_kv)
# Stores pre-allocated tensors: (dO, vs_T_2d, dP_dropped_2d, dV_flat, dS)
# ─────────────────────────────────────────────────────────────────────────────
_buffer_cache = {}


def _get_buffers(bs, seq_q, seq_kv, device):
    key = (bs, seq_q, seq_kv)
    if key not in _buffer_cache:
        n_kv_heads = NUM_KEY_VALUE_HEADS
        n_heads    = NUM_ATTENTION_HEADS
        n_groups   = n_heads // n_kv_heads
        n_bkv      = bs * n_kv_heads

        dO         = torch.empty((bs, n_heads, seq_q, HEAD_DIM),
                                  dtype=torch.bfloat16, device=device)
        vs_T_2d    = torch.empty((n_bkv, HEAD_DIM, seq_kv),
                                  dtype=torch.bfloat16, device=device)
        dP_dropped_2d = torch.empty((n_bkv, n_groups * seq_q, seq_kv),
                                     dtype=torch.bfloat16, device=device)
        dV_flat    = torch.empty((n_bkv, seq_kv, HEAD_DIM),
                                  dtype=torch.bfloat16, device=device)
        dS         = torch.empty((bs, n_heads, seq_q, seq_kv),
                                  dtype=torch.bfloat16, device=device)
        _buffer_cache[key] = (dO, vs_T_2d, dP_dropped_2d, dV_flat, dS)
    return _buffer_cache[key]


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: tiled transpose [bs, sq, n_heads, d] -> [bs, n_heads, sq, d]
#
# Swaps dims 1 (sq) and 2 (n_heads), keeps d as inner dim.
# Each CTA handles a TILE_SQ x TILE_H tile for one batch element.
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
    batch_idx   = pid // tile_per_batch
    tile_idx    = pid % tile_per_batch
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
# Triton kernel: transpose-reshape value_states
#   Input:  [bs * n_kv_heads, skv, d]    (contiguous; view of [bs,8,skv,d])
#   Output: [bs * n_kv_heads, d, skv]    (contiguous)
#
# This replaces: value_states.transpose(-2,-1).reshape(bs*8,d,skv).contiguous()
# Each CTA handles a TILE_SKV x TILE_D tile of (skv, d) for one (b, kv_head).
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _transpose_vs_kernel(
    src_ptr,              # [n_bkv, skv, d]  bfloat16  (contiguous)
    dst_ptr,              # [n_bkv, d, skv]  bfloat16  (output)
    n_bkv,                # bs * n_kv_heads
    skv,                  # seq_kv
    HEAD_DIM: tl.constexpr,
    TILE_SKV: tl.constexpr,
    TILE_D: tl.constexpr,
):
    num_tiles_skv = tl.cdiv(skv, TILE_SKV)
    num_tiles_d   = tl.cdiv(HEAD_DIM, TILE_D)

    pid = tl.program_id(0)
    tiles_per_bkv = num_tiles_skv * num_tiles_d
    bkv_idx  = pid // tiles_per_bkv
    tile_idx = pid % tiles_per_bkv
    tile_skv = tile_idx % num_tiles_skv
    tile_d   = tile_idx // num_tiles_skv

    skv_start = tile_skv * TILE_SKV
    d_start   = tile_d   * TILE_D

    offs_skv = skv_start + tl.arange(0, TILE_SKV)  # [TILE_SKV]
    offs_d   = d_start   + tl.arange(0, TILE_D)    # [TILE_D]

    valid_skv = offs_skv < skv
    valid_d   = offs_d   < HEAD_DIM
    valid_mask = valid_skv[:, None] & valid_d[None, :]

    # Source layout: [n_bkv, skv, d]
    # Element [bkv, s, d_] is at bkv * skv * HEAD_DIM + s * HEAD_DIM + d_
    src_base = bkv_idx * skv * HEAD_DIM
    src_offsets = (src_base
                   + offs_skv[:, None] * HEAD_DIM   # [TILE_SKV, 1]
                   + offs_d[None, :])                # [1, TILE_D]
    # vals shape: [TILE_SKV, TILE_D]
    vals = tl.load(src_ptr + src_offsets, mask=valid_mask, other=0.0)

    # Destination layout: [n_bkv, d, skv]
    # Element [bkv, d_, s] is at bkv * HEAD_DIM * skv + d_ * skv + s
    # We store vals[skv_local, d_local] at dst[bkv, d_start+d_local, skv_start+skv_local]
    # = dst_base + (d_start + d_local) * skv + (skv_start + skv_local)
    dst_base = bkv_idx * HEAD_DIM * skv
    dst_offsets = (dst_base
                   + offs_d[None, :] * skv            # [1, TILE_D] broadcast
                   + offs_skv[:, None])                # [TILE_SKV, 1] broadcast
    # dst_offsets shape: [TILE_SKV, TILE_D] — same shape as vals ✓
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


# Threshold for shape-adaptive dispatch: total elements
# Below this threshold, multi-stream overhead may dominate; use sequential path
_STREAM_THRESHOLD = 80 * 512 * 512  # ~20M elements


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
    n_bkv  = bs * n_kv_heads

    # Ensure module-level streams and events are initialized (cached)
    _ensure_streams()

    # Get or allocate persistent buffers for this shape
    dO, vs_T_2d, dP_dropped_2d, dV_flat, dS = _get_buffers(bs, seq_q, seq_kv, device)

    # ── Step 1: Custom tiled transpose [bs, sq, 80, d] -> [bs, 80, sq, d] ────
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
    dO_2d = dO.reshape(n_bkv, n_groups * seq_q, HEAD_DIM)

    # ── Step 2: Transpose value_states via Triton kernel ─────────────────────
    TILE_SKV = 16
    TILE_D   = 16
    num_tiles_skv = triton.cdiv(seq_kv, TILE_SKV)
    num_tiles_d   = triton.cdiv(HEAD_DIM, TILE_D)
    vs_transpose_grid = n_bkv * num_tiles_skv * num_tiles_d

    # value_states is [bs, 8, skv, d] — reshape to [bs*8, skv, d] for the kernel
    vs_2d = value_states.reshape(n_bkv, seq_kv, HEAD_DIM)
    if not vs_2d.is_contiguous():
        vs_2d = vs_2d.contiguous()

    _transpose_vs_kernel[(vs_transpose_grid,)](
        vs_2d,
        vs_T_2d,
        n_bkv, seq_kv,
        HEAD_DIM=HEAD_DIM,
        TILE_SKV=TILE_SKV,
        TILE_D=TILE_D,
    )

    # ── Step 3: Prepare BMM2 input ────────────────────────────────────────────
    P_dropped_2d = attn_weights_dropped.reshape(n_bkv, n_groups * seq_q, seq_kv)
    P_dropped_2d_T = P_dropped_2d.transpose(-2, -1)

    # Flatten for Triton softmax-bwd kernel
    total_rows = bs * n_heads * seq_q
    attn_weights_flat = attn_weights.reshape(total_rows, seq_kv)
    dropout_mask_flat = dropout_mask.reshape(total_rows, seq_kv)

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0
    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 16384)
    SINGLE_PASS = (seq_kv <= BLOCK_KV)
    ROWS_PER_CTA = 4
    grid_size = triton.cdiv(total_rows, ROWS_PER_CTA)

    dS_flat = dS.reshape(total_rows, seq_kv)
    dP_dropped_flat = dP_dropped_2d.reshape(total_rows, seq_kv)

    # ── Shape-adaptive dispatch ───────────────────────────────────────────────
    # For small workloads, skip multi-stream overhead (stream fork/join/sync cost
    # can exceed the benefit of overlapping two independent BMMs).
    workload_size = bs * seq_q * seq_kv * n_heads
    use_streams = (workload_size >= _STREAM_THRESHOLD)

    if use_streams:
        # ── Step 4a: Launch both BMMs concurrently on separate streams ────────
        default_stream = torch.cuda.current_stream()
        _start_event.record(default_stream)

        # Stream A: BMM1 → softmax-bwd
        with torch.cuda.stream(_stream_a):
            _stream_a.wait_event(_start_event)
            # BMM1: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
            torch.bmm(dO_2d, vs_T_2d, out=dP_dropped_2d)

        # Stream B: BMM2
        with torch.cuda.stream(_stream_b):
            _stream_b.wait_event(_start_event)
            # BMM2: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
            torch.bmm(P_dropped_2d_T, dO_2d, out=dV_flat)

        # ── Step 5a: Softmax-bwd on stream A (overlaps with BMM2 on stream B) ─
        with torch.cuda.stream(_stream_a):
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

        # ── Step 6a: Sync both streams back to the default stream ─────────────
        _event_a.record(_stream_a)
        _event_b.record(_stream_b)
        default_stream.wait_event(_event_a)
        default_stream.wait_event(_event_b)

    else:
        # ── Step 4b: Sequential path for small workloads ──────────────────────
        # BMM1: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
        torch.bmm(dO_2d, vs_T_2d, out=dP_dropped_2d)

        # Softmax-bwd
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

        # BMM2: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
        torch.bmm(P_dropped_2d_T, dO_2d, out=dV_flat)

    # ── Step 7: Reshape outputs ────────────────────────────────────────────────
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV
