"""
Optimized attention-backward kernel using BF16 GEMMs + Triton softmax backward.

Strategy:
  - GEMM1 (dP = dO @ V^T) as cuBLAS BMM on stream1
  - GEMM2 (dV) as cuBLAS BMM on stream2 (launched first for overlap)
  - Softmax backward as Triton kernel on current stream after waiting on stream1
  - Use attn_weights_dropped instead of (attn_weights + dropout_mask) to eliminate
    one HBM tensor load from the softmax-bwd kernel inner loop
  - GQA-aware GEMM avoiding V expansion

The softmax-bwd formula using attn_weights_dropped (Pd):
  dP_masked[q,k] = dP_raw[q,k] * sign(Pd[q,k]) * inv_keep
                 = tl.where(Pd[q,k] > 0, dP_raw[q,k] * inv_keep, 0.0)
  rowsum[q]      = sum_k(dP_masked[q,k] * P[q,k])
  dS[q,k]        = P[q,k] * (dP_masked[q,k] - rowsum[q])

This reads only dP_raw (f32), Pd (bf16), P (bf16) -- no bool mask load.

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

# Pre-create CUDA streams for overlapping the two independent GEMMs
_stream1 = None
_stream2 = None

def _get_streams():
    global _stream1, _stream2
    if _stream1 is None:
        _stream1 = torch.cuda.Stream()
        _stream2 = torch.cuda.Stream()
    return _stream1, _stream2


@triton.autotune(
    configs=[
        # Two-tensor loads: dP_raw (f32) + Pd (bf16) + P (bf16) in inner loop
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_KV': 64},   num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_KV': 128},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_KV': 256},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_KV': 64},   num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_KV': 128},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_KV': 256},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_KV': 64},   num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_KV': 128},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_KV': 256},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_KV': 64},   num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_KV': 128},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_KV': 256},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 64, 'BLOCK_KV': 64},   num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 64, 'BLOCK_KV': 128},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 64, 'BLOCK_KV': 64},   num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 64, 'BLOCK_KV': 128},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_KV': 64},   num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_KV': 128},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_KV': 64},   num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_KV': 128},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_KV': 64},   num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_KV': 128},  num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_KV': 128},  num_warps=16, num_stages=2),
    ],
    key=['sq', 'skv'],
)
@triton.jit
def softmax_bwd_kernel(
    # dP_raw: [bs, 80, sq, skv] float32 -- result of GEMM1
    dP_ptr,  stride_dp_bs,  stride_dp_h,  stride_dp_sq,  stride_dp_skv,
    # P (attn_weights): [bs, 80, sq, skv] bfloat16
    P_ptr,   stride_p_bs,   stride_p_h,   stride_p_sq,   stride_p_skv,
    # Pd (attn_weights_dropped): [bs, 80, sq, skv] bfloat16
    Pd_ptr,  stride_pd_bs,  stride_pd_h,  stride_pd_sq,  stride_pd_skv,
    # dS output: [bs, 80, sq, skv] bfloat16
    dS_ptr,  stride_ds_bs,  stride_ds_h,  stride_ds_sq,  stride_ds_skv,
    bs, n_heads, sq, skv,
    inv_keep: tl.constexpr,   # 1.0 / (1.0 - dropout)
    BLOCK_SQ: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """
    Softmax backward using attn_weights_dropped to avoid loading bool mask.

    Formula:
      dP_masked[q,k] = tl.where(Pd[q,k] > 0, dP_raw[q,k] * inv_keep, 0.0)
      rowsum[q]      = sum_k(dP_masked[q,k] * P[q,k])
      dS[q,k]        = P[q,k] * (dP_masked[q,k] - rowsum[q])

    Grid: (bs * n_heads, cdiv(sq, BLOCK_SQ))
    """
    pid_bh = tl.program_id(0)
    pid_sq = tl.program_id(1)

    batch_idx = pid_bh // n_heads
    head      = pid_bh % n_heads

    sq_start = pid_sq * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)   # [BLOCK_SQ]
    sq_mask  = sq_offs < sq

    kv_offs  = tl.arange(0, BLOCK_KV)               # [BLOCK_KV]

    # Base pointers
    dP_base = dP_ptr + batch_idx * stride_dp_bs + head * stride_dp_h
    P_base  = P_ptr  + batch_idx * stride_p_bs  + head * stride_p_h
    Pd_base = Pd_ptr + batch_idx * stride_pd_bs + head * stride_pd_h
    dS_base = dS_ptr + batch_idx * stride_ds_bs + head * stride_ds_h

    num_kv_blocks = tl.cdiv(skv, BLOCK_KV)

    # ---- Pass 1: compute rowsum ----
    rowsum = tl.zeros((BLOCK_SQ,), dtype=tl.float32)

    for kv_tile in range(num_kv_blocks):
        kv_start    = kv_tile * BLOCK_KV
        kv_tile_off = kv_start + kv_offs
        kv_mask     = kv_tile_off < skv
        combined    = sq_mask[:, None] & kv_mask[None, :]

        # Load dP_raw tile (float32)
        dP_ptrs = dP_base + sq_offs[:, None] * stride_dp_sq + kv_tile_off[None, :] * stride_dp_skv
        dP_tile = tl.load(dP_ptrs, mask=combined, other=0.0)  # already float32

        # Load Pd tile (bfloat16 -> float32)
        Pd_ptrs = Pd_base + sq_offs[:, None] * stride_pd_sq + kv_tile_off[None, :] * stride_pd_skv
        Pd_tile = tl.load(Pd_ptrs, mask=combined, other=0.0).to(tl.float32)

        # Load P tile (bfloat16 -> float32)
        P_ptrs  = P_base  + sq_offs[:, None] * stride_p_sq  + kv_tile_off[None, :] * stride_p_skv
        P_tile  = tl.load(P_ptrs,  mask=combined, other=0.0).to(tl.float32)

        # Compute masked dP using Pd sign (avoids bool mask load)
        dP_masked = tl.where(Pd_tile > 0.0, dP_tile * inv_keep, 0.0)

        # Accumulate rowsum = sum(dP_masked * P)
        rowsum += tl.sum(dP_masked * P_tile, axis=1)

    # ---- Pass 2: compute dS, store ----
    for kv_tile in range(num_kv_blocks):
        kv_start    = kv_tile * BLOCK_KV
        kv_tile_off = kv_start + kv_offs
        kv_mask     = kv_tile_off < skv
        combined    = sq_mask[:, None] & kv_mask[None, :]

        dP_ptrs = dP_base + sq_offs[:, None] * stride_dp_sq + kv_tile_off[None, :] * stride_dp_skv
        dP_tile = tl.load(dP_ptrs, mask=combined, other=0.0)

        Pd_ptrs = Pd_base + sq_offs[:, None] * stride_pd_sq + kv_tile_off[None, :] * stride_pd_skv
        Pd_tile = tl.load(Pd_ptrs, mask=combined, other=0.0).to(tl.float32)

        P_ptrs  = P_base  + sq_offs[:, None] * stride_p_sq  + kv_tile_off[None, :] * stride_p_skv
        P_tile  = tl.load(P_ptrs,  mask=combined, other=0.0).to(tl.float32)

        dP_masked = tl.where(Pd_tile > 0.0, dP_tile * inv_keep, 0.0)

        # dS = P * (dP_masked - rowsum)
        dS_tile = P_tile * (dP_masked - rowsum[:, None])

        dS_ptrs = dS_base + sq_offs[:, None] * stride_ds_sq + kv_tile_off[None, :] * stride_ds_skv
        tl.store(dS_ptrs, dS_tile.to(tl.bfloat16), mask=combined)


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

    device = grad_attn_output.device
    inv_keep = 1.0 / (1.0 - attention_dropout)

    # ---- Prepare dO in [bs, 80, seq_q, 128] layout ----
    # grad_attn_output: [bs, seq_q, 80, 128] -> permute -> [bs, 80, seq_q, 128]
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()  # [bs, 80, sq, d]

    # ---- Prepare for GEMM2 (dV) ----
    # dO: [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    dO_5d = dO.view(bs, n_kv_heads, n_groups, seq_q, d)
    dO_for_dV = dO_5d.reshape(bs * n_kv_heads, n_groups * seq_q, d).contiguous()

    # P_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
    P_dropped_5d = attn_weights_dropped.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    P_dropped_gqa = P_dropped_5d.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # Pre-allocate output tensors
    dV_gqa = torch.empty((bs * n_kv_heads, seq_kv, d), dtype=torch.bfloat16, device=device)
    dS     = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=device)

    # Pre-allocate dP_raw (float32) for GEMM1 output
    dP_raw = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.float32, device=device)

    # Get streams
    stream1, stream2 = _get_streams()
    current_stream = torch.cuda.current_stream(device)

    # ---- GEMM2 (BF16): dV on stream2 -- launched FIRST for maximum overlap ----
    with torch.cuda.stream(stream2):
        stream2.wait_stream(current_stream)
        # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
        torch.bmm(P_dropped_gqa.transpose(-2, -1), dO_for_dV, out=dV_gqa)

    # ---- GEMM1: dP_raw = dO @ V^T on stream1 ----
    # dO: [bs, 80, sq, d], V: [bs, 8, skv, d]
    # For GQA: expand V view [bs, 8, skv, d] -> [bs*8, skv, d]
    # dO: [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    # But we need [bs*80, sq, d] @ [bs*80, d, skv] -> [bs*80, sq, skv]
    # Map each head to its kv_head: head h -> kv_head h//10
    # Use [bs*8, 10, sq, d] @ [bs*8, 1, d, skv] -> [bs*8, 10, sq, skv]
    dO_gemm1 = dO_5d.reshape(bs * n_kv_heads, n_groups * seq_q, d)  # [bs*8, 10*sq, d]
    V_gemm1 = value_states.reshape(bs * n_kv_heads, seq_kv, d)      # [bs*8, skv, d]

    with torch.cuda.stream(stream1):
        stream1.wait_stream(current_stream)
        # [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
        dP_raw_grouped = torch.bmm(dO_gemm1, V_gemm1.transpose(-2, -1))
        # Reshape to [bs, 80, sq, skv]
        dP_raw.copy_(dP_raw_grouped.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)
                                   .reshape(bs, n_heads, seq_q, seq_kv))

    # Wait for GEMM1 to complete before softmax-bwd
    current_stream.wait_stream(stream1)

    # ---- Softmax backward kernel on current stream ----
    # Ensures GEMM2 can overlap with this kernel
    P_contig  = attn_weights if attn_weights.is_contiguous() else attn_weights.contiguous()
    Pd_contig = attn_weights_dropped if attn_weights_dropped.is_contiguous() else attn_weights_dropped.contiguous()

    grid_dS = lambda meta: (bs * n_heads, triton.cdiv(seq_q, meta['BLOCK_SQ']))
    softmax_bwd_kernel[grid_dS](
        dP_raw,
        dP_raw.stride(0), dP_raw.stride(1), dP_raw.stride(2), dP_raw.stride(3),
        P_contig,
        P_contig.stride(0), P_contig.stride(1), P_contig.stride(2), P_contig.stride(3),
        Pd_contig,
        Pd_contig.stride(0), Pd_contig.stride(1), Pd_contig.stride(2), Pd_contig.stride(3),
        dS,
        dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
        bs, n_heads, seq_q, seq_kv,
        inv_keep=inv_keep,
    )

    # Wait for GEMM2 to complete
    current_stream.wait_stream(stream2)

    # Reshape dV to [bs, 8, skv, d]
    dV = dV_gqa.view(bs, n_kv_heads, seq_kv, d)

    return dS, dV
