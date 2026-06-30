"""
Optimized attention-backward kernel using BF16 GEMMs + fused Triton softmax backward.

Strategy:
  Fuse GEMM1 (dP = dO @ V^T) into the softmax backward Triton kernel to eliminate
  the intermediate dP tensor from HBM entirely. The fused kernel:
    1. For each (batch*head, sq_block), loads dO slice into SRAM once.
    2. Tiles over kv dimension, computing dP_tile = dO @ V_tile^T in registers.
    3. Pass 1: accumulate rowsum = sum(dP_tile * P_dropped_tile) over kv tiles.
    4. Pass 2: recompute dP_tile, compute dS = P_dropped * dP - P * rowsum, store.

  GQA mapping: head h -> kv-head h // 10

  GEMM2 (dV) stays as cuBLAS BMM on a separate stream.

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
        # BLOCK_SQ x BLOCK_KV x HEAD_DIM -- tile over kv, load dO slice once
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_KV': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_KV': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_KV': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_KV': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_KV': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_KV': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_KV': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_KV': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SQ': 64, 'BLOCK_KV': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SQ': 64, 'BLOCK_KV': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SQ': 64, 'BLOCK_KV': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SQ': 64, 'BLOCK_KV': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_KV': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_KV': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_KV': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_KV': 128}, num_warps=4, num_stages=3),
    ],
    key=['sq', 'skv'],
)
@triton.jit
def fused_softmax_bwd_kernel(
    # dO: [bs, seq_q, 80, 128] bfloat16 -- native layout (not transposed)
    dO_ptr, stride_do_bs, stride_do_sq, stride_do_h, stride_do_d,
    # V:  [bs, 8, seq_kv, 128] bfloat16
    V_ptr,  stride_v_bs,  stride_v_kv, stride_v_skv, stride_v_d,
    # P (attn_weights): [bs, 80, sq, skv] bfloat16
    P_ptr,  stride_p_bs,  stride_p_h,  stride_p_sq,  stride_p_skv,
    # Pd (attn_weights_dropped): [bs, 80, sq, skv] bfloat16
    Pd_ptr, stride_pd_bs, stride_pd_h, stride_pd_sq, stride_pd_skv,
    # dS output: [bs, 80, sq, skv] bfloat16
    dS_ptr, stride_ds_bs, stride_ds_h, stride_ds_sq, stride_ds_skv,
    bs, n_heads, sq, skv,
    n_groups: tl.constexpr,   # 10
    HEAD_DIM: tl.constexpr,   # 128
    BLOCK_SQ: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """
    Fused kernel: computes dP = dO @ V^T on-the-fly inside softmax backward.
    Eliminates the intermediate dP tensor from HBM.

    Grid: (bs * n_heads, cdiv(sq, BLOCK_SQ))
    Each program handles BLOCK_SQ rows of a specific (batch, head).

    For GQA: kv_head = head // n_groups
    dO layout: [bs, sq, 80, 128] -- indexed as [b, q, h, d]
    V  layout: [bs, 8, skv, 128] -- indexed as [b, kv_h, k, d]

    Pass 1: rowsum[q] = sum_k(dP[q,k] * Pd[q,k])
            where dP[q,k] = sum_d(dO[q,d] * V[k,d])
    Pass 2: dS[q,k] = Pd[q,k]*dP[q,k] - P[q,k]*rowsum[q]
    """
    pid_bh = tl.program_id(0)
    pid_sq = tl.program_id(1)

    batch_idx = pid_bh // n_heads
    head      = pid_bh % n_heads
    kv_head   = head // n_groups  # GQA: map head to kv_head

    sq_start = pid_sq * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)   # [BLOCK_SQ]
    sq_mask  = sq_offs < sq

    d_offs   = tl.arange(0, HEAD_DIM)               # [HEAD_DIM]
    kv_offs  = tl.arange(0, BLOCK_KV)               # [BLOCK_KV]

    # Base pointers
    dO_base = dO_ptr + batch_idx * stride_do_bs + head * stride_do_h
    V_base  = V_ptr  + batch_idx * stride_v_bs  + kv_head * stride_v_kv
    P_base  = P_ptr  + batch_idx * stride_p_bs  + head * stride_p_h
    Pd_base = Pd_ptr + batch_idx * stride_pd_bs + head * stride_pd_h
    dS_base = dS_ptr + batch_idx * stride_ds_bs + head * stride_ds_h

    # Load dO tile: [BLOCK_SQ, HEAD_DIM] -- loaded ONCE and reused in both passes
    # dO layout: [bs, sq, 80, 128] -> stride_do_sq, stride_do_d
    dO_ptrs = dO_base + sq_offs[:, None] * stride_do_sq + d_offs[None, :] * stride_do_d
    dO_tile = tl.load(dO_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)
    # dO_tile: [BLOCK_SQ, HEAD_DIM]

    num_kv_blocks = tl.cdiv(skv, BLOCK_KV)

    # ---- Pass 1: compute rowsum ----
    rowsum = tl.zeros((BLOCK_SQ,), dtype=tl.float32)

    for kv_tile in range(num_kv_blocks):
        kv_start    = kv_tile * BLOCK_KV
        kv_tile_off = kv_start + kv_offs   # [BLOCK_KV]
        kv_mask     = kv_tile_off < skv

        # Load V tile: [BLOCK_KV, HEAD_DIM]
        V_ptrs = V_base + kv_tile_off[:, None] * stride_v_skv + d_offs[None, :] * stride_v_d
        V_tile = tl.load(V_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        # V_tile: [BLOCK_KV, HEAD_DIM]

        # Compute dP_tile = dO_tile @ V_tile^T: [BLOCK_SQ, BLOCK_KV]
        dP_tile = tl.dot(dO_tile, V_tile.T)   # [BLOCK_SQ, BLOCK_KV]

        # Load Pd tile: [BLOCK_SQ, BLOCK_KV]
        combined_mask = sq_mask[:, None] & kv_mask[None, :]
        Pd_ptrs = Pd_base + sq_offs[:, None] * stride_pd_sq + kv_tile_off[None, :] * stride_pd_skv
        Pd_tile = tl.load(Pd_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # Accumulate rowsum = sum(dP * Pd) over kv
        rowsum += tl.sum(dP_tile * Pd_tile, axis=1)   # [BLOCK_SQ]

    # ---- Pass 2: recompute dP, compute dS, store ----
    for kv_tile in range(num_kv_blocks):
        kv_start    = kv_tile * BLOCK_KV
        kv_tile_off = kv_start + kv_offs   # [BLOCK_KV]
        kv_mask     = kv_tile_off < skv
        combined_mask = sq_mask[:, None] & kv_mask[None, :]

        # Load V tile: [BLOCK_KV, HEAD_DIM]
        V_ptrs = V_base + kv_tile_off[:, None] * stride_v_skv + d_offs[None, :] * stride_v_d
        V_tile = tl.load(V_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

        # Recompute dP_tile: [BLOCK_SQ, BLOCK_KV]
        dP_tile = tl.dot(dO_tile, V_tile.T)

        # Load Pd and P tiles
        Pd_ptrs = Pd_base + sq_offs[:, None] * stride_pd_sq + kv_tile_off[None, :] * stride_pd_skv
        Pd_tile = tl.load(Pd_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        P_ptrs  = P_base  + sq_offs[:, None] * stride_p_sq  + kv_tile_off[None, :] * stride_p_skv
        P_tile  = tl.load(P_ptrs,  mask=combined_mask, other=0.0).to(tl.float32)

        # dS = Pd * dP - P * rowsum
        dS_tile = Pd_tile * dP_tile - P_tile * rowsum[:, None]

        dS_ptrs = dS_base + sq_offs[:, None] * stride_ds_sq + kv_tile_off[None, :] * stride_ds_skv
        tl.store(dS_ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


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

    # ---- Prepare for GEMM2 (dV) ----
    # dO: [bs, seq_q, 80, 128] -> [bs, seq_q, 8, 10, 128]
    dO_5d = grad_attn_output.view(bs, seq_q, n_kv_heads, n_groups, d)

    # P_dropped: [bs, 80, seq_q, seq_kv] -> [bs, 8, 10, seq_q, seq_kv]
    P_dropped_5d = attn_weights_dropped.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)

    # For dV GEMM: [bs*8, 10*seq_q, d] and [bs*8, 10*seq_q, seq_kv]
    dO_for_dV = dO_5d.permute(0, 2, 3, 1, 4).reshape(bs * n_kv_heads, n_groups * seq_q, d).contiguous()
    P_dropped_gqa = P_dropped_5d.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # Pre-allocate output tensors
    dV_gqa = torch.empty((bs * n_kv_heads, seq_kv, d), dtype=torch.bfloat16, device=device)
    dS     = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=device)

    # Get streams
    stream1, stream2 = _get_streams()
    current_stream = torch.cuda.current_stream(device)

    # ---- GEMM2 (BF16): dV on stream2 ----
    with torch.cuda.stream(stream2):
        stream2.wait_stream(current_stream)
        # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
        torch.bmm(P_dropped_gqa.transpose(-2, -1), dO_for_dV, out=dV_gqa)

    # ---- Fused Triton kernel: GEMM1 + softmax backward on stream1 ----
    # Ensure inputs are contiguous for optimal Triton access
    dO_contig  = grad_attn_output if grad_attn_output.is_contiguous() else grad_attn_output.contiguous()
    V_contig   = value_states if value_states.is_contiguous() else value_states.contiguous()
    P_contig   = attn_weights if attn_weights.is_contiguous() else attn_weights.contiguous()
    Pd_contig  = attn_weights_dropped if attn_weights_dropped.is_contiguous() else attn_weights_dropped.contiguous()

    with torch.cuda.stream(stream1):
        stream1.wait_stream(current_stream)
        grid_dS = lambda meta: (bs * n_heads, triton.cdiv(seq_q, meta['BLOCK_SQ']))
        fused_softmax_bwd_kernel[grid_dS](
            dO_contig,
            dO_contig.stride(0), dO_contig.stride(1), dO_contig.stride(2), dO_contig.stride(3),
            V_contig,
            V_contig.stride(0), V_contig.stride(1), V_contig.stride(2), V_contig.stride(3),
            P_contig,
            P_contig.stride(0), P_contig.stride(1), P_contig.stride(2), P_contig.stride(3),
            Pd_contig,
            Pd_contig.stride(0), Pd_contig.stride(1), Pd_contig.stride(2), Pd_contig.stride(3),
            dS,
            dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
            bs, n_heads, seq_q, seq_kv,
            n_groups=n_groups,
            HEAD_DIM=HEAD_DIM,
        )

    # Synchronize both streams back to current stream
    current_stream.wait_stream(stream1)
    current_stream.wait_stream(stream2)

    # Reshape dV to [bs, 8, skv, d]
    dV = dV_gqa.view(bs, n_kv_heads, seq_kv, d)

    return dS, dV
