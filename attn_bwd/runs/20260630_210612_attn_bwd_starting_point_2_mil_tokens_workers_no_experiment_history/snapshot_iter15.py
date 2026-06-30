"""
Optimized attention-backward kernel using BF16 GEMM for dV + fused Triton kernel for dS.

Strategy:
  1. Fused dS kernel: eliminate the intermediate dP tensor from HBM entirely.
     - For each (batch, head, sq_tile) CTA, loop over kv-tiles twice:
       Pass 1: compute dP_tile = dot(dO_tile, V_tile^T), apply dropout mask,
               accumulate rowsum += sum(dP_masked * P_tile)
       Pass 2: recompute dP_tile, reapply mask, compute dS_tile = P_tile * (dP_masked - rowsum)
     - dO_tile is loaded once per sq_tile and held in registers across the kv loop.
     - V is accessed at kv-head index head // n_groups (GQA mapping).
     - No intermediate dP written to HBM — eliminates one full [bs,80,sq,skv] read/write.

  2. For dV: use GQA-aware BF16 BMM (unchanged from iteration #12).
     - dO: [bs,8,10,sq,d] -> [bs*8, 10*sq, d]
     - P_dropped: [bs,8,10,sq,skv] -> [bs*8, 10*sq, skv]
     - bmm(P^T, dO) -> [bs*8, skv, d] = dV_gqa, sum over groups already implicit.

  3. Stream parallelism: fused dS kernel and dV GEMM launched on separate streams.

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

# Pre-create CUDA streams for overlapping the two independent computations
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
        # BLOCK_SQ x BLOCK_D x BLOCK_SKV configurations
        # BLOCK_D must equal HEAD_DIM=128 (constexpr)
        # Small BLOCK_SQ, various BLOCK_SKV, num_stages
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 64},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 128}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 64},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 128}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_SKV': 64},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_SKV': 128}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_SKV': 64},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_SKV': 128}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 64, 'BLOCK_SKV': 64},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 64, 'BLOCK_SKV': 128}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_SKV': 64},  num_warps=4,  num_stages=3),
        # Larger warps for B200
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 128}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_SKV': 128}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SQ': 64, 'BLOCK_SKV': 64},  num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SQ': 64, 'BLOCK_SKV': 128}, num_warps=16, num_stages=2),
        # num_stages=4 for better latency hiding on B200 HBM
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 64},  num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 128}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_SKV': 64},  num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_SKV': 128}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_SQ': 64, 'BLOCK_SKV': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SQ': 64, 'BLOCK_SKV': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_SKV': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_SKV': 128}, num_warps=8,  num_stages=4),
        # num_stages=3 variety
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 128}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_SKV': 128}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SQ': 64, 'BLOCK_SKV': 64},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SQ': 64, 'BLOCK_SKV': 128}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SQ': 64, 'BLOCK_SKV': 128}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SQ': 64, 'BLOCK_SKV': 128}, num_warps=16, num_stages=4),
        # BLOCK_SKV=256 for large seq_kv (reduces loop iterations)
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 256}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_SKV': 256}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 256}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_SKV': 256}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 256}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_SKV': 256}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 256}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SQ': 32, 'BLOCK_SKV': 256}, num_warps=8,  num_stages=4),
    ],
    key=['sq', 'skv'],
)
@triton.jit
def fused_ds_kernel(
    # dO: [bs, 80, sq, d] bfloat16  (already transposed from [bs, sq, 80, d])
    dO_ptr, stride_do_bs, stride_do_h, stride_do_sq, stride_do_d,
    # V (unexpanded): [bs, 8, skv, d] bfloat16
    V_ptr, stride_v_bs, stride_v_kvh, stride_v_skv, stride_v_d,
    # P (attn_weights): [bs, 80, sq, skv] bfloat16
    P_ptr, stride_p_bs, stride_p_h, stride_p_sq, stride_p_skv,
    # dropout_mask: [bs, 80, sq, skv] bool
    mask_ptr, stride_m_bs, stride_m_h, stride_m_sq, stride_m_skv,
    # dS output: [bs, 80, sq, skv] bfloat16
    dS_ptr, stride_ds_bs, stride_ds_h, stride_ds_sq, stride_ds_skv,
    bs, n_heads, n_kv_heads, sq, skv,
    inv_keep_prob: tl.constexpr,
    n_groups: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused dS kernel: computes dS = P * (dP_masked - rowsum) without writing dP to HBM.
    
    Grid: (bs * n_heads, cdiv(sq, BLOCK_SQ))
    
    For each (batch, head, sq_tile):
      Pass 1: for each kv-tile, compute dP_tile = dO_tile @ V_tile^T, accumulate rowsum
      Pass 2: for each kv-tile, recompute dP_tile, compute and store dS_tile
    
    dO_tile [BLOCK_SQ, BLOCK_D] stays in registers across the kv loop.
    GQA: V is accessed at kv_head = head // n_groups.
    """
    pid_bh = tl.program_id(0)
    pid_sq = tl.program_id(1)

    batch_idx = pid_bh // n_heads
    head      = pid_bh % n_heads
    kv_head   = head // n_groups   # GQA: map query head to kv head

    sq_start = pid_sq * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < sq

    d_offs   = tl.arange(0, BLOCK_D)
    skv_offs = tl.arange(0, BLOCK_SKV)

    # Base pointers for this (batch, head) tile
    dO_base  = dO_ptr  + batch_idx * stride_do_bs + head    * stride_do_h
    V_base   = V_ptr   + batch_idx * stride_v_bs  + kv_head * stride_v_kvh
    P_base   = P_ptr   + batch_idx * stride_p_bs  + head    * stride_p_h
    M_base   = mask_ptr+ batch_idx * stride_m_bs  + head    * stride_m_h
    dS_base  = dS_ptr  + batch_idx * stride_ds_bs + head    * stride_ds_h

    num_skv_blocks = tl.cdiv(skv, BLOCK_SKV)

    # Load dO_tile [BLOCK_SQ, BLOCK_D] once — hold in registers
    do_ptrs = dO_base + sq_offs[:, None] * stride_do_sq + d_offs[None, :] * stride_do_d
    do_mask = sq_mask[:, None]  # d_offs always valid since BLOCK_D == HEAD_DIM
    dO_tile = tl.load(do_ptrs, mask=do_mask, other=0.0).to(tl.float32)

    # ----- Pass 1: compute rowsum(dP_masked * P) -----
    rowsum = tl.zeros((BLOCK_SQ,), dtype=tl.float32)

    for skv_tile in range(num_skv_blocks):
        skv_start     = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask_t    = skv_tile_offs < skv

        combined_mask = sq_mask[:, None] & skv_mask_t[None, :]

        # Load V_tile [BLOCK_SKV, BLOCK_D] — transpose for dot product
        v_ptrs  = V_base + skv_tile_offs[:, None] * stride_v_skv + d_offs[None, :] * stride_v_d
        V_tile  = tl.load(v_ptrs, mask=skv_mask_t[:, None], other=0.0).to(tl.float32)
        # dP_tile = dO_tile @ V_tile^T: [BLOCK_SQ, BLOCK_D] @ [BLOCK_D, BLOCK_SKV] -> [BLOCK_SQ, BLOCK_SKV]
        dP_tile = tl.dot(dO_tile, tl.trans(V_tile))  # [BLOCK_SQ, BLOCK_SKV]

        # Load dropout mask and apply
        m_ptrs    = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile    = tl.load(m_ptrs, mask=combined_mask, other=0).to(tl.float32)
        dp_masked = dP_tile * m_tile * inv_keep_prob

        # Load P_tile and accumulate rowsum
        p_ptrs  = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile  = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        rowsum  += tl.sum(dp_masked * p_tile, axis=1)

    # ----- Pass 2: compute dS and store -----
    for skv_tile in range(num_skv_blocks):
        skv_start     = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask_t    = skv_tile_offs < skv

        combined_mask = sq_mask[:, None] & skv_mask_t[None, :]

        # Recompute dP_tile (dO stays in registers)
        v_ptrs  = V_base + skv_tile_offs[:, None] * stride_v_skv + d_offs[None, :] * stride_v_d
        V_tile  = tl.load(v_ptrs, mask=skv_mask_t[:, None], other=0.0).to(tl.float32)
        dP_tile = tl.dot(dO_tile, tl.trans(V_tile))  # [BLOCK_SQ, BLOCK_SKV]

        # Reapply dropout mask
        m_ptrs    = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile    = tl.load(m_ptrs, mask=combined_mask, other=0).to(tl.float32)
        dp_masked = dP_tile * m_tile * inv_keep_prob

        # Load P_tile and compute dS
        p_ptrs  = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile  = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        dS_tile = p_tile * (dp_masked - rowsum[:, None])

        # Store dS_tile
        ds_ptrs = dS_base + sq_offs[:, None] * stride_ds_sq + skv_tile_offs[None, :] * stride_ds_skv
        tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


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

    # grad_attn_output: [bs, sq, 80, d] -> [bs, 80, sq, d]
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()  # [bs, 80, sq, d] bfloat16

    # Reshape dO for GQA-aware dV GEMM:
    # [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    dO_grouped = dO.view(bs, n_kv_heads, n_groups, seq_q, d)
    dO_gqa = dO_grouped.reshape(bs * n_kv_heads, n_groups * seq_q, d)  # [bs*8, 10*sq, d]

    # P_dropped for dV: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
    P_dropped_gqa = attn_weights_dropped.view(bs, n_kv_heads, n_groups, seq_q, seq_kv) \
                                        .reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # Pre-allocate output tensors
    dV_gqa = torch.empty((bs * n_kv_heads, seq_kv, d), dtype=torch.bfloat16, device=device)
    dS     = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=device)

    # Get the two streams
    stream1, stream2 = _get_streams()
    current_stream = torch.cuda.current_stream(device)

    inv_keep = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    P_attn = attn_weights.contiguous()         # [bs, 80, sq, skv] bfloat16
    dmask  = dropout_mask.contiguous()         # [bs, 80, sq, skv] bool
    V_contig = value_states.contiguous()       # [bs, 8, skv, d] bfloat16

    # ---- Fused dS kernel on stream1 ----
    with torch.cuda.stream(stream1):
        stream1.wait_stream(current_stream)
        grid_dS = lambda meta: (bs * n_heads, triton.cdiv(seq_q, meta['BLOCK_SQ']))
        fused_ds_kernel[grid_dS](
            dO,     dO.stride(0),     dO.stride(1),     dO.stride(2),     dO.stride(3),
            V_contig, V_contig.stride(0), V_contig.stride(1), V_contig.stride(2), V_contig.stride(3),
            P_attn, P_attn.stride(0), P_attn.stride(1), P_attn.stride(2), P_attn.stride(3),
            dmask,  dmask.stride(0),  dmask.stride(1),  dmask.stride(2),  dmask.stride(3),
            dS,     dS.stride(0),     dS.stride(1),     dS.stride(2),     dS.stride(3),
            bs, n_heads, n_kv_heads, seq_q, seq_kv,
            inv_keep_prob=inv_keep,
            n_groups=n_groups,
            BLOCK_D=d,
        )

    # ---- GEMM 2 (BF16): dV = P_dropped_gqa^T @ dO_gqa on stream2 ----
    with torch.cuda.stream(stream2):
        stream2.wait_stream(current_stream)
        # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
        torch.bmm(P_dropped_gqa.transpose(-2, -1), dO_gqa, out=dV_gqa)

    # Synchronize both streams back to current stream before returning
    current_stream.wait_stream(stream1)
    current_stream.wait_stream(stream2)

    # Reshape dV to [bs, 8, skv, d]
    dV = dV_gqa.view(bs, n_kv_heads, seq_kv, d).to(torch.bfloat16)

    return dS, dV
