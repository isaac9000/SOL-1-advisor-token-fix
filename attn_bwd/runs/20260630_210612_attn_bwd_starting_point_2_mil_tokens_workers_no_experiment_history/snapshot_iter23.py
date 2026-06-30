"""
Optimized attention-backward kernel using BF16 GEMMs + Triton softmax backward.

Strategy:
  1. Avoid materializing V_exp [bs, 80, skv, d] entirely.
     For dP: reshape dO from [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
             use V as [bs*8, skv, d], compute bmm -> [bs*8, 10*sq, skv]
             reshape to [bs, 80, sq, skv]. No V copy needed!

  2. For dV: reshape P_dropped from [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
             reshape dO from [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
             compute [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
             This directly gives dV summed over groups -- no separate reduction!

  3. Stream parallelism: launch GEMM1 (dP) and GEMM2 (dV) on separate CUDA streams
     so they overlap in execution. Both are independent computations.

  4. Triton softmax-backward kernel: single-pass when entire skv row fits in SRAM,
     otherwise two-pass. Large BLOCK_SKV configs with num_warps=8/16 for B200.
     Uses attn_weights_dropped directly instead of bool dropout_mask — saves one
     full HBM read per pass by detecting the mask from nonzero values.

  5. Eliminate the permute().contiguous() on grad_attn_output by using as_strided
     to directly produce a [bs*8, 10*sq, d] contiguous buffer without going through
     the [bs, 80, sq, d] intermediate layout.

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
        # Single-pass configs (large BLOCK_SKV to cover full skv in one tile)
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 512},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SQ': 2,  'BLOCK_SKV': 512},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 2,  'BLOCK_SKV': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 2,  'BLOCK_SKV': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 512},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 512},  num_warps=16, num_stages=2),
        # Two-pass fallback configs (smaller tiles)
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 256},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 2,  'BLOCK_SKV': 256},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 256},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 8,  'BLOCK_SKV': 128},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 8,  'BLOCK_SKV': 256},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 128},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 64},   num_warps=8,  num_stages=2),
        # num_stages=3 variants for pipelining
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 512},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 1024}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 1024}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SQ': 2,  'BLOCK_SKV': 512},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SQ': 2,  'BLOCK_SKV': 1024}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 512},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 256},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SQ': 8,  'BLOCK_SKV': 256},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SQ': 8,  'BLOCK_SKV': 128},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 128},  num_warps=8,  num_stages=3),
    ],
    key=['sq', 'skv'],
)
@triton.jit
def softmax_bwd_kernel(
    # dP: [bs, 80, sq, skv] bfloat16
    dP_ptr, stride_dp_bs, stride_dp_h, stride_dp_sq, stride_dp_skv,
    # P (attn_weights): [bs, 80, sq, skv] bfloat16
    P_ptr, stride_p_bs, stride_p_h, stride_p_sq, stride_p_skv,
    # P_dropped (attn_weights_dropped): [bs, 80, sq, skv] bfloat16
    # Used instead of bool mask: nonzero means kept, zero means dropped
    Pd_ptr, stride_pd_bs, stride_pd_h, stride_pd_sq, stride_pd_skv,
    # dS output: [bs, 80, sq, skv] bfloat16
    dS_ptr, stride_ds_bs, stride_ds_h, stride_ds_sq, stride_ds_skv,
    bs, n_heads, sq, skv,
    inv_keep_prob: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """
    Single-pass softmax backward when BLOCK_SKV >= skv (entire row fits in one tile),
    otherwise falls back to two-pass.

    Uses attn_weights_dropped (bf16) instead of bool dropout_mask to detect which
    positions were kept: a nonzero P_dropped value means kept, zero means dropped.
    This eliminates one separate HBM tensor load compared to loading a bool mask.

    dp_masked = dP_raw * (P_dropped != 0).float() * inv_keep_prob

    Grid: (bs * n_heads, cdiv(sq, BLOCK_SQ))
    """
    pid_bh = tl.program_id(0)
    pid_sq = tl.program_id(1)

    batch_idx = pid_bh // n_heads
    head      = pid_bh % n_heads

    sq_start = pid_sq * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < sq

    skv_offs = tl.arange(0, BLOCK_SKV)

    dP_base  = dP_ptr  + batch_idx * stride_dp_bs  + head * stride_dp_h
    P_base   = P_ptr   + batch_idx * stride_p_bs   + head * stride_p_h
    Pd_base  = Pd_ptr  + batch_idx * stride_pd_bs  + head * stride_pd_h
    dS_base  = dS_ptr  + batch_idx * stride_ds_bs  + head * stride_ds_h

    num_skv_blocks = tl.cdiv(skv, BLOCK_SKV)

    # ----- Single-pass: when the entire skv dimension fits in one BLOCK_SKV tile -----
    if BLOCK_SKV >= skv:
        skv_tile_offs = skv_offs
        skv_msk       = skv_tile_offs < skv
        combined_mask = sq_mask[:, None] & skv_msk[None, :]

        dp_ptrs  = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
        dp_tile  = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # Load P_dropped and detect mask as nonzero
        pd_ptrs  = Pd_base + sq_offs[:, None] * stride_pd_sq + skv_tile_offs[None, :] * stride_pd_skv
        pd_tile  = tl.load(pd_ptrs, mask=combined_mask, other=0.0).to(tl.float32)
        # Positions where P_dropped != 0 were kept by dropout
        kept     = tl.where(pd_tile != 0.0, 1.0, 0.0)
        dp_masked = dp_tile * kept * inv_keep_prob

        p_ptrs   = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile   = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # Compute rowsum in registers — no second HBM pass needed
        rowsum = tl.sum(dp_masked * p_tile, axis=1)

        dS_tile = p_tile * (dp_masked - rowsum[:, None])

        ds_ptrs = dS_base + sq_offs[:, None] * stride_ds_sq + skv_tile_offs[None, :] * stride_ds_skv
        tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)

    else:
        # ----- Two-pass fallback: skv is larger than BLOCK_SKV -----

        # Pass 1: compute rowsum(dP_masked * P)
        rowsum = tl.zeros((BLOCK_SQ,), dtype=tl.float32)

        for skv_tile in range(num_skv_blocks):
            skv_start     = skv_tile * BLOCK_SKV
            skv_tile_offs = skv_start + skv_offs
            skv_msk       = skv_tile_offs < skv

            combined_mask = sq_mask[:, None] & skv_msk[None, :]

            dp_ptrs  = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
            dp_tile  = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

            # Load P_dropped and detect mask as nonzero
            pd_ptrs  = Pd_base + sq_offs[:, None] * stride_pd_sq + skv_tile_offs[None, :] * stride_pd_skv
            pd_tile  = tl.load(pd_ptrs, mask=combined_mask, other=0.0).to(tl.float32)
            kept     = tl.where(pd_tile != 0.0, 1.0, 0.0)
            dp_masked = dp_tile * kept * inv_keep_prob

            p_ptrs   = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
            p_tile   = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

            rowsum  += tl.sum(dp_masked * p_tile, axis=1)

        # Pass 2: compute dS and store
        for skv_tile in range(num_skv_blocks):
            skv_start     = skv_tile * BLOCK_SKV
            skv_tile_offs = skv_start + skv_offs
            skv_msk       = skv_tile_offs < skv

            combined_mask = sq_mask[:, None] & skv_msk[None, :]

            dp_ptrs   = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
            dp_tile   = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

            # Load P_dropped and detect mask as nonzero
            pd_ptrs   = Pd_base + sq_offs[:, None] * stride_pd_sq + skv_tile_offs[None, :] * stride_pd_skv
            pd_tile   = tl.load(pd_ptrs, mask=combined_mask, other=0.0).to(tl.float32)
            kept      = tl.where(pd_tile != 0.0, 1.0, 0.0)
            dp_masked = dp_tile * kept * inv_keep_prob

            p_ptrs    = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
            p_tile    = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

            dS_tile = p_tile * (dp_masked - rowsum[:, None])

            ds_ptrs = dS_base + sq_offs[:, None] * stride_ds_sq + skv_tile_offs[None, :] * stride_ds_skv
            tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


def _make_dO_gqa_contiguous(grad_attn_output, bs, n_kv_heads, n_groups, seq_q, d):
    """
    Convert grad_attn_output from [bs, sq, 80, d] to [bs*8, 10*sq, d]
    without going through the intermediate [bs, 80, sq, d] contiguous layout.
    """
    s0 = grad_attn_output.stride(0)  # sq * n_heads * d
    s1 = grad_attn_output.stride(1)  # n_heads * d
    s2 = grad_attn_output.stride(2)  # d
    s3 = grad_attn_output.stride(3)  # 1

    # View as [bs, n_kv_heads, n_groups, sq, d]
    view_5d = torch.as_strided(
        grad_attn_output,
        size=(bs, n_kv_heads, n_groups, seq_q, d),
        stride=(s0, s2 * n_groups, s2, s1, s3),
    )
    # Now make contiguous as [bs*n_kv_heads, n_groups*sq, d]
    return view_5d.reshape(bs * n_kv_heads, n_groups * seq_q, d).contiguous()


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

    # === Key optimization: avoid the permute().contiguous() on [bs, 80, sq, d] ===
    dO_gqa = _make_dO_gqa_contiguous(grad_attn_output, bs, n_kv_heads, n_groups, seq_q, d)
    # dO_gqa: [bs*8, 10*sq, d] bfloat16 contiguous

    # V: [bs, 8, skv, d] -> [bs*8, skv, d] — NO expansion needed!
    V_gqa = value_states.reshape(bs * n_kv_heads, seq_kv, d)  # [bs*8, skv, d]

    # P_dropped for dV: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
    P_dropped_gqa = attn_weights_dropped.view(bs, n_kv_heads, n_groups, seq_q, seq_kv) \
                                        .reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # Pre-allocate output tensors
    dP_gqa = torch.empty((bs * n_kv_heads, n_groups * seq_q, seq_kv), dtype=torch.bfloat16, device=device)
    dV_gqa = torch.empty((bs * n_kv_heads, seq_kv, d), dtype=torch.bfloat16, device=device)

    # Get the two streams
    stream1, stream2 = _get_streams()
    current_stream = torch.cuda.current_stream(device)

    # ---- GEMM 1 (BF16): dP = dO_gqa @ V_gqa^T on stream1 ----
    with torch.cuda.stream(stream1):
        stream1.wait_stream(current_stream)
        # [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
        torch.bmm(dO_gqa, V_gqa.transpose(-2, -1), out=dP_gqa)

    # ---- GEMM 2 (BF16): dV = P_dropped_gqa^T @ dO_gqa on stream2 ----
    with torch.cuda.stream(stream2):
        stream2.wait_stream(current_stream)
        # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
        torch.bmm(P_dropped_gqa.transpose(-2, -1), dO_gqa, out=dV_gqa)

    # Synchronize both streams back to current stream before using results
    current_stream.wait_stream(stream1)
    current_stream.wait_stream(stream2)

    # Reshape dP_gqa back to [bs, 80, sq, skv]
    dP_raw = dP_gqa.view(bs, n_kv_heads, n_groups, seq_q, seq_kv) \
                   .reshape(bs, n_heads, seq_q, seq_kv)  # [bs, 80, sq, skv] BF16

    # Reshape dV to [bs, 8, skv, d]
    dV = dV_gqa.view(bs, n_kv_heads, seq_kv, d).to(torch.bfloat16)

    # ---- Triton kernel: softmax backward ----
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=device)

    P_attn   = attn_weights.contiguous()         # [bs, 80, sq, skv] bfloat16
    P_drop   = attn_weights_dropped.contiguous() # [bs, 80, sq, skv] bfloat16 — replaces bool mask

    inv_keep = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Use single autotuned kernel for all sizes
    grid_dS = lambda meta: (bs * n_heads, triton.cdiv(seq_q, meta['BLOCK_SQ']))
    softmax_bwd_kernel[grid_dS](
        dP_raw,  dP_raw.stride(0),  dP_raw.stride(1),  dP_raw.stride(2),  dP_raw.stride(3),
        P_attn,  P_attn.stride(0),  P_attn.stride(1),  P_attn.stride(2),  P_attn.stride(3),
        P_drop,  P_drop.stride(0),  P_drop.stride(1),  P_drop.stride(2),  P_drop.stride(3),
        dS,      dS.stride(0),      dS.stride(1),      dS.stride(2),      dS.stride(3),
        bs, n_heads, seq_q, seq_kv,
        inv_keep_prob=inv_keep,
    )

    return dS, dV
