"""
Optimized attention-backward kernel using BF16 GEMMs + Triton softmax backward.

Strategy:
  1. Avoid materializing V_exp [bs, 80, skv, d] entirely.
     For dP: avoid permute+contiguous of dO by using matmul with broadcasting
             directly from [bs, sq, 8, 10, d] layout.
             dO.view(bs, sq, n_kv, ng, d) @ V.view(bs, 1, n_kv, d, skv)
             -> [bs, sq, n_kv, ng, skv] -> permute -> [bs, n_kv, ng, sq, skv]
             -> reshape [bs, 80, sq, skv]

  2. For dV: similarly compute from original layout.
             attn_weights_dropped.view(bs, n_kv, ng, sq, skv) contracted with
             dO.view(bs, sq, n_kv, ng, d) -> direct dV sum over groups.

  3. Stream parallelism: launch GEMM1 (dP) and GEMM2 (dV) on separate CUDA streams
     so they overlap in execution. Both are independent computations.

  4. Triton softmax-backward kernel: uses attn_weights_dropped directly instead of
     dropout_mask (bool), eliminating the mask load and simplifying arithmetic.
     rowsum = sum(dP_raw * P_dropped)  [no mask/inv_keep needed]
     dS = P_dropped * dP_raw - P * rowsum  [equivalent formulation]

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
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 512},  num_warps=8),
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 1024}, num_warps=8),
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 1024}, num_warps=16),
        triton.Config({'BLOCK_SQ': 2,  'BLOCK_SKV': 512},  num_warps=8),
        triton.Config({'BLOCK_SQ': 2,  'BLOCK_SKV': 1024}, num_warps=8),
        triton.Config({'BLOCK_SQ': 2,  'BLOCK_SKV': 1024}, num_warps=16),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 512},  num_warps=8),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 1024}, num_warps=8),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 512},  num_warps=16),
        # Two-pass fallback configs (smaller tiles, also with higher warps)
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 256},  num_warps=8),
        triton.Config({'BLOCK_SQ': 2,  'BLOCK_SKV': 256},  num_warps=8),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 256},  num_warps=8),
        triton.Config({'BLOCK_SQ': 8,  'BLOCK_SKV': 128},  num_warps=8),
        triton.Config({'BLOCK_SQ': 8,  'BLOCK_SKV': 256},  num_warps=8),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 128},  num_warps=8),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 64},   num_warps=8),
    ],
    key=['sq', 'skv'],
)
@triton.jit
def softmax_bwd_kernel(
    # dP: [bs, 80, sq, skv] bfloat16  (raw, before dropout masking)
    dP_ptr, stride_dp_bs, stride_dp_h, stride_dp_sq, stride_dp_skv,
    # P (attn_weights): [bs, 80, sq, skv] bfloat16
    P_ptr, stride_p_bs, stride_p_h, stride_p_sq, stride_p_skv,
    # Pd (attn_weights_dropped): [bs, 80, sq, skv] bfloat16
    Pd_ptr, stride_pd_bs, stride_pd_h, stride_pd_sq, stride_pd_skv,
    # dS output: [bs, 80, sq, skv] bfloat16
    dS_ptr, stride_ds_bs, stride_ds_h, stride_ds_sq, stride_ds_skv,
    bs, n_heads, sq, skv,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """
    Softmax backward using attn_weights_dropped directly instead of bool mask.

    Key identity: P_dropped = P * mask / (1-p)
    Therefore:
      rowsum = sum(dP_masked * P)
             = sum(dP_raw * mask * inv_keep * P)
             = sum(dP_raw * P_dropped)   <-- use this!
      dP_masked = dP_raw * mask * inv_keep
               = dP_raw * P_dropped / P  (where P != 0)
      dS = P * (dP_masked - rowsum)
         = P * (dP_raw * P_dropped / P - rowsum)
         = P_dropped * dP_raw - P * rowsum

    This eliminates the bool mask load entirely and simplifies arithmetic.

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

    dP_base = dP_ptr + batch_idx * stride_dp_bs + head * stride_dp_h
    P_base  = P_ptr  + batch_idx * stride_p_bs  + head * stride_p_h
    Pd_base = Pd_ptr + batch_idx * stride_pd_bs + head * stride_pd_h
    dS_base = dS_ptr + batch_idx * stride_ds_bs + head * stride_ds_h

    num_skv_blocks = tl.cdiv(skv, BLOCK_SKV)

    # ----- Single-pass: when the entire skv dimension fits in one BLOCK_SKV tile -----
    if BLOCK_SKV >= skv:
        skv_tile_offs = skv_offs
        skv_msk       = skv_tile_offs < skv
        combined_mask = sq_mask[:, None] & skv_msk[None, :]

        # Load dP_raw
        dp_ptrs  = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
        dp_tile  = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # Load P_dropped
        pd_ptrs  = Pd_base + sq_offs[:, None] * stride_pd_sq + skv_tile_offs[None, :] * stride_pd_skv
        pd_tile  = tl.load(pd_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # Pass 1: rowsum = sum(dP_raw * P_dropped)  -- no P load needed for rowsum!
        rowsum = tl.sum(dp_tile * pd_tile, axis=1)

        # Load P (attn_weights) -- needed for dS computation
        p_ptrs   = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile   = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # dS = P_dropped * dP_raw - P * rowsum
        dS_tile = pd_tile * dp_tile - p_tile * rowsum[:, None]

        ds_ptrs = dS_base + sq_offs[:, None] * stride_ds_sq + skv_tile_offs[None, :] * stride_ds_skv
        tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)

    else:
        # ----- Two-pass fallback: skv is larger than BLOCK_SKV -----

        # Pass 1: compute rowsum = sum(dP_raw * P_dropped)
        rowsum = tl.zeros((BLOCK_SQ,), dtype=tl.float32)

        for skv_tile in range(num_skv_blocks):
            skv_start     = skv_tile * BLOCK_SKV
            skv_tile_offs = skv_start + skv_offs
            skv_msk       = skv_tile_offs < skv
            combined_mask = sq_mask[:, None] & skv_msk[None, :]

            dp_ptrs  = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
            dp_tile  = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

            pd_ptrs  = Pd_base + sq_offs[:, None] * stride_pd_sq + skv_tile_offs[None, :] * stride_pd_skv
            pd_tile  = tl.load(pd_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

            rowsum  += tl.sum(dp_tile * pd_tile, axis=1)

        # Pass 2: compute dS = P_dropped * dP_raw - P * rowsum and store
        for skv_tile in range(num_skv_blocks):
            skv_start     = skv_tile * BLOCK_SKV
            skv_tile_offs = skv_start + skv_offs
            skv_msk       = skv_tile_offs < skv
            combined_mask = sq_mask[:, None] & skv_msk[None, :]

            dp_ptrs   = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
            dp_tile   = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

            pd_ptrs   = Pd_base + sq_offs[:, None] * stride_pd_sq + skv_tile_offs[None, :] * stride_pd_skv
            pd_tile   = tl.load(pd_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

            p_ptrs    = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
            p_tile    = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

            # dS = P_dropped * dP_raw - P * rowsum
            dS_tile = pd_tile * dp_tile - p_tile * rowsum[:, None]

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

    # grad_attn_output: [bs, seq_q, 80, d]
    # Avoid permute+contiguous copy to [bs, 80, seq_q, d].
    #
    # Instead, use the native layout directly with matmul broadcasting:
    #   dO_5d: [bs, seq_q, n_kv_heads, n_groups, d]  (zero-copy view)
    #   V_5d:  [bs, 1, n_kv_heads, d, seq_kv]        (zero-copy view+transpose)
    #
    # matmul [bs, seq_q, n_kv_heads, n_groups, d] @ [bs, 1, n_kv_heads, d, seq_kv]
    #   -> [bs, seq_q, n_kv_heads, n_groups, seq_kv]
    # Then permute (0,2,3,1,4) -> [bs, n_kv_heads, n_groups, seq_q, seq_kv]
    # reshape -> [bs, n_heads, seq_q, seq_kv]  -- this is dP_raw
    #
    # For dV:
    #   P_dropped_5d: [bs, n_kv_heads, n_groups, seq_q, seq_kv]  (zero-copy view)
    #   dO_5d:        [bs, seq_q, n_kv_heads, n_groups, d]        (zero-copy view)
    #   For each kv head, dV[b,k,s,d] = sum_g,q P_dropped[b,k,g,q,s] * dO[b,q,k,g,d]
    #   = P_dropped[b,k,:,:,s]^T @ dO[b,:,k,:,d]  summed over groups
    #   This is a [n_kv_heads] batch of [seq_kv, n_groups*seq_q] @ [n_groups*seq_q, d]
    #   => reshape P_dropped to [bs*n_kv_heads, n_groups*seq_q, seq_kv]
    #   and dO to [bs*n_kv_heads, n_groups*seq_q, d]  (via contiguous permute of 5d)
    #
    # The key saving: for GEMM1 (the larger one computing dP), we avoid the large
    # permute+contiguous of [bs,80,seq_q,d]. For GEMM2 (dV), we still need a
    # contiguous form but can permute the 5d shape to [bs,n_kv_heads,n_groups,seq_q,d].

    # Zero-copy views of grad_attn_output
    # [bs, seq_q, 80, d] -> [bs, seq_q, 8, 10, d]
    dO_5d = grad_attn_output.view(bs, seq_q, n_kv_heads, n_groups, d)

    # V: [bs, 8, seq_kv, d]
    # We need V^T: [bs, 8, d, seq_kv] for the matmul
    # Reshape to [bs, 1, n_kv_heads, d, seq_kv] for broadcasting over seq_q dimension
    V_T = value_states.transpose(-2, -1)  # [bs, 8, d, seq_kv] -- non-contiguous but ok
    V_T_5d = V_T.unsqueeze(1)  # [bs, 1, 8, d, seq_kv]

    # P_dropped: [bs, 80, seq_q, seq_kv] -> [bs, 8, 10, seq_q, seq_kv] (zero-copy view)
    P_dropped_5d = attn_weights_dropped.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)

    # Permute dO_5d for GEMM2 (dV): need [bs*8, 10*seq_q, d]
    # dO_5d: [bs, seq_q, 8, 10, d] -> permute to [bs, 8, 10, seq_q, d] -> [bs*8, 10*seq_q, d]
    # This permute is unavoidable for dV GEMM but only involves seq_q*d*n_kv_heads elements
    # per batch (vs n_heads*seq_q*d for the original approach)
    dO_for_dV = dO_5d.permute(0, 2, 3, 1, 4).reshape(bs * n_kv_heads, n_groups * seq_q, d).contiguous()

    # P_dropped for dV GEMM: [bs, 8, 10, seq_q, seq_kv] -> [bs*8, 10*seq_q, seq_kv]
    P_dropped_gqa = P_dropped_5d.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # Pre-allocate output tensors
    # dP result from GEMM1: [bs, seq_q, n_kv_heads, n_groups, seq_kv]
    dP_5d = torch.empty((bs, seq_q, n_kv_heads, n_groups, seq_kv), dtype=torch.bfloat16, device=device)
    dV_gqa = torch.empty((bs * n_kv_heads, seq_kv, d), dtype=torch.bfloat16, device=device)

    # Get the two streams
    stream1, stream2 = _get_streams()
    current_stream = torch.cuda.current_stream(device)

    # ---- GEMM 1 (BF16): dP = dO_5d @ V_T_5d on stream1 ----
    # [bs, seq_q, 8, 10, d] @ [bs, 1, 8, d, seq_kv] -> [bs, seq_q, 8, 10, seq_kv]
    # No contiguous copy needed for dO -- uses native layout!
    with torch.cuda.stream(stream1):
        stream1.wait_stream(current_stream)
        torch.matmul(dO_5d, V_T_5d, out=dP_5d)

    # ---- GEMM 2 (BF16): dV = P_dropped_gqa^T @ dO_for_dV on stream2 ----
    with torch.cuda.stream(stream2):
        stream2.wait_stream(current_stream)
        # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
        torch.bmm(P_dropped_gqa.transpose(-2, -1), dO_for_dV, out=dV_gqa)

    # Synchronize both streams back to current stream before using results
    current_stream.wait_stream(stream1)
    current_stream.wait_stream(stream2)

    # Reshape dP_5d from [bs, seq_q, n_kv_heads, n_groups, seq_kv]
    # -> permute to [bs, n_kv_heads, n_groups, seq_q, seq_kv]
    # -> reshape to [bs, n_heads, seq_q, seq_kv]
    # The permute here is over a smaller tensor (seq_kv is the last dim, no d)
    dP_raw = dP_5d.permute(0, 2, 3, 1, 4).reshape(bs, n_heads, seq_q, seq_kv).contiguous()

    # Reshape dV to [bs, 8, skv, d]
    dV = dV_gqa.view(bs, n_kv_heads, seq_kv, d).to(torch.bfloat16)

    # ---- Triton kernel: softmax backward ----
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=device)

    # Conditional contiguous to avoid unnecessary copies when inputs are already contiguous
    P_attn  = attn_weights if attn_weights.is_contiguous() else attn_weights.contiguous()
    Pd_attn = attn_weights_dropped if attn_weights_dropped.is_contiguous() else attn_weights_dropped.contiguous()

    # No inv_keep_prob needed — the math uses P_dropped directly!

    grid_dS = lambda meta: (bs * n_heads, triton.cdiv(seq_q, meta['BLOCK_SQ']))
    softmax_bwd_kernel[grid_dS](
        dP_raw, dP_raw.stride(0), dP_raw.stride(1), dP_raw.stride(2), dP_raw.stride(3),
        P_attn, P_attn.stride(0), P_attn.stride(1), P_attn.stride(2), P_attn.stride(3),
        Pd_attn, Pd_attn.stride(0), Pd_attn.stride(1), Pd_attn.stride(2), Pd_attn.stride(3),
        dS,     dS.stride(0),     dS.stride(1),     dS.stride(2),     dS.stride(3),
        bs, n_heads, seq_q, seq_kv,
    )

    return dS, dV
