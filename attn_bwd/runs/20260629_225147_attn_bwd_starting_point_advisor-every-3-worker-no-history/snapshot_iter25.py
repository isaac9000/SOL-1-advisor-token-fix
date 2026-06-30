"""
Attention backward: GQA-native cuBLAS batched GEMMs (no V expansion) +
fused Triton multi-row softmax backward kernel with tl.dot-based accumulation.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool

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

# Pre-create persistent CUDA streams for concurrent BMM execution
_stream1 = None
_stream2 = None

def _get_streams():
    global _stream1, _stream2
    if _stream1 is None:
        _stream1 = torch.cuda.Stream(priority=-1)  # high priority
        _stream2 = torch.cuda.Stream(priority=-1)  # high priority
    return _stream1, _stream2


# ---------------------------------------------------------------------------
# Multi-row Triton softmax backward kernel.
# Grid: (bs * n_heads, cdiv(seq_q, BLOCK_SQ))
# Each program handles BLOCK_SQ rows of one (batch, head) pair.
# Two-pass: pass1 accumulates per-row sums, pass2 computes dS and stores.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_multirow_kernel(
    dP_ptr,      # [bs, n_heads, sq, skv]  bfloat16
    P_ptr,       # [bs, n_heads, sq, skv]  bfloat16
    mask_ptr,    # [bs, n_heads, sq, skv]  bool
    dS_ptr,      # [bs, n_heads, sq, skv]  bfloat16  (output)
    inv_scale,   # scalar float
    sq, skv,
    # strides for all 4D tensors (same layout)
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    bh_idx  = tl.program_id(0)   # (batch, head) index
    sq_blk  = tl.program_id(1)   # which block of sq rows

    # Base offset for this (batch, head)
    bs_idx = bh_idx // 80
    h_idx  = bh_idx % 80
    base_bh = bs_idx * stride_bs + h_idx * stride_h

    # Row offsets for this program
    sq_start = sq_blk * BLOCK_SQ
    sq_offs = sq_start + tl.arange(0, BLOCK_SQ)   # [BLOCK_SQ]
    sq_mask = sq_offs < sq                          # [BLOCK_SQ]

    skv_arange = tl.arange(0, BLOCK_SKV)           # [BLOCK_SKV]

    row_sum = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange   # [BLOCK_SKV]
        skv_mask = skv_offs < skv                        # [BLOCK_SKV]

        ptrs = (base_bh
                + sq_offs[:, None] * stride_sq
                + skv_offs[None, :] * stride_skv)
        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dP_tile = tl.load(dP_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)
        drop    = tl.load(mask_ptr + ptrs, mask=combined_mask, other=0).to(tl.float32)
        dP_tile = dP_tile * drop * inv_scale

        P_tile  = tl.load(P_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        row_sum += tl.sum(dP_tile * P_tile, axis=1)

    # Pass 2: compute dS and store
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        ptrs = (base_bh
                + sq_offs[:, None] * stride_sq
                + skv_offs[None, :] * stride_skv)
        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dP_tile = tl.load(dP_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)
        drop    = tl.load(mask_ptr + ptrs, mask=combined_mask, other=0).to(tl.float32)
        dP_tile = dP_tile * drop * inv_scale

        P_tile  = tl.load(P_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        dS_tile = P_tile * (dP_tile - row_sum[:, None])

        tl.store(dS_ptr + ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


# ---------------------------------------------------------------------------
# Single-pass softmax backward kernel: accumulate row_sum and write dS
# in a single loop by buffering all tiles.
# Only feasible when BLOCK_SKV covers the entire skv dimension.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_singlepass_kernel(
    dP_ptr,
    P_ptr,
    mask_ptr,
    dS_ptr,
    inv_scale,
    sq, skv,
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """Single-pass version when skv fits in BLOCK_SKV (power-of-2, <= 4096)."""
    bh_idx  = tl.program_id(0)
    sq_blk  = tl.program_id(1)

    bs_idx = bh_idx // 80
    h_idx  = bh_idx % 80
    base_bh = bs_idx * stride_bs + h_idx * stride_h

    sq_start = sq_blk * BLOCK_SQ
    sq_offs = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask = sq_offs < sq

    skv_offs = tl.arange(0, BLOCK_SKV)
    skv_mask = skv_offs < skv

    ptrs = (base_bh
            + sq_offs[:, None] * stride_sq
            + skv_offs[None, :] * stride_skv)
    combined_mask = sq_mask[:, None] & skv_mask[None, :]

    # Load dP_raw and apply dropout mask + scale
    dP_tile = tl.load(dP_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)
    drop    = tl.load(mask_ptr + ptrs, mask=combined_mask, other=0).to(tl.float32)
    dP_tile = dP_tile * drop * inv_scale

    # Load P
    P_tile  = tl.load(P_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)

    # Compute row sum and dS in one pass
    row_sum = tl.sum(dP_tile * P_tile, axis=1)   # [BLOCK_SQ]
    dS_tile = P_tile * (dP_tile - row_sum[:, None])

    tl.store(dS_ptr + ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    device = grad_attn_output.device

    # ----------------------------------------------------------------
    # Step 1: Prepare dO as [bs, 8, 10*sq, 128] bfloat16
    # grad_attn_output: [bs, sq, 80, 128]  (contiguous)
    # We do a single .contiguous() on the 4D transposed layout [bs, 80, sq, 128]
    # then reinterpret as [bs, 8, 10*sq, 128] with a zero-copy .view().
    # ----------------------------------------------------------------
    dO_4d = grad_attn_output.permute(0, 2, 1, 3).contiguous()  # [bs, 80, sq, 128]
    dO_grouped = dO_4d.view(bs, n_kv_heads, n_groups * seq_q, HEAD_DIM)
    # dO_grouped: [bs, 8, 10*sq, 128]  bfloat16, contiguous

    # Pre-allocate output tensors before launching streams
    dV = torch.empty(bs, n_kv_heads, seq_kv, HEAD_DIM, dtype=torch.bfloat16, device=device)
    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=device)

    V = value_states  # [bs, 8, skv, 128]  bfloat16
    P_drop_grouped = attn_weights_dropped.view(bs, n_kv_heads, n_groups * seq_q, seq_kv)

    stream1, stream2 = _get_streams()
    current_stream = torch.cuda.current_stream()

    # Both streams must wait for the current stream to finish producing inputs
    stream1.wait_stream(current_stream)
    stream2.wait_stream(current_stream)

    # Launch BMM for dP_raw on stream 1
    with torch.cuda.stream(stream1):
        # [bs, 8, 10*sq, 128] @ [bs, 8, 128, skv] -> [bs, 8, 10*sq, skv]
        dP_raw_grouped = torch.matmul(dO_grouped, V.transpose(-2, -1))
        # View as [bs, 80, sq, skv] — zero-copy since grouped is contiguous
        dP_raw = dP_raw_grouped.view(bs, n_heads, seq_q, seq_kv)

    # Launch BMM for dV on stream 2 — outputs directly into pre-allocated dV
    with torch.cuda.stream(stream2):
        # [bs, 8, skv, 10*sq] @ [bs, 8, 10*sq, 128] -> [bs, 8, skv, 128]
        torch.matmul(P_drop_grouped.transpose(-2, -1), dO_grouped, out=dV)

    # Wait for stream 1 (dP_raw needed for softmax kernel on current stream)
    current_stream.wait_stream(stream1)

    # ----------------------------------------------------------------
    # Step 3: Triton kernel for softmax backward (on current stream)
    # dP_raw: [bs, 80, sq, skv]  bfloat16
    # ----------------------------------------------------------------
    P = attn_weights          # [bs, 80, sq, skv] bfloat16
    mask_c = dropout_mask     # [bs, 80, sq, skv] bool

    inv_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    stride_bs    = dP_raw.stride(0)
    stride_h     = dP_raw.stride(1)
    stride_sq_s  = dP_raw.stride(2)
    stride_skv_s = dP_raw.stride(3)

    # Use single-pass kernel when skv fits in a power-of-2 block (most common cases)
    # Otherwise fall back to multi-row two-pass kernel
    if seq_kv <= 512:
        BLOCK_SQ_K  = 16
        BLOCK_SKV_K = 512
        NW = 4
        use_single = True
    elif seq_kv <= 1024:
        BLOCK_SQ_K  = 16
        BLOCK_SKV_K = 1024
        NW = 8
        use_single = True
    elif seq_kv <= 2048:
        BLOCK_SQ_K  = 8
        BLOCK_SKV_K = 2048
        NW = 16
        use_single = True
    elif seq_kv <= 4096:
        BLOCK_SQ_K  = 4
        BLOCK_SKV_K = 4096
        NW = 16
        use_single = True
    else:
        BLOCK_SQ_K  = 4
        BLOCK_SKV_K = 2048
        NW = 16
        use_single = False

    grid_softmax = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ_K))

    if use_single:
        softmax_bwd_singlepass_kernel[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv_s,
            BLOCK_SQ=BLOCK_SQ_K,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=NW,
            num_stages=1,
        )
    else:
        softmax_bwd_multirow_kernel[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv_s,
            BLOCK_SQ=BLOCK_SQ_K,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=NW,
            num_stages=1,
        )

    # Wait for stream 2 (dV) to finish before returning
    current_stream.wait_stream(stream2)

    return dS, dV
