"""
Optimized attention-backward kernel using BF16 GEMMs + Triton fused softmax backward.

Strategy:
  1. Fused softmax backward: compute dP = dO @ V^T on-the-fly inside the kernel,
     avoiding materializing dP to HBM. One program per (batch, query_head, sq_row).
     KV head = query_head // 10 for GQA mapping.

  2. For dV: reshape P_dropped from [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
             reshape dO from [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
             compute [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
             This directly gives dV summed over groups -- no separate reduction!

  3. Stream parallelism: launch dV GEMM on stream2 overlapped with Triton kernel.

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
        # Two-pass configs for large skv
        triton.Config({'BLOCK_SKV': 64},   num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SKV': 128},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SKV': 128},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SKV': 128},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SKV': 256},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SKV': 256},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SKV': 256},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SKV': 512},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SKV': 512},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SKV': 512},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SKV': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SKV': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SKV': 1024}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SKV': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SKV': 4096}, num_warps=16, num_stages=2),
    ],
    key=['sq', 'skv'],
)
@triton.jit
def fused_softmax_bwd_kernel(
    # dO: [bs, 80, sq, d] bfloat16
    dO_ptr, stride_do_bs, stride_do_h, stride_do_sq, stride_do_d,
    # V (value_states): [bs, 8, skv, d] bfloat16
    V_ptr, stride_v_bs, stride_v_kvh, stride_v_skv, stride_v_d,
    # P (attn_weights): [bs, 80, sq, skv] bfloat16
    P_ptr, stride_p_bs, stride_p_h, stride_p_sq, stride_p_skv,
    # dropout_mask: [bs, 80, sq, skv] bool
    mask_ptr, stride_m_bs, stride_m_h, stride_m_sq, stride_m_skv,
    # dS output: [bs, 80, sq, skv] bfloat16
    dS_ptr, stride_ds_bs, stride_ds_h, stride_ds_sq, stride_ds_skv,
    bs, n_heads, n_kv_heads, n_groups, sq, skv,
    inv_keep_prob: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused softmax backward: computes dP = dO @ V^T on-the-fly, applies dropout
    masking, and computes dS = P * (dP_masked - rowsum(dP_masked * P)).

    When BLOCK_SKV >= skv (entire KV row fits in registers), uses a single pass:
    compute rowsum and dS in one sweep, eliminating a second HBM round-trip.

    One program per (batch, query_head, sq_row) for simplest indexing.
    Grid: (bs * n_heads * sq,)
    """
    pid = tl.program_id(0)

    # Decode (batch, head, sq_row) from flat index
    sq_row    = pid % sq
    tmp       = pid // sq
    head_idx  = tmp % n_heads
    batch_idx = tmp // n_heads

    # KV head mapping for GQA
    kv_head_idx = head_idx // n_groups  # = head_idx // 10

    # Pointers to the single row of dO: [d]
    dO_row_base = dO_ptr + batch_idx * stride_do_bs + head_idx * stride_do_h + sq_row * stride_do_sq
    # Pointers to V for this KV head: [skv, d]
    V_base = V_ptr + batch_idx * stride_v_bs + kv_head_idx * stride_v_kvh
    # Pointers to P row: [skv]
    P_row_base = P_ptr + batch_idx * stride_p_bs + head_idx * stride_p_h + sq_row * stride_p_sq
    # Pointers to mask row: [skv]
    M_row_base = mask_ptr + batch_idx * stride_m_bs + head_idx * stride_m_h + sq_row * stride_m_sq
    # Pointers to dS row: [skv]
    dS_row_base = dS_ptr + batch_idx * stride_ds_bs + head_idx * stride_ds_h + sq_row * stride_ds_sq

    d_offs = tl.arange(0, BLOCK_D)  # [BLOCK_D]

    # Load dO row: [BLOCK_D] (head_dim = 128 = BLOCK_D, always fits)
    dO_row = tl.load(dO_row_base + d_offs * stride_do_d).to(tl.float32)  # [BLOCK_D]

    skv_offs = tl.arange(0, BLOCK_SKV)

    if BLOCK_SKV >= skv:
        # ---- SINGLE-PASS: entire KV sequence fits in one tile ----
        # Load everything at once, compute rowsum and dS in a single sweep.
        skv_valid = skv_offs < skv

        # Load V tile: [BLOCK_SKV, BLOCK_D]
        v_ptrs = V_base + skv_offs[:, None] * stride_v_skv + d_offs[None, :] * stride_v_d
        v_tile = tl.load(v_ptrs, mask=skv_valid[:, None], other=0.0).to(tl.float32)

        # dP_tile[skv] = dot(dO_row, v_tile[skv, :])
        dp_tile = tl.sum(dO_row[None, :] * v_tile, axis=1)  # [BLOCK_SKV]

        # Load dropout mask
        m_ptrs = M_row_base + skv_offs * stride_m_skv
        m_tile = tl.load(m_ptrs, mask=skv_valid, other=0).to(tl.float32)
        dp_masked = dp_tile * m_tile * inv_keep_prob

        # Load P tile
        p_ptrs = P_row_base + skv_offs * stride_p_skv
        p_tile = tl.load(p_ptrs, mask=skv_valid, other=0.0).to(tl.float32)

        # Compute rowsum = sum(dP_masked * P) — masked-out entries contribute 0
        rowsum = tl.sum(dp_masked * p_tile, axis=0)

        # dS = P * (dP_masked - rowsum)
        dS_tile = p_tile * (dp_masked - rowsum)

        # Store dS
        ds_ptrs = dS_row_base + skv_offs * stride_ds_skv
        tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=skv_valid)

    else:
        # ---- TWO-PASS: skv is large, need tiling ----
        num_skv_blocks = tl.cdiv(skv, BLOCK_SKV)

        # Pass 1: compute dP_masked for each skv block, accumulate rowsum
        rowsum = 0.0

        for skv_tile in range(num_skv_blocks):
            skv_start     = skv_tile * BLOCK_SKV
            skv_tile_offs = skv_start + skv_offs
            skv_valid     = skv_tile_offs < skv

            # Load V tile: [BLOCK_SKV, BLOCK_D]
            v_ptrs = V_base + skv_tile_offs[:, None] * stride_v_skv + d_offs[None, :] * stride_v_d
            v_tile = tl.load(v_ptrs, mask=skv_valid[:, None], other=0.0).to(tl.float32)

            # dP_tile[skv] = dot(dO_row, v_tile[skv, :]) = sum over d
            dp_tile = tl.sum(dO_row[None, :] * v_tile, axis=1)  # [BLOCK_SKV]

            # Load dropout mask
            m_ptrs = M_row_base + skv_tile_offs * stride_m_skv
            m_tile = tl.load(m_ptrs, mask=skv_valid, other=0).to(tl.float32)
            dp_masked = dp_tile * m_tile * inv_keep_prob

            # Load P tile
            p_ptrs = P_row_base + skv_tile_offs * stride_p_skv
            p_tile = tl.load(p_ptrs, mask=skv_valid, other=0.0).to(tl.float32)

            # Accumulate rowsum = sum(dP_masked * P)
            rowsum += tl.sum(dp_masked * p_tile, axis=0)

        # Pass 2: compute dS = P * (dP_masked - rowsum) and store
        for skv_tile in range(num_skv_blocks):
            skv_start     = skv_tile * BLOCK_SKV
            skv_tile_offs = skv_start + skv_offs
            skv_valid     = skv_tile_offs < skv

            # Load V tile again
            v_ptrs = V_base + skv_tile_offs[:, None] * stride_v_skv + d_offs[None, :] * stride_v_d
            v_tile = tl.load(v_ptrs, mask=skv_valid[:, None], other=0.0).to(tl.float32)

            # Recompute dP
            dp_tile = tl.sum(dO_row[None, :] * v_tile, axis=1)

            # Load dropout mask
            m_ptrs = M_row_base + skv_tile_offs * stride_m_skv
            m_tile = tl.load(m_ptrs, mask=skv_valid, other=0).to(tl.float32)
            dp_masked = dp_tile * m_tile * inv_keep_prob

            # Load P tile
            p_ptrs = P_row_base + skv_tile_offs * stride_p_skv
            p_tile = tl.load(p_ptrs, mask=skv_valid, other=0.0).to(tl.float32)

            # dS = P * (dP_masked - rowsum)
            dS_tile = p_tile * (dp_masked - rowsum)

            # Store dS
            ds_ptrs = dS_row_base + skv_tile_offs * stride_ds_skv
            tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=skv_valid)


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

    # V: [bs, 8, skv, d] -> [bs*8, skv, d] for dV GEMM
    V_gqa = value_states.reshape(bs * n_kv_heads, seq_kv, d)  # [bs*8, skv, d]

    # P_dropped for dV: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
    P_dropped_gqa = attn_weights_dropped.view(bs, n_kv_heads, n_groups, seq_q, seq_kv) \
                                        .reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # Pre-allocate output tensors
    dV_gqa = torch.empty((bs * n_kv_heads, seq_kv, d), dtype=torch.bfloat16, device=device)
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=device)

    # Get the two streams
    stream1, stream2 = _get_streams()
    current_stream = torch.cuda.current_stream(device)

    # ---- GEMM (BF16): dV = P_dropped_gqa^T @ dO_gqa on stream2 ----
    with torch.cuda.stream(stream2):
        stream2.wait_stream(current_stream)
        # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
        torch.bmm(P_dropped_gqa.transpose(-2, -1), dO_gqa, out=dV_gqa)

    # ---- Fused Triton kernel: softmax backward with on-the-fly dP computation ----
    # These should already be contiguous, but ensure correctness
    P_attn = attn_weights if attn_weights.is_contiguous() else attn_weights.contiguous()
    V_cont = value_states if value_states.is_contiguous() else value_states.contiguous()
    dmask  = dropout_mask if dropout_mask.is_contiguous() else dropout_mask.contiguous()
    dO_cont = dO  # already contiguous from permute+contiguous above

    inv_keep = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Grid: one program per (batch, query_head, sq_row)
    total_rows = bs * n_heads * seq_q
    grid = (total_rows,)

    fused_softmax_bwd_kernel[grid](
        dO_cont, dO_cont.stride(0), dO_cont.stride(1), dO_cont.stride(2), dO_cont.stride(3),
        V_cont,  V_cont.stride(0),  V_cont.stride(1),  V_cont.stride(2),  V_cont.stride(3),
        P_attn,  P_attn.stride(0),  P_attn.stride(1),  P_attn.stride(2),  P_attn.stride(3),
        dmask,   dmask.stride(0),   dmask.stride(1),   dmask.stride(2),   dmask.stride(3),
        dS,      dS.stride(0),      dS.stride(1),      dS.stride(2),      dS.stride(3),
        bs, n_heads, n_kv_heads, n_groups, seq_q, seq_kv,
        inv_keep_prob=inv_keep,
        BLOCK_D=HEAD_DIM,
    )

    # Synchronize dV stream back to current stream
    current_stream.wait_stream(stream2)

    # Reshape dV to [bs, 8, skv, d]
    dV = dV_gqa.view(bs, n_kv_heads, seq_kv, d)

    return dS, dV
