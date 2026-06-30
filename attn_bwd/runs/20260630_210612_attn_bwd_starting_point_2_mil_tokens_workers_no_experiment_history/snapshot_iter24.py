"""
Optimized attention-backward kernel using fused dP+softmax-bwd Triton kernel + cuBLAS dV.

Strategy:
  NEW: Fused kernel for dP computation + softmax backward:
    - One Triton kernel loads dO tiles and V tiles, computes dP on-chip,
      applies dropout mask, accumulates rowsum, then computes dS.
    - Eliminates the intermediate [bs, 80, sq, skv] dP tensor from HBM entirely.
    - Two-pass: pass 1 accumulates rowsum, pass 2 writes dS.
    - GQA-aware: head // n_groups maps each of 80 heads to one of 8 KV heads.

  dV computation remains as cuBLAS BMM (already efficient).

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

# Pre-create CUDA streams for overlapping computations
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
        # BLOCK_SQ x BLOCK_SKV x BLOCK_HD (inner dim for tl.dot)
        # HEAD_DIM=128 fixed, BLOCK_HD must divide HEAD_DIM
        triton.Config({'BLOCK_SQ': 16,  'BLOCK_SKV': 64,  'BLOCK_HD': 64},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 16,  'BLOCK_SKV': 128, 'BLOCK_HD': 64},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 16,  'BLOCK_SKV': 64,  'BLOCK_HD': 128}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 16,  'BLOCK_SKV': 128, 'BLOCK_HD': 128}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 32,  'BLOCK_SKV': 64,  'BLOCK_HD': 64},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 32,  'BLOCK_SKV': 64,  'BLOCK_HD': 128}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 32,  'BLOCK_SKV': 128, 'BLOCK_HD': 64},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 32,  'BLOCK_SKV': 128, 'BLOCK_HD': 128}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SQ': 64,  'BLOCK_SKV': 64,  'BLOCK_HD': 64},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 64,  'BLOCK_SKV': 64,  'BLOCK_HD': 128}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 64,  'BLOCK_SKV': 128, 'BLOCK_HD': 64},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 64,  'BLOCK_SKV': 128, 'BLOCK_HD': 128}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 16,  'BLOCK_SKV': 64,  'BLOCK_HD': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SQ': 16,  'BLOCK_SKV': 128, 'BLOCK_HD': 128}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SQ': 32,  'BLOCK_SKV': 64,  'BLOCK_HD': 128}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SQ': 32,  'BLOCK_SKV': 128, 'BLOCK_HD': 128}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SQ': 64,  'BLOCK_SKV': 64,  'BLOCK_HD': 128}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SQ': 64,  'BLOCK_SKV': 128, 'BLOCK_HD': 128}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SQ': 16,  'BLOCK_SKV': 64,  'BLOCK_HD': 64},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 16,  'BLOCK_SKV': 128, 'BLOCK_HD': 128}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 32,  'BLOCK_SKV': 64,  'BLOCK_HD': 64},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SQ': 32,  'BLOCK_SKV': 128, 'BLOCK_HD': 128}, num_warps=8,  num_stages=2),
    ],
    key=['sq', 'skv'],
)
@triton.jit
def fused_dp_softmax_bwd_kernel(
    # dO: [bs, 80, sq, HEAD_DIM] bfloat16 — laid out as contiguous after transpose
    dO_ptr, stride_do_bs, stride_do_h, stride_do_sq, stride_do_d,
    # V:  [bs,  8, skv, HEAD_DIM] bfloat16
    V_ptr,  stride_v_bs,  stride_v_kvh, stride_v_skv, stride_v_d,
    # P (attn_weights): [bs, 80, sq, skv] bfloat16
    P_ptr,  stride_p_bs,  stride_p_h,  stride_p_sq,  stride_p_skv,
    # P_dropped (attn_weights_dropped): [bs, 80, sq, skv] bfloat16
    Pd_ptr, stride_pd_bs, stride_pd_h, stride_pd_sq, stride_pd_skv,
    # dS output: [bs, 80, sq, skv] bfloat16
    dS_ptr, stride_ds_bs, stride_ds_h, stride_ds_sq, stride_ds_skv,
    bs, n_heads, n_groups, sq, skv,
    inv_keep_prob: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    """
    Fused kernel: computes dP = dO @ V^T on-chip and immediately applies
    softmax backward, never writing the intermediate dP to HBM.

    Grid: (bs * n_heads, cdiv(sq, BLOCK_SQ))

    For each (batch, head) and sq-block:
      Pass 1: iterate over kv-blocks, compute dP tile via tl.dot,
              apply dropout mask, accumulate rowsum = sum(dP_masked * P, axis=1)
      Pass 2: iterate over kv-blocks again, recompute dP tile,
              compute dS = P * (dP_masked - rowsum) and store.

    GQA: kv_head = head // n_groups (maps 80 Q-heads -> 8 KV-heads)
    """
    pid_bh = tl.program_id(0)
    pid_sq = tl.program_id(1)

    batch_idx = pid_bh // n_heads
    head      = pid_bh % n_heads
    kv_head   = head // n_groups  # GQA mapping: which KV head

    sq_start = pid_sq * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < sq

    skv_offs = tl.arange(0, BLOCK_SKV)
    hd_offs  = tl.arange(0, BLOCK_HD)

    # Base pointers for this (batch, head)
    dO_base = dO_ptr + batch_idx * stride_do_bs + head * stride_do_h
    V_base  = V_ptr  + batch_idx * stride_v_bs  + kv_head * stride_v_kvh
    P_base  = P_ptr  + batch_idx * stride_p_bs  + head * stride_p_h
    Pd_base = Pd_ptr + batch_idx * stride_pd_bs + head * stride_pd_h
    dS_base = dS_ptr + batch_idx * stride_ds_bs + head * stride_ds_h

    num_skv_blocks = tl.cdiv(skv, BLOCK_SKV)
    num_hd_blocks  = HEAD_DIM // BLOCK_HD  # exact since HEAD_DIM is multiple of BLOCK_HD

    # ---- Pass 1: accumulate rowsum ----
    rowsum = tl.zeros((BLOCK_SQ,), dtype=tl.float32)

    for skv_tile in range(num_skv_blocks):
        skv_start     = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_msk       = skv_tile_offs < skv

        # Compute dP_tile = dO_block @ V_tile^T via blocked inner dot
        # dO_block: [BLOCK_SQ, HEAD_DIM], V_tile: [BLOCK_SKV, HEAD_DIM]
        # dP_tile:  [BLOCK_SQ, BLOCK_SKV]
        acc = tl.zeros((BLOCK_SQ, BLOCK_SKV), dtype=tl.float32)

        for hd_tile in range(num_hd_blocks):
            hd_start     = hd_tile * BLOCK_HD
            hd_tile_offs = hd_start + hd_offs

            # Load dO block: [BLOCK_SQ, BLOCK_HD]
            do_ptrs = dO_base + sq_offs[:, None] * stride_do_sq + hd_tile_offs[None, :] * stride_do_d
            do_block = tl.load(do_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)

            # Load V tile: [BLOCK_SKV, BLOCK_HD]
            v_ptrs = V_base + skv_tile_offs[:, None] * stride_v_skv + hd_tile_offs[None, :] * stride_v_d
            v_tile = tl.load(v_ptrs, mask=skv_msk[:, None], other=0.0).to(tl.float32)

            # dP accumulation: [BLOCK_SQ, BLOCK_HD] @ [BLOCK_HD, BLOCK_SKV] -> [BLOCK_SQ, BLOCK_SKV]
            acc += tl.dot(do_block.to(tl.bfloat16), tl.trans(v_tile).to(tl.bfloat16), allow_tf32=False).to(tl.float32)

        # Load P_dropped to detect dropout mask
        pd_ptrs = Pd_base + sq_offs[:, None] * stride_pd_sq + skv_tile_offs[None, :] * stride_pd_skv
        pd_tile = tl.load(pd_ptrs, mask=sq_mask[:, None] & skv_msk[None, :], other=0.0).to(tl.float32)
        kept = tl.where(pd_tile != 0.0, 1.0, 0.0)
        dp_masked = acc * kept * inv_keep_prob

        # Load P for rowsum accumulation
        p_ptrs = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile = tl.load(p_ptrs, mask=sq_mask[:, None] & skv_msk[None, :], other=0.0).to(tl.float32)

        rowsum += tl.sum(dp_masked * p_tile, axis=1)

    # ---- Pass 2: compute and store dS ----
    for skv_tile in range(num_skv_blocks):
        skv_start     = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_msk       = skv_tile_offs < skv
        combined      = sq_mask[:, None] & skv_msk[None, :]

        # Recompute dP_tile
        acc = tl.zeros((BLOCK_SQ, BLOCK_SKV), dtype=tl.float32)

        for hd_tile in range(num_hd_blocks):
            hd_start     = hd_tile * BLOCK_HD
            hd_tile_offs = hd_start + hd_offs

            do_ptrs  = dO_base + sq_offs[:, None] * stride_do_sq + hd_tile_offs[None, :] * stride_do_d
            do_block = tl.load(do_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)

            v_ptrs  = V_base + skv_tile_offs[:, None] * stride_v_skv + hd_tile_offs[None, :] * stride_v_d
            v_tile  = tl.load(v_ptrs, mask=skv_msk[:, None], other=0.0).to(tl.float32)

            acc += tl.dot(do_block.to(tl.bfloat16), tl.trans(v_tile).to(tl.bfloat16), allow_tf32=False).to(tl.float32)

        # Reload dropout-masked dP
        pd_ptrs   = Pd_base + sq_offs[:, None] * stride_pd_sq + skv_tile_offs[None, :] * stride_pd_skv
        pd_tile   = tl.load(pd_ptrs, mask=combined, other=0.0).to(tl.float32)
        kept      = tl.where(pd_tile != 0.0, 1.0, 0.0)
        dp_masked = acc * kept * inv_keep_prob

        # Reload P
        p_ptrs  = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile  = tl.load(p_ptrs, mask=combined, other=0.0).to(tl.float32)

        dS_tile = p_tile * (dp_masked - rowsum[:, None])

        ds_ptrs = dS_base + sq_offs[:, None] * stride_ds_sq + skv_tile_offs[None, :] * stride_ds_skv
        tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=combined)


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


def _make_dO_transposed_contiguous(grad_attn_output, bs, n_heads, seq_q, d):
    """
    Convert grad_attn_output from [bs, sq, 80, d] to [bs, 80, sq, d] contiguous.
    """
    return grad_attn_output.permute(0, 2, 1, 3).contiguous()


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

    inv_keep = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # For dV: we need [bs*8, 10*sq, d] dO and [bs*8, 10*sq, skv] P_dropped
    dO_gqa = _make_dO_gqa_contiguous(grad_attn_output, bs, n_kv_heads, n_groups, seq_q, d)
    # dO_gqa: [bs*8, 10*sq, d] bfloat16 contiguous

    # V: [bs, 8, skv, d] -> [bs*8, skv, d] — NO expansion needed!
    V_gqa = value_states.reshape(bs * n_kv_heads, seq_kv, d)

    # P_dropped for dV: [bs, 80, sq, skv] -> [bs*8, 10*sq, skv]
    P_dropped_gqa = attn_weights_dropped.view(bs, n_kv_heads, n_groups, seq_q, seq_kv) \
                                        .reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # Pre-allocate output tensors
    dV_gqa = torch.empty((bs * n_kv_heads, seq_kv, d), dtype=torch.bfloat16, device=device)
    dS     = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=device)

    # For the fused kernel we need dO in [bs, 80, sq, d] layout
    dO_transposed = _make_dO_transposed_contiguous(grad_attn_output, bs, n_heads, seq_q, d)
    # [bs, 80, sq, d] contiguous bfloat16

    P_attn = attn_weights.contiguous()          # [bs, 80, sq, skv] bfloat16
    P_drop = attn_weights_dropped.contiguous()  # [bs, 80, sq, skv] bfloat16

    stream1, stream2 = _get_streams()
    current_stream = torch.cuda.current_stream(device)

    # ---- FUSED KERNEL: dP+softmax_bwd on stream1 ----
    with torch.cuda.stream(stream1):
        stream1.wait_stream(current_stream)
        grid_dS = lambda meta: (bs * n_heads, triton.cdiv(seq_q, meta['BLOCK_SQ']))
        fused_dp_softmax_bwd_kernel[grid_dS](
            dO_transposed,
            dO_transposed.stride(0), dO_transposed.stride(1),
            dO_transposed.stride(2), dO_transposed.stride(3),
            value_states,
            value_states.stride(0), value_states.stride(1),
            value_states.stride(2), value_states.stride(3),
            P_attn,
            P_attn.stride(0), P_attn.stride(1), P_attn.stride(2), P_attn.stride(3),
            P_drop,
            P_drop.stride(0), P_drop.stride(1), P_drop.stride(2), P_drop.stride(3),
            dS,
            dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
            bs, n_heads, n_groups, seq_q, seq_kv,
            inv_keep_prob=inv_keep,
            HEAD_DIM=d,
        )

    # ---- GEMM 2 (BF16): dV = P_dropped_gqa^T @ dO_gqa on stream2 ----
    with torch.cuda.stream(stream2):
        stream2.wait_stream(current_stream)
        # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
        torch.bmm(P_dropped_gqa.transpose(-2, -1), dO_gqa, out=dV_gqa)

    # Synchronize both streams back to current stream
    current_stream.wait_stream(stream1)
    current_stream.wait_stream(stream2)

    # Reshape dV to [bs, 8, skv, d]
    dV = dV_gqa.view(bs, n_kv_heads, seq_kv, d).to(torch.bfloat16)

    return dS, dV
