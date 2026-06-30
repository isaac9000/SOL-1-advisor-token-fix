"""
Optimized attention-backward kernel — cuBLAS BMMs with stream parallelism.

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
import math

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10


@triton.jit
def _softmax_bwd_dropout_kernel_singlepass(
    dP_dropped_ptr,   # [N_rows, seq_kv]  bfloat16
    P_ptr,            # [N_rows, seq_kv]  bfloat16
    mask_ptr,         # [N_rows, seq_kv]  bool
    dS_ptr,           # [N_rows, seq_kv]  bfloat16  output
    seq_kv: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * seq_kv

    offsets = tl.arange(0, BLOCK_SIZE)
    mask_cond = offsets < seq_kv

    dp_dropped = tl.load(dP_dropped_ptr + row_start + offsets,
                         mask=mask_cond, other=0.0).to(tl.float32)
    p_val = tl.load(P_ptr + row_start + offsets,
                    mask=mask_cond, other=0.0).to(tl.float32)
    m_val = tl.load(mask_ptr + row_start + offsets,
                    mask=mask_cond, other=0).to(tl.float32)

    dp = dp_dropped * m_val * scale
    row_sum = tl.sum(dp * p_val, axis=0)
    ds = p_val * (dp - row_sum)

    tl.store(dS_ptr + row_start + offsets,
             ds.to(tl.bfloat16),
             mask=mask_cond)


@triton.jit
def _softmax_bwd_dropout_kernel_multiblock(
    dP_dropped_ptr,   # [N_rows, seq_kv]  bfloat16
    P_ptr,            # [N_rows, seq_kv]  bfloat16
    mask_ptr,         # [N_rows, seq_kv]  bool
    dS_ptr,           # [N_rows, seq_kv]  bfloat16  output
    seq_kv: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * seq_kv

    acc = tl.zeros([1], dtype=tl.float32)
    for block_start in tl.range(0, seq_kv, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask_cond = offsets < seq_kv

        dp_dropped = tl.load(dP_dropped_ptr + row_start + offsets,
                             mask=mask_cond, other=0.0).to(tl.float32)
        p_val = tl.load(P_ptr + row_start + offsets,
                        mask=mask_cond, other=0.0).to(tl.float32)
        m_val = tl.load(mask_ptr + row_start + offsets,
                        mask=mask_cond, other=0).to(tl.float32)

        dp = dp_dropped * m_val * scale
        acc += tl.sum(dp * p_val, axis=0)

    row_sum = tl.sum(acc, axis=0)

    for block_start in tl.range(0, seq_kv, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask_cond = offsets < seq_kv

        dp_dropped = tl.load(dP_dropped_ptr + row_start + offsets,
                             mask=mask_cond, other=0.0).to(tl.float32)
        p_val = tl.load(P_ptr + row_start + offsets,
                        mask=mask_cond, other=0.0).to(tl.float32)
        m_val = tl.load(mask_ptr + row_start + offsets,
                        mask=mask_cond, other=0).to(tl.float32)

        dp = dp_dropped * m_val * scale
        ds = p_val * (dp - row_sum)

        tl.store(dS_ptr + row_start + offsets,
                 ds.to(tl.bfloat16),
                 mask=mask_cond)


def fused_softmax_bwd_dropout(dP_dropped, P, mask, scale, seq_kv):
    N_rows = dP_dropped.shape[0]
    dS = torch.empty_like(dP_dropped)

    SINGLE_PASS_MAX = 4096
    pow2 = 2 ** math.ceil(math.log2(seq_kv)) if seq_kv > 1 else 1

    grid = (N_rows,)

    if pow2 <= SINGLE_PASS_MAX:
        BLOCK_SIZE = pow2
        _softmax_bwd_dropout_kernel_singlepass[grid](
            dP_dropped, P, mask, dS,
            seq_kv=seq_kv,
            scale=scale,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        BLOCK_SIZE = 2048
        _softmax_bwd_dropout_kernel_multiblock[grid](
            dP_dropped, P, mask, dS,
            seq_kv=seq_kv,
            scale=scale,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return dS


# Persistent side stream for BMM2 overlap
_side_stream = None

def _get_side_stream():
    global _side_stream
    if _side_stream is None:
        _side_stream = torch.cuda.Stream()
    return _side_stream


def custom_kernel(data):
    (dO_in, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = dO_in.shape[0]
    seq_q  = dO_in.shape[1]
    seq_kv = value_states.shape[2]
    n_kv   = NUM_KEY_VALUE_HEADS   # 8
    n_g    = N_GROUPS              # 10
    d      = HEAD_DIM              # 128
    n_heads = NUM_ATTENTION_HEADS  # 80

    # ── Step 1: Compute dO (no contiguous, keep strided) ─────────────────────
    # [bs, sq, 80, d] -> [bs, 80, sq, d] (strided, no copy)
    dO = dO_in.transpose(1, 2)  # [bs, 80, sq, d] bfloat16 — strided

    # Reshape for bs*80 batched matmul: each head gets its own [sq, d] matrix
    # [bs, 80, sq, d] -> [bs*80, sq, d]  — need contiguous for bmm
    dO_flat_80 = dO.contiguous().reshape(bs * n_heads, seq_q, d)  # [bs*80, sq, d]

    # Value states expanded lazily: [bs, 8, skv, d] -> [bs, 80, skv, d] via expand
    # Then reshape to [bs*80, skv, d] — expand is zero-copy, reshape needs contiguous
    vs_exp = value_states.unsqueeze(2).expand(bs, n_kv, n_g, seq_kv, d) \
                         .reshape(bs * n_heads, seq_kv, d)  # needs contiguous for bmm
    # vs_exp is strided after expand; make contiguous only if needed by bmm
    # Actually torch.bmm requires contiguous — use reshape which may force it
    vs_exp = vs_exp.contiguous()  # [bs*80, skv, d]

    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs*80, sq, skv]
    Pd_flat_80 = attn_weights_dropped.reshape(bs * n_heads, seq_q, seq_kv)

    # ── Step 2: Fork — launch BMM2 on side stream BEFORE BMM1 ────────────────
    side_stream    = _get_side_stream()
    default_stream = torch.cuda.current_stream()

    side_stream.wait_stream(default_stream)

    with torch.cuda.stream(side_stream):
        # BMM2: dV = Pd^T @ dO  shape: [bs*80, skv, sq] @ [bs*80, sq, d] -> [bs*80, skv, d]
        # Then sum over the 10 groups: reshape [bs, 8, 10, skv, d] -> sum over dim=2
        dV_flat_fp32 = torch.bmm(
            Pd_flat_80.float().transpose(-2, -1),   # [bs*80, skv, sq]
            dO_flat_80.float()                       # [bs*80, sq, d]
        )  # [bs*80, skv, d]
        # Sum over groups: [bs*80, skv, d] -> [bs, 8, 10, skv, d] -> sum -> [bs, 8, skv, d]
        dV_fp32 = dV_flat_fp32.reshape(bs, n_kv, n_g, seq_kv, d).sum(dim=2)

    # ── Step 3: BMM1 on default stream ───────────────────────────────────────
    # dP_dropped = dO @ V^T
    # [bs*80, sq, d] @ [bs*80, d, skv] -> [bs*80, sq, skv]
    dP_dropped_flat = torch.bmm(dO_flat_80, vs_exp.transpose(-2, -1))  # cuBLAS bf16

    # ── Step 4: Softmax backward (Triton, on default stream) ─────────────────
    dP_dropped = dP_dropped_flat.reshape(bs, n_heads, seq_q, seq_kv)

    scale  = 1.0 / (1.0 - attention_dropout)
    N_rows = bs * n_heads * seq_q

    dP_dropped_2d = dP_dropped.reshape(N_rows, seq_kv)
    P_2d          = attn_weights.reshape(N_rows, seq_kv)
    mask_2d       = dropout_mask.reshape(N_rows, seq_kv)

    dS_2d = fused_softmax_bwd_dropout(dP_dropped_2d, P_2d, mask_2d, scale, seq_kv)
    dS    = dS_2d.reshape(bs, n_heads, seq_q, seq_kv)

    # ── Step 5: Wait for BMM2, finalize dV ───────────────────────────────────
    default_stream.wait_stream(side_stream)
    dV = dV_fp32.to(torch.bfloat16)

    return dS, dV
