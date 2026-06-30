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

    # ── Step 1: Compute dO on the default stream (needed by both BMMs) ───────
    # [bs, sq, 80, d] -> [bs, 80, sq, d] -> contiguous bfloat16
    dO = dO_in.transpose(1, 2).contiguous()  # [bs, 80, sq, d] bfloat16

    # Reshape for GQA-aware computation using cuBLAS BMMs
    # [bs, 80, sq, d] -> [bs*8, 10*sq, d]
    M = n_g * seq_q
    dO_flat  = dO.reshape(bs * n_kv, M, d)                       # [bs*8, 10*sq, d]
    vs_flat  = value_states.reshape(bs * n_kv, seq_kv, d)        # [bs*8, skv, d]
    Pd_flat  = attn_weights_dropped.reshape(bs * n_kv, M, seq_kv) # [bs*8, 10*sq, skv]

    # ── Step 2: Fork — launch BMM2 on side stream BEFORE BMM1 ────────────────
    # BMM2: dV = Pd^T @ dO  (independent of BMM1 output)
    # Both streams need dO_flat which is now ready on the default stream.
    side_stream    = _get_side_stream()
    default_stream = torch.cuda.current_stream()

    # Tell side stream to wait until dO_flat / Pd_flat are ready
    side_stream.wait_stream(default_stream)

    with torch.cuda.stream(side_stream):
        # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
        dV_flat_fp32 = torch.bmm(
            Pd_flat.transpose(-2, -1).float(),
            dO_flat.float()
        )

    # ── Step 3: BMM1 on default stream (can run in parallel with BMM2) ───────
    # dP_dropped = dO @ V^T
    # [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
    dP_dropped_flat = torch.bmm(dO_flat, vs_flat.transpose(-2, -1))  # cuBLAS bf16

    # ── Step 4: Softmax backward (Triton, on default stream) ─────────────────
    dP_dropped = dP_dropped_flat.reshape(bs, n_kv * n_g, seq_q, seq_kv)

    scale  = 1.0 / (1.0 - attention_dropout)
    N_rows = bs * NUM_ATTENTION_HEADS * seq_q

    dP_dropped_2d = dP_dropped.reshape(N_rows, seq_kv)
    P_2d          = attn_weights.reshape(N_rows, seq_kv)
    mask_2d       = dropout_mask.reshape(N_rows, seq_kv)

    dS_2d = fused_softmax_bwd_dropout(dP_dropped_2d, P_2d, mask_2d, scale, seq_kv)
    dS    = dS_2d.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ── Step 5: Wait for BMM2, finalize dV ───────────────────────────────────
    default_stream.wait_stream(side_stream)
    dV = dV_flat_fp32.to(torch.bfloat16).reshape(bs, n_kv, seq_kv, d)

    return dS, dV
