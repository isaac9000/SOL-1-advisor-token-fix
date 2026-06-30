"""
Optimized attention-backward kernel — fused Triton softmax-bwd + GQA-aware BMMs.

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

    # Load all data — single read from global memory
    dp_dropped = tl.load(dP_dropped_ptr + row_start + offsets,
                         mask=mask_cond, other=0.0).to(tl.float32)
    p_val = tl.load(P_ptr + row_start + offsets,
                    mask=mask_cond, other=0.0).to(tl.float32)
    m_val = tl.load(mask_ptr + row_start + offsets,
                    mask=mask_cond, other=0).to(tl.float32)

    # Compute dP (in-register)
    dp = dp_dropped * m_val * scale

    # Compute row_sum = sum(dP * P) — in-register reduction, no extra global read
    row_sum = tl.sum(dp * p_val, axis=0)

    # Compute dS = P * (dP - row_sum) — all in-register
    ds = p_val * (dp - row_sum)

    # Single write to global memory
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
    """
    Two-pass kernel for large seq_kv (when full row doesn't fit in registers).
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * seq_kv

    # First pass: compute sum(dP * P)
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

    # Second pass: compute dS and store
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
    """
    dP_dropped: [N_rows, seq_kv] bfloat16
    P:          [N_rows, seq_kv] bfloat16
    mask:       [N_rows, seq_kv] bool
    Returns dS: [N_rows, seq_kv] bfloat16
    """
    N_rows = dP_dropped.shape[0]
    dS = torch.empty_like(dP_dropped)

    pow2 = 2 ** math.ceil(math.log2(seq_kv)) if seq_kv > 1 else 1

    grid = (N_rows,)

    SINGLE_PASS_MAX = 4096

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


@torch.compile(mode="max-autotune", fullgraph=True)
def _compute_bmms(dO_in, value_states, attn_weights_dropped, bs, seq_q, seq_kv, n_kv, n_g, d):
    """
    Compute both BMMs with torch.compile(max-autotune) so the compiler can
    fuse the transpose into the GEMM (no standalone .contiguous() copy).

    dO_in:              [bs, sq, n_kv*n_g, d]   bfloat16
    value_states:       [bs, n_kv, skv, d]       bfloat16
    attn_weights_dropped: [bs, n_kv*n_g, sq, skv] bfloat16

    Returns:
        dP_dropped_flat: [bs*n_kv, n_g*sq, skv]  bfloat16
        dV_flat:         [bs*n_kv, skv, d]        bfloat16
    """
    # Reshape dO_in from [bs, sq, n_kv*n_g, d] to [bs*n_kv, n_g*sq, d]
    # via permute: [bs, sq, n_kv, n_g, d] -> [bs, n_kv, n_g, sq, d] -> [bs*n_kv, n_g*sq, d]
    # The compiler sees the permute and can fuse it into the GEMM as a strided read.
    dO_perm = dO_in.reshape(bs, seq_q, n_kv, n_g, d).permute(0, 2, 3, 1, 4).reshape(bs * n_kv, n_g * seq_q, d)

    # value_states: [bs, n_kv, skv, d] -> [bs*n_kv, skv, d]
    vs_flat = value_states.reshape(bs * n_kv, seq_kv, d)

    # BMM 1: dP_dropped = dO @ V^T
    # [bs*n_kv, n_g*sq, d] @ [bs*n_kv, d, skv] -> [bs*n_kv, n_g*sq, skv]
    dP_dropped_flat = torch.bmm(dO_perm, vs_flat.transpose(-2, -1))

    # attn_weights_dropped: [bs, n_kv*n_g, sq, skv] -> [bs*n_kv, n_g*sq, skv]
    Pd_flat = attn_weights_dropped.reshape(bs * n_kv, n_g * seq_q, seq_kv)

    # BMM 2: dV = P^T @ dO
    # [bs*n_kv, skv, n_g*sq] @ [bs*n_kv, n_g*sq, d] -> [bs*n_kv, skv, d]
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_perm)

    return dP_dropped_flat, dV_flat


def custom_kernel(data):
    (dO_in, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = dO_in.shape[0]
    seq_q  = dO_in.shape[1]
    seq_kv = value_states.shape[2]
    n_kv   = NUM_KEY_VALUE_HEADS   # 8
    n_g    = N_GROUPS              # 10
    d      = HEAD_DIM              # 128

    # Compute both BMMs using torch.compile(max-autotune) to fuse transpose into GEMM
    dP_dropped_flat, dV_flat = _compute_bmms(
        dO_in, value_states, attn_weights_dropped,
        bs, seq_q, seq_kv, n_kv, n_g, d
    )

    # Reshape back: [bs*n_kv, n_g*sq, skv] -> [bs, 80, sq, skv]
    dP_dropped = dP_dropped_flat.reshape(bs, n_kv * n_g, seq_q, seq_kv)

    # ── Fused softmax backward + dropout (single-pass Triton kernel) ──────────
    scale = 1.0 / (1.0 - attention_dropout)

    # Flatten row dimensions: [bs, 80, sq, skv] -> [bs*80*sq, skv]
    N_rows = bs * NUM_ATTENTION_HEADS * seq_q
    dP_dropped_2d = dP_dropped.contiguous().reshape(N_rows, seq_kv)
    P_2d = attn_weights.contiguous().reshape(N_rows, seq_kv)
    mask_2d = dropout_mask.contiguous().reshape(N_rows, seq_kv)

    dS_2d = fused_softmax_bwd_dropout(dP_dropped_2d, P_2d, mask_2d, scale, seq_kv)
    dS = dS_2d.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # Reshape dV to [bs, 8, skv, d]
    dV = dV_flat.reshape(bs, n_kv, seq_kv, d).to(torch.bfloat16)

    return dS, dV
