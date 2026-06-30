"""
Optimized attention-backward kernel — fused Triton softmax-bwd + fused two-output BMM.

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


# ---------------------------------------------------------------------------
# Fused two-output GEMM kernel: computes both
#   dP_dropped[b, g*q_start + q, kv] = sum_d dO[b, g, q, d] * V[b, kv, d]
#   dV[b, kv, d]                     = sum_{g,q} Pd[b, g, q, kv] * dO[b, g, q, d]
#
# Grid: (bs * n_kv, cdiv(seq_kv, BLOCK_KV))
#   - We tile over KV dimension
#   - Inner loop: over seq_q blocks, loading dO and P tiles
#
# For each (batch*kv_head, kv_tile):
#   - Iterate over all (n_groups * seq_q) query rows
#   - For each row: load dO[d] once, accumulate into dV[kv_tile, d]
#                   also compute dot with V[kv_tile, d] -> writes dP_dropped row
#
# Shapes after reshape (bs*n_kv view):
#   dO_flat:   [bs*8, 10*sq, 128]   bfloat16
#   vs_flat:   [bs*8, skv, 128]     bfloat16
#   Pd_flat:   [bs*8, 10*sq, skv]   bfloat16  (attn_weights_dropped)
#   dP_out:    [bs*8, 10*sq, skv]   bfloat16  (output)
#   dV_out:    [bs*8, skv, 128]     bfloat16  (output)
# ---------------------------------------------------------------------------

@triton.jit
def _fused_bmm_kernel(
    # Inputs
    dO_ptr,    # [B, M, D]   bfloat16   (B = bs*n_kv, M = n_g*seq_q, D = head_dim)
    V_ptr,     # [B, N, D]   bfloat16   (N = seq_kv)
    Pd_ptr,    # [B, M, N]   bfloat16   (attn_weights_dropped)
    # Outputs
    dP_ptr,    # [B, M, N]   bfloat16
    dV_ptr,    # [B, N, D]   bfloat16
    # Strides for dO [B, M, D]
    dO_stride_b, dO_stride_m, dO_stride_d,
    # Strides for V [B, N, D]
    V_stride_b, V_stride_n, V_stride_d,
    # Strides for Pd [B, M, N]
    Pd_stride_b, Pd_stride_m, Pd_stride_n,
    # Strides for dP [B, M, N]
    dP_stride_b, dP_stride_m, dP_stride_n,
    # Strides for dV [B, N, D]
    dV_stride_b, dV_stride_n, dV_stride_d,
    # Dimensions
    B, M, N, D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused kernel computing both:
      dP[b, m, n] = sum_d dO[b, m, d] * V[b, n, d]   (for all m in BLOCK_M, n in BLOCK_N)
      dV[b, n, d] = sum_m Pd[b, m, n] * dO[b, m, d]  (accumulated for all m)

    Grid: (B, cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
    Each program handles one (b, m_block, n_block) tile.
    It computes dP for that tile and accumulates into dV using atomic adds.
    """
    pid_b  = tl.program_id(0)
    pid_m  = tl.program_id(1)
    pid_n  = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, BLOCK_D)  # D must be <= BLOCK_D (constexpr = HEAD_DIM=128)

    m_mask = m_offs < M
    n_mask = n_offs < N

    # Load dO tile: [BLOCK_M, BLOCK_D]
    dO_base = pid_b * dO_stride_b
    dO_ptrs = dO_base + m_offs[:, None] * dO_stride_m + d_offs[None, :] * dO_stride_d
    dO_tile = tl.load(dO_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

    # Load V tile: [BLOCK_N, BLOCK_D]
    V_base = pid_b * V_stride_b
    V_ptrs = V_base + n_offs[:, None] * V_stride_n + d_offs[None, :] * V_stride_d
    V_tile = tl.load(V_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

    # Compute dP tile: [BLOCK_M, BLOCK_N] = dO_tile @ V_tile^T
    dP_tile = tl.dot(dO_tile, tl.trans(V_tile))  # [BLOCK_M, BLOCK_N]

    # Store dP tile
    dP_base = pid_b * dP_stride_b
    dP_ptrs = dP_base + m_offs[:, None] * dP_stride_m + n_offs[None, :] * dP_stride_n
    tl.store(dP_ptrs, dP_tile.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])

    # Load Pd tile: [BLOCK_M, BLOCK_N]
    Pd_base = pid_b * Pd_stride_b
    Pd_ptrs = Pd_base + m_offs[:, None] * Pd_stride_m + n_offs[None, :] * Pd_stride_n
    Pd_tile = tl.load(Pd_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

    # Compute dV contribution: [BLOCK_N, BLOCK_D] = Pd_tile^T @ dO_tile
    dV_contrib = tl.dot(tl.trans(Pd_tile), dO_tile)  # [BLOCK_N, BLOCK_D]

    # Atomic add into dV
    dV_base = pid_b * dV_stride_b
    dV_ptrs = dV_base + n_offs[:, None] * dV_stride_n + d_offs[None, :] * dV_stride_d
    tl.atomic_add(dV_ptrs, dV_contrib.to(tl.float32), mask=n_mask[:, None])


def fused_bmm_dp_dv(dO_flat, vs_flat, Pd_flat, seq_kv, bs_kv, M, D):
    """
    dO_flat: [bs*8, 10*sq, 128]   bfloat16
    vs_flat: [bs*8, skv, 128]     bfloat16
    Pd_flat: [bs*8, 10*sq, skv]   bfloat16
    Returns:
        dP_flat: [bs*8, 10*sq, skv]  bfloat16
        dV_flat: [bs*8, skv, 128]    bfloat16
    """
    B = bs_kv
    N = seq_kv

    dP_flat = torch.empty(B, M, N, dtype=torch.bfloat16, device=dO_flat.device)
    dV_flat = torch.zeros(B, N, D, dtype=torch.float32, device=dO_flat.device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D  # 128, constexpr

    grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _fused_bmm_kernel[grid](
        dO_flat, vs_flat, Pd_flat,
        dP_flat, dV_flat,
        dO_flat.stride(0), dO_flat.stride(1), dO_flat.stride(2),
        vs_flat.stride(0), vs_flat.stride(1), vs_flat.stride(2),
        Pd_flat.stride(0), Pd_flat.stride(1), Pd_flat.stride(2),
        dP_flat.stride(0), dP_flat.stride(1), dP_flat.stride(2),
        dV_flat.stride(0), dV_flat.stride(1), dV_flat.stride(2),
        B=B, M=M, N=N, D=BLOCK_D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return dP_flat, dV_flat.to(torch.bfloat16)


def custom_kernel(data):
    (dO_in, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = dO_in.shape[0]
    seq_q  = dO_in.shape[1]
    seq_kv = value_states.shape[2]
    n_kv   = NUM_KEY_VALUE_HEADS   # 8
    n_g    = N_GROUPS              # 10
    d      = HEAD_DIM              # 128

    # [bs, sq, 80, d] -> [bs, 80, sq, d] -> contiguous bfloat16
    dO = dO_in.transpose(1, 2).contiguous()  # [bs, 80, sq, d] bfloat16

    # Reshape for GQA-aware computation
    # [bs, 80, sq, d] -> [bs*8, 10*sq, d]
    M = n_g * seq_q
    dO_flat = dO.reshape(bs, n_kv, n_g, seq_q, d).reshape(bs * n_kv, M, d)
    vs_flat = value_states.reshape(bs * n_kv, seq_kv, d)
    Pd_flat = attn_weights_dropped.reshape(bs * n_kv, M, seq_kv)

    # Check if fused kernel is feasible (D must equal HEAD_DIM=128 which is constexpr)
    # Use fused kernel to compute both dP and dV in one pass over dO
    try:
        dP_dropped_flat, dV_flat = fused_bmm_dp_dv(
            dO_flat.contiguous(),
            vs_flat.contiguous(),
            Pd_flat.contiguous(),
            seq_kv, bs * n_kv, M, d
        )
    except Exception:
        # Fallback to separate BMMs if fused kernel fails
        dP_dropped_flat = torch.bmm(dO_flat, vs_flat.transpose(-2, -1))
        dV_flat_fp = torch.bmm(Pd_flat.transpose(-2, -1).float(), dO_flat.float())
        dV_flat = dV_flat_fp.to(torch.bfloat16)

    # Reshape back: [bs*8, 10*sq, skv] -> [bs, 80, sq, skv]
    dP_dropped = dP_dropped_flat.reshape(bs, n_kv * n_g, seq_q, seq_kv)

    # ── Fused softmax backward + dropout (single-pass Triton kernel) ──────────
    scale = 1.0 / (1.0 - attention_dropout)

    N_rows = bs * NUM_ATTENTION_HEADS * seq_q
    dP_dropped_2d = dP_dropped.contiguous().reshape(N_rows, seq_kv)
    P_2d = attn_weights.contiguous().reshape(N_rows, seq_kv)
    mask_2d = dropout_mask.contiguous().reshape(N_rows, seq_kv)

    dS_2d = fused_softmax_bwd_dropout(dP_dropped_2d, P_2d, mask_2d, scale, seq_kv)
    dS = dS_2d.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # Reshape dV to [bs, 8, skv, d]
    dV = dV_flat.reshape(bs, n_kv, seq_kv, d)

    return dS, dV
