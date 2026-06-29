"""
Optimized attention-backward kernel using a fused Triton kernel.

Fuses:
  1. dP = dO @ V^T  (GEMM)
  2. dropout mask + scale
  3. softmax backward: dS = P*(dP - sum(dP*P, dim=-1))
  4. dV += P_dropped^T @ dO  (GEMM accumulation)

The large [bs, 80, sq, skv] intermediate dP is NEVER written to HBM.
GQA: 80 query heads, 8 KV heads, 10 groups per KV head, head_dim=128.

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
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10


@triton.jit
def _attn_bwd_kernel(
    # Inputs
    dO_ptr,        # [bs, 80, sq, 128]  bf16
    P_ptr,         # [bs, 80, sq, skv]  bf16
    Pd_ptr,        # [bs, 80, sq, skv]  bf16  (attn_weights_dropped)
    V_ptr,         # [bs,  8, skv, 128] bf16
    Mask_ptr,      # [bs, 80, sq, skv]  bool (uint8)
    # Outputs
    dS_ptr,        # [bs, 80, sq, skv]  bf16
    dV_ptr,        # [bs,  8, skv, 128] bf16
    # Dimensions
    bs: tl.constexpr,
    sq: tl.constexpr,
    skv: tl.constexpr,
    n_heads: tl.constexpr,      # 80
    n_kv_heads: tl.constexpr,   # 8
    n_groups: tl.constexpr,     # 10
    head_dim: tl.constexpr,     # 128
    inv_keep_prob: tl.constexpr,  # 1/(1-dropout)
    # Strides for dO [bs, 80, sq, 128]
    dO_stride_b: tl.constexpr,
    dO_stride_h: tl.constexpr,
    dO_stride_q: tl.constexpr,
    # Strides for P/Pd/Mask [bs, 80, sq, skv]
    P_stride_b: tl.constexpr,
    P_stride_h: tl.constexpr,
    P_stride_q: tl.constexpr,
    # Strides for V [bs, 8, skv, 128]
    V_stride_b: tl.constexpr,
    V_stride_h: tl.constexpr,
    V_stride_k: tl.constexpr,
    # Strides for dV [bs, 8, skv, 128]
    dV_stride_b: tl.constexpr,
    dV_stride_h: tl.constexpr,
    dV_stride_k: tl.constexpr,
    # Block sizes
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """
    Grid: (bs * n_heads, ceil(sq / BLOCK_Q))
    Each program handles a [BLOCK_Q x skv] tile of one (batch, head).

    For dS: we compute the full row of dS for BLOCK_Q queries.
    For dV: we accumulate in float32 registers over the BLOCK_Q queries,
            then atomically add to the kv-head slot.
    """
    # Program ids
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)

    b_idx  = pid_bh // n_heads
    h_idx  = pid_bh % n_heads
    kv_idx = h_idx // n_groups  # which KV head

    q_start = pid_q * BLOCK_Q
    q_offs  = q_start + tl.arange(0, BLOCK_Q)
    q_mask  = q_offs < sq

    d_offs  = tl.arange(0, head_dim)
    kv_offs = tl.arange(0, BLOCK_KV)

    # Base pointers for this (batch, head)
    dO_base = dO_ptr + b_idx * dO_stride_b + h_idx * dO_stride_h
    P_base  = P_ptr  + b_idx * P_stride_b  + h_idx * P_stride_h
    Pd_base = Pd_ptr + b_idx * P_stride_b  + h_idx * P_stride_h
    Mask_base = Mask_ptr + b_idx * P_stride_b + h_idx * P_stride_h
    V_base  = V_ptr  + b_idx * V_stride_b  + kv_idx * V_stride_h
    dS_base = dS_ptr + b_idx * P_stride_b  + h_idx * P_stride_h
    dV_base = dV_ptr + b_idx * dV_stride_b + kv_idx * dV_stride_h

    # Load dO tile: [BLOCK_Q, head_dim]
    dO_ptrs = dO_base + (q_offs[:, None] * dO_stride_q + d_offs[None, :])
    dO_tile = tl.load(dO_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    # We need Di = sum_k(dP_k * P_k) per row = sum_k((dP_dropped_k / keep * P_k))
    # = sum over all kv of dP_ij * P_ij
    # We compute this in a first pass over kv blocks, then use it in second pass.

    # First pass: compute Di[BLOCK_Q] = sum_{k} dP[q,k] * P[q,k]
    # dP[q,k] = dP_dropped[q,k] * mask[q,k] * inv_keep_prob
    # = (dO @ V^T)[q,k] * mask[q,k] * inv_keep_prob
    # But attn_weights_dropped / P already encode dropout differently—
    # Actually we have: Pd = attn_weights_dropped (post-dropout weights, already scaled)
    # Reference: dP = dP_dropped * mask / (1 - p)
    # where dP_dropped = dO @ V^T
    # But we also have: Pd = P * mask / (1-p)   [standard dropout scaling]
    # Di = sum(dP * P) = sum(dO @ V^T * mask * inv_keep_prob * P)

    # Accumulate Di across all kv blocks
    Di = tl.zeros([BLOCK_Q], dtype=tl.float32)

    n_kv_blocks = tl.cdiv(skv, BLOCK_KV)

    for kv_blk in range(0, n_kv_blocks):
        kv_start = kv_blk * BLOCK_KV
        kv_block = kv_start + kv_offs
        kv_mask  = kv_block < skv

        # Load V tile [BLOCK_KV, head_dim]
        V_ptrs = V_base + (kv_block[:, None] * V_stride_k + d_offs[None, :])
        V_tile = tl.load(V_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

        # dP_dropped = dO @ V^T  -> [BLOCK_Q, BLOCK_KV]
        dPd_tile = tl.dot(dO_tile, tl.trans(V_tile))  # float32

        # Load dropout mask [BLOCK_Q, BLOCK_KV]
        Mask_ptrs = Mask_base + (q_offs[:, None] * P_stride_q + kv_block[None, :])
        mask_tile = tl.load(Mask_ptrs, mask=(q_mask[:, None] & kv_mask[None, :]), other=0).to(tl.float32)

        # dP = dP_dropped * mask * inv_keep_prob
        dP_tile = dPd_tile * mask_tile * inv_keep_prob

        # Load P tile [BLOCK_Q, BLOCK_KV]
        P_ptrs = P_base + (q_offs[:, None] * P_stride_q + kv_block[None, :])
        P_tile = tl.load(P_ptrs, mask=(q_mask[:, None] & kv_mask[None, :]), other=0.0).to(tl.float32)

        # Accumulate Di += sum_kv(dP * P)
        Di += tl.sum(dP_tile * P_tile, axis=1)

    # Second pass: compute dS and accumulate dV
    # dV accumulator [BLOCK_KV, head_dim]
    # We accumulate over all BLOCK_Q queries: dV += Pd^T @ dO
    # But we need to iterate kv in blocks — actually for dV we iterate over kv blocks
    # and accumulate. For each kv block, we do a full outer product with dO.

    for kv_blk in range(0, n_kv_blocks):
        kv_start = kv_blk * BLOCK_KV
        kv_block = kv_start + kv_offs
        kv_mask  = kv_block < skv

        # Load V tile [BLOCK_KV, head_dim]
        V_ptrs = V_base + (kv_block[:, None] * V_stride_k + d_offs[None, :])
        V_tile = tl.load(V_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

        # dP_dropped = dO @ V^T  -> [BLOCK_Q, BLOCK_KV]
        dPd_tile = tl.dot(dO_tile, tl.trans(V_tile))

        # Load dropout mask
        Mask_ptrs = Mask_base + (q_offs[:, None] * P_stride_q + kv_block[None, :])
        mask_tile = tl.load(Mask_ptrs, mask=(q_mask[:, None] & kv_mask[None, :]), other=0).to(tl.float32)

        # dP = dP_dropped * mask * inv_keep_prob
        dP_tile = dPd_tile * mask_tile * inv_keep_prob

        # Load P tile
        P_ptrs = P_base + (q_offs[:, None] * P_stride_q + kv_block[None, :])
        P_tile = tl.load(P_ptrs, mask=(q_mask[:, None] & kv_mask[None, :]), other=0.0).to(tl.float32)

        # dS = P * (dP - Di[:, None])
        dS_tile = P_tile * (dP_tile - Di[:, None])

        # Write dS tile
        dS_ptrs = dS_base + (q_offs[:, None] * P_stride_q + kv_block[None, :])
        tl.store(dS_ptrs, dS_tile.to(tl.bfloat16),
                 mask=(q_mask[:, None] & kv_mask[None, :]))

        # Load Pd tile [BLOCK_Q, BLOCK_KV]
        Pd_ptrs = Pd_base + (q_offs[:, None] * P_stride_q + kv_block[None, :])
        Pd_tile = tl.load(Pd_ptrs, mask=(q_mask[:, None] & kv_mask[None, :]), other=0.0).to(tl.float32)

        # dV += Pd^T @ dO  -> [BLOCK_KV, head_dim]
        dV_tile = tl.dot(tl.trans(Pd_tile), dO_tile)  # [BLOCK_KV, head_dim]

        # Atomic add to dV (multiple heads contribute to same kv-head)
        dV_ptrs = dV_base + (kv_block[:, None] * dV_stride_k + d_offs[None, :])
        tl.atomic_add(dV_ptrs, dV_tile.to(tl.float32),
                      mask=kv_mask[:, None])


def _attn_backward_triton(
    grad_attn_output,     # [bs, sq, 80, 128]  bf16
    attn_weights,         # [bs, 80, sq, skv]  bf16
    attn_weights_dropped, # [bs, 80, sq, skv]  bf16
    value_states,         # [bs,  8, skv, 128] bf16
    dropout_mask,         # [bs, 80, sq, skv]  bool
    attention_dropout,    # float
):
    bs    = grad_attn_output.shape[0]
    sq    = grad_attn_output.shape[1]
    skv   = value_states.shape[2]

    inv_keep_prob = 1.0 / (1.0 - attention_dropout)

    # Transpose dO to [bs, 80, sq, 128] — contiguous
    dO = grad_attn_output.transpose(1, 2).contiguous()

    # Allocate outputs
    dS  = torch.empty((bs, NUM_ATTENTION_HEADS, sq, skv),
                      dtype=torch.bfloat16, device=dO.device)
    # dV uses float32 for atomic accumulation, convert at end
    dV_f32 = torch.zeros((bs, NUM_KEY_VALUE_HEADS, skv, HEAD_DIM),
                         dtype=torch.float32, device=dO.device)

    # Make inputs contiguous
    attn_weights         = attn_weights.contiguous()
    attn_weights_dropped = attn_weights_dropped.contiguous()
    value_states         = value_states.contiguous()
    dropout_mask         = dropout_mask.contiguous()

    BLOCK_Q  = 32
    BLOCK_KV = 32

    grid = (bs * NUM_ATTENTION_HEADS, triton.cdiv(sq, BLOCK_Q))

    # Strides
    dO_s = dO.stride()             # (b, h, q, d)
    P_s  = attn_weights.stride()   # (b, h, q, k)
    V_s  = value_states.stride()   # (b, h, k, d)
    dV_s = dV_f32.stride()         # (b, h, k, d)

    _attn_bwd_kernel[grid](
        dO, attn_weights, attn_weights_dropped, value_states, dropout_mask,
        dS, dV_f32,
        bs, sq, skv,
        NUM_ATTENTION_HEADS, NUM_KEY_VALUE_HEADS, N_GROUPS, HEAD_DIM,
        inv_keep_prob,
        dO_s[0], dO_s[1], dO_s[2],
        P_s[0],  P_s[1],  P_s[2],
        V_s[0],  V_s[1],  V_s[2],
        dV_s[0], dV_s[1], dV_s[2],
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV,
        num_warps=4,
        num_stages=2,
    )

    dV = dV_f32.to(torch.bfloat16)
    return dS, dV


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    return _attn_backward_triton(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        attention_dropout,
    )
