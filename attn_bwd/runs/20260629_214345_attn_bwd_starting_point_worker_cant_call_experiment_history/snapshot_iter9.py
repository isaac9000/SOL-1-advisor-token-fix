"""
Optimized attention-backward kernel — fully fused Flash-Attention-style tiling.

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
def _flash_attn_bwd_kernel(
    # Inputs
    dO_ptr,           # [bs, n_heads, seq_q, d]  bfloat16
    P_ptr,            # [bs, n_heads, seq_q, seq_kv]  bfloat16
    Pd_ptr,           # [bs, n_heads, seq_q, seq_kv]  bfloat16  (post-dropout)
    V_ptr,            # [bs, n_kv, seq_kv, d]  bfloat16
    mask_ptr,         # [bs, n_heads, seq_q, seq_kv]  bool
    # Outputs
    dS_ptr,           # [bs, n_heads, seq_q, seq_kv]  bfloat16
    dV_ptr,           # [bs, n_kv, seq_kv, d]  bfloat16
    # Scalars
    scale,            # 1.0 / (1.0 - dropout)
    # Dimensions
    bs: tl.constexpr,
    n_heads: tl.constexpr,   # 80
    n_kv: tl.constexpr,      # 8
    n_groups: tl.constexpr,  # 10
    seq_q: tl.constexpr,
    seq_kv: tl.constexpr,
    d: tl.constexpr,         # 128
    # Tile sizes
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Flash-Attention backward kernel.
    Grid: (bs * n_heads, cdiv(seq_q, BLOCK_Q))
    
    Each program handles:
    - One (batch, head) pair
    - One tile of seq_q rows
    - Loops over all seq_kv blocks
    
    For each seq_q tile:
      Pass 1: loop over seq_kv blocks to accumulate row_sum[BLOCK_Q]
      Pass 2: loop over seq_kv blocks again to write dS and accumulate dV
    
    dV uses atomic adds for GQA group reduction.
    """
    pid_bh = tl.program_id(0)  # batch * n_heads index
    pid_q  = tl.program_id(1)  # seq_q tile index

    b_idx  = pid_bh // n_heads
    h_idx  = pid_bh % n_heads
    kv_idx = h_idx // n_groups  # which KV head

    # Base offsets for this (batch, head) pair
    q_tile_start = pid_q * BLOCK_Q
    q_offs = q_tile_start + tl.arange(0, BLOCK_Q)
    q_mask = q_offs < seq_q

    d_offs = tl.arange(0, BLOCK_D)

    # Strides (all row-major, last dim is innermost)
    # dO: [bs, n_heads, seq_q, d]
    dO_base = b_idx * (n_heads * seq_q * d) + h_idx * (seq_q * d)
    # P, Pd, mask, dS: [bs, n_heads, seq_q, seq_kv]
    P_base  = b_idx * (n_heads * seq_q * seq_kv) + h_idx * (seq_q * seq_kv)
    # V, dV: [bs, n_kv, seq_kv, d]
    V_base  = b_idx * (n_kv * seq_kv * d) + kv_idx * (seq_kv * d)

    # Load dO tile: [BLOCK_Q, BLOCK_D]
    dO_ptrs = dO_base + q_offs[:, None] * d + d_offs[None, :]
    dO_tile = tl.load(dO_ptr + dO_ptrs,
                      mask=q_mask[:, None] & (d_offs[None, :] < d),
                      other=0.0).to(tl.float32)

    # ── Pass 1: compute row_sum[BLOCK_Q] ─────────────────────────────────────
    row_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)

    for kv_start in tl.range(0, seq_kv, BLOCK_KV):
        kv_offs = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offs < seq_kv

        # Load V tile: [BLOCK_KV, BLOCK_D]
        V_ptrs = V_base + kv_offs[:, None] * d + d_offs[None, :]
        V_tile = tl.load(V_ptr + V_ptrs,
                         mask=kv_mask[:, None] & (d_offs[None, :] < d),
                         other=0.0).to(tl.float32)

        # dP_tile = dO_tile @ V_tile^T  -> [BLOCK_Q, BLOCK_KV]
        dP_tile = tl.dot(dO_tile, tl.trans(V_tile))  # [BLOCK_Q, BLOCK_KV]

        # Load dropout mask: [BLOCK_Q, BLOCK_KV]
        mk_ptrs = P_base + q_offs[:, None] * seq_kv + kv_offs[None, :]
        m_tile = tl.load(mask_ptr + mk_ptrs,
                         mask=q_mask[:, None] & kv_mask[None, :],
                         other=0).to(tl.float32)

        # Apply dropout scale
        dP_tile = dP_tile * m_tile * scale

        # Load P tile: [BLOCK_Q, BLOCK_KV]
        P_tile = tl.load(P_ptr + mk_ptrs,
                         mask=q_mask[:, None] & kv_mask[None, :],
                         other=0.0).to(tl.float32)

        # Accumulate row_sum += sum_kv(dP * P)
        row_sum += tl.sum(dP_tile * P_tile, axis=1)

    # ── Pass 2: write dS and accumulate dV ───────────────────────────────────
    for kv_start in tl.range(0, seq_kv, BLOCK_KV):
        kv_offs = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offs < seq_kv

        # Load V tile again
        V_ptrs = V_base + kv_offs[:, None] * d + d_offs[None, :]
        V_tile = tl.load(V_ptr + V_ptrs,
                         mask=kv_mask[:, None] & (d_offs[None, :] < d),
                         other=0.0).to(tl.float32)

        # Recompute dP tile
        dP_tile = tl.dot(dO_tile, tl.trans(V_tile))  # [BLOCK_Q, BLOCK_KV]

        mk_ptrs = P_base + q_offs[:, None] * seq_kv + kv_offs[None, :]
        m_tile = tl.load(mask_ptr + mk_ptrs,
                         mask=q_mask[:, None] & kv_mask[None, :],
                         other=0).to(tl.float32)

        dP_tile = dP_tile * m_tile * scale

        P_tile = tl.load(P_ptr + mk_ptrs,
                         mask=q_mask[:, None] & kv_mask[None, :],
                         other=0.0).to(tl.float32)

        # dS = P * (dP - row_sum[:, None])
        dS_tile = P_tile * (dP_tile - row_sum[:, None])

        # Store dS
        tl.store(dS_ptr + mk_ptrs,
                 dS_tile.to(tl.bfloat16),
                 mask=q_mask[:, None] & kv_mask[None, :])

        # dV accumulation: Pd_tile^T @ dO_tile -> [BLOCK_KV, BLOCK_D]
        # Load Pd tile (post-dropout weights)
        Pd_tile = tl.load(Pd_ptr + mk_ptrs,
                          mask=q_mask[:, None] & kv_mask[None, :],
                          other=0.0).to(tl.float32)

        # dV_contrib = Pd_tile^T @ dO_tile  -> [BLOCK_KV, BLOCK_D]
        dV_contrib = tl.dot(tl.trans(Pd_tile), dO_tile)  # [BLOCK_KV, BLOCK_D]

        # Atomic add to dV (for GQA: multiple Q-heads map to same KV-head)
        dV_ptrs = V_base + kv_offs[:, None] * d + d_offs[None, :]
        tl.atomic_add(dV_ptr + dV_ptrs,
                      dV_contrib.to(tl.float32),
                      mask=kv_mask[:, None] & (d_offs[None, :] < d))


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

    # Make all inputs contiguous
    P  = attn_weights.contiguous()
    Pd = attn_weights_dropped.contiguous()
    V  = value_states.contiguous()
    mk = dropout_mask.contiguous()

    # Outputs
    dS = torch.empty_like(P)
    dV = torch.zeros(bs, n_kv, seq_kv, d, dtype=torch.float32, device=dO.device)

    scale = 1.0 / (1.0 - attention_dropout)

    # Tile sizes
    BLOCK_Q  = 16
    BLOCK_KV = 64
    BLOCK_D  = 128  # = HEAD_DIM, must be power-of-2 constexpr

    n_heads = NUM_ATTENTION_HEADS
    grid = (bs * n_heads, triton.cdiv(seq_q, BLOCK_Q))

    _flash_attn_bwd_kernel[grid](
        dO, P, Pd, V, mk,
        dS, dV,
        scale,
        bs=bs,
        n_heads=n_heads,
        n_kv=n_kv,
        n_groups=n_g,
        seq_q=seq_q,
        seq_kv=seq_kv,
        d=d,
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
        BLOCK_D=BLOCK_D,
    )

    dV_bf16 = dV.to(torch.bfloat16)
    return dS, dV_bf16
