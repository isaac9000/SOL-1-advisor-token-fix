"""
Triton-fused attention-backward kernel.

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


# ---------------------------------------------------------------------------
# Kernel 1: grad_attn_scores
#   For each (batch, head, sq_tile): load dO tile, load V tile (via GQA map),
#   compute dP = dO @ V^T, apply dropout backward, apply softmax backward.
#   Output: grad_attn_scores [bs, 80, sq, skv]
# ---------------------------------------------------------------------------
@triton.jit
def attn_bwd_dS_kernel(
    # inputs
    dO_ptr,       # [bs, 80, sq, 128]  bfloat16  (pre-transposed)
    P_ptr,        # [bs, 80, sq, skv]  bfloat16
    Pd_ptr,       # [bs, 80, sq, skv]  bfloat16  (dropped)
    V_ptr,        # [bs,  8, skv, 128] bfloat16
    mask_ptr,     # [bs, 80, sq, skv]  bool
    # outputs
    dS_ptr,       # [bs, 80, sq, skv]  bfloat16
    # strides for dO [bs, h, sq, d]
    dO_stride_b, dO_stride_h, dO_stride_sq, dO_stride_d,
    # strides for P/Pd/mask/dS [bs, h, sq, skv]
    P_stride_b, P_stride_h, P_stride_sq, P_stride_skv,
    # strides for V [bs, kv_h, skv, d]
    V_stride_b, V_stride_kvh, V_stride_skv, V_stride_d,
    # dims
    bs, n_heads: tl.constexpr, sq, skv,
    n_groups: tl.constexpr, head_dim: tl.constexpr,
    dropout_scale: tl.constexpr,
    # tile sizes
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Grid: (bs * n_heads, triton.cdiv(sq, BLOCK_SQ))
    pid_bh = tl.program_id(0)
    pid_sq = tl.program_id(1)

    b = pid_bh // n_heads
    h = pid_bh % n_heads
    kv_h = h // n_groups

    sq_start = pid_sq * BLOCK_SQ
    sq_offs = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask = sq_offs < sq

    d_offs = tl.arange(0, BLOCK_D)

    # Load dO tile: [BLOCK_SQ, BLOCK_D]
    dO_base = b * dO_stride_b + h * dO_stride_h
    dO_ptrs = dO_ptr + dO_base + sq_offs[:, None] * dO_stride_sq + d_offs[None, :] * dO_stride_d
    dO_tile = tl.load(dO_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)

    # Accumulate dP = dO @ V^T  over all skv blocks
    # We'll write dP to dS_ptr after softmax bwd
    # Strategy: iterate over BLOCK_SKV tiles of skv, compute dot, load P, mask, etc.

    P_base = b * P_stride_b + h * P_stride_h
    V_base = b * V_stride_b + kv_h * V_stride_kvh

    skv_tiles = tl.cdiv(skv, BLOCK_SKV)

    for skv_tile in range(skv_tiles):
        skv_start = skv_tile * BLOCK_SKV
        skv_offs = skv_start + tl.arange(0, BLOCK_SKV)
        skv_mask = skv_offs < skv

        # Load V tile: [BLOCK_SKV, BLOCK_D]
        V_ptrs = V_ptr + V_base + skv_offs[:, None] * V_stride_skv + d_offs[None, :] * V_stride_d
        V_tile = tl.load(V_ptrs, mask=skv_mask[:, None], other=0.0).to(tl.float32)

        # dP_tile = dO_tile @ V_tile^T  -> [BLOCK_SQ, BLOCK_SKV]
        dP_tile = tl.dot(dO_tile, tl.trans(V_tile))  # fp32

        # Load dropout mask and apply
        mask_ptrs = mask_ptr + P_base + sq_offs[:, None] * P_stride_sq + skv_offs[None, :] * P_stride_skv
        dmask = tl.load(mask_ptrs, mask=(sq_mask[:, None] & skv_mask[None, :]), other=0)
        dP_tile = tl.where(dmask, dP_tile * dropout_scale, 0.0)

        # Load P tile
        P_ptrs = P_ptr + P_base + sq_offs[:, None] * P_stride_sq + skv_offs[None, :] * P_stride_skv
        P_tile = tl.load(P_ptrs, mask=(sq_mask[:, None] & skv_mask[None, :]), other=0.0).to(tl.float32)

        # We need sum(dP * P) across skv for softmax bwd, but that requires all skv.
        # So we can't do it per tile without a two-pass approach.
        # Store dP to dS_ptr temporarily (we'll fix this with softmax bwd after).
        # Actually, we need a full row of skv to compute the softmax backward.
        # For large skv this is tricky in a single-pass tiled kernel.
        # Instead, store dP (pre-softmax) in dS temporarily and do softmax bwd in a 2nd pass.
        # But that wastes bandwidth. Let's instead store dP*P in a register accumulator.

        # Store dP tile directly (we'll overwrite with dS after computing row sum)
        dS_ptrs = dS_ptr + P_base + sq_offs[:, None] * P_stride_sq + skv_offs[None, :] * P_stride_skv
        tl.store(dS_ptrs, dP_tile.to(tl.bfloat16), mask=(sq_mask[:, None] & skv_mask[None, :]))

    # Now do softmax backward in a separate pass over skv
    # Pass 2: compute row sum of dP*P, then compute dS = P*(dP - sum)
    # Accumulate row_sum: [BLOCK_SQ]
    row_sum = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    for skv_tile in range(skv_tiles):
        skv_start = skv_tile * BLOCK_SKV
        skv_offs = skv_start + tl.arange(0, BLOCK_SKV)
        skv_mask = skv_offs < skv

        dS_ptrs = dS_ptr + P_base + sq_offs[:, None] * P_stride_sq + skv_offs[None, :] * P_stride_skv
        dP_tile = tl.load(dS_ptrs, mask=(sq_mask[:, None] & skv_mask[None, :]), other=0.0).to(tl.float32)

        P_ptrs = P_ptr + P_base + sq_offs[:, None] * P_stride_sq + skv_offs[None, :] * P_stride_skv
        P_tile = tl.load(P_ptrs, mask=(sq_mask[:, None] & skv_mask[None, :]), other=0.0).to(tl.float32)

        row_sum += tl.sum(dP_tile * P_tile, axis=1)

    # Pass 3: compute dS = P * (dP - row_sum) and store
    for skv_tile in range(skv_tiles):
        skv_start = skv_tile * BLOCK_SKV
        skv_offs = skv_start + tl.arange(0, BLOCK_SKV)
        skv_mask = skv_offs < skv

        dS_ptrs = dS_ptr + P_base + sq_offs[:, None] * P_stride_sq + skv_offs[None, :] * P_stride_skv
        dP_tile = tl.load(dS_ptrs, mask=(sq_mask[:, None] & skv_mask[None, :]), other=0.0).to(tl.float32)

        P_ptrs = P_ptr + P_base + sq_offs[:, None] * P_stride_sq + skv_offs[None, :] * P_stride_skv
        P_tile = tl.load(P_ptrs, mask=(sq_mask[:, None] & skv_mask[None, :]), other=0.0).to(tl.float32)

        dS_tile = P_tile * (dP_tile - row_sum[:, None])
        tl.store(dS_ptrs, dS_tile.to(tl.bfloat16), mask=(sq_mask[:, None] & skv_mask[None, :]))


# ---------------------------------------------------------------------------
# Kernel 2: grad_value_states
#   For each (batch, kv_head, skv_tile): loop over 10 query-heads,
#   accumulate dV += P̃^T @ dO.
#   Output: grad_value_states [bs, 8, skv, 128]
# ---------------------------------------------------------------------------
@triton.jit
def attn_bwd_dV_kernel(
    # inputs
    Pd_ptr,       # [bs, 80, sq, skv]  bfloat16
    dO_ptr,       # [bs, 80, sq, 128]  bfloat16  (pre-transposed)
    # outputs
    dV_ptr,       # [bs,  8, skv, 128] bfloat16
    # strides for Pd [bs, h, sq, skv]
    Pd_stride_b, Pd_stride_h, Pd_stride_sq, Pd_stride_skv,
    # strides for dO [bs, h, sq, d]
    dO_stride_b, dO_stride_h, dO_stride_sq, dO_stride_d,
    # strides for dV [bs, kv_h, skv, d]
    dV_stride_b, dV_stride_kvh, dV_stride_skv, dV_stride_d,
    # dims
    bs, n_heads: tl.constexpr, sq, skv,
    n_groups: tl.constexpr, head_dim: tl.constexpr,
    # tile sizes
    BLOCK_SKV: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Grid: (bs * n_kv_heads, triton.cdiv(skv, BLOCK_SKV))
    n_kv_heads = n_heads // n_groups
    pid_bkvh = tl.program_id(0)
    pid_skv = tl.program_id(1)

    b = pid_bkvh // n_kv_heads
    kv_h = pid_bkvh % n_kv_heads

    skv_start = pid_skv * BLOCK_SKV
    skv_offs = skv_start + tl.arange(0, BLOCK_SKV)
    skv_mask = skv_offs < skv

    d_offs = tl.arange(0, BLOCK_D)

    # Accumulator for dV: [BLOCK_SKV, BLOCK_D]
    dV_acc = tl.zeros([BLOCK_SKV, BLOCK_D], dtype=tl.float32)

    sq_tiles = tl.cdiv(sq, BLOCK_SQ)

    # Loop over all 10 query-heads in this kv-group
    for g in range(n_groups):
        h = kv_h * n_groups + g

        Pd_base = b * Pd_stride_b + h * Pd_stride_h
        dO_base = b * dO_stride_b + h * dO_stride_h

        # Loop over sq tiles
        for sq_tile in range(sq_tiles):
            sq_start = sq_tile * BLOCK_SQ
            sq_offs = sq_start + tl.arange(0, BLOCK_SQ)
            sq_mask = sq_offs < sq

            # Load Pd tile: [BLOCK_SQ, BLOCK_SKV]  (we need P^T @ dO -> [skv, d])
            # We want Pd^T[skv, sq] @ dO[sq, d] -> [skv, d]
            # Load Pd[sq, skv]: [BLOCK_SQ, BLOCK_SKV]
            Pd_ptrs = Pd_ptr + Pd_base + sq_offs[:, None] * Pd_stride_sq + skv_offs[None, :] * Pd_stride_skv
            Pd_tile = tl.load(Pd_ptrs, mask=(sq_mask[:, None] & skv_mask[None, :]), other=0.0).to(tl.float32)

            # Load dO tile: [BLOCK_SQ, BLOCK_D]
            dO_ptrs = dO_ptr + dO_base + sq_offs[:, None] * dO_stride_sq + d_offs[None, :] * dO_stride_d
            dO_tile = tl.load(dO_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)

            # dV_acc += Pd^T @ dO = [BLOCK_SKV, BLOCK_SQ] @ [BLOCK_SQ, BLOCK_D]
            dV_acc += tl.dot(tl.trans(Pd_tile), dO_tile)

    # Store dV
    dV_base = b * dV_stride_b + kv_h * dV_stride_kvh
    dV_ptrs = dV_ptr + dV_base + skv_offs[:, None] * dV_stride_skv + d_offs[None, :] * dV_stride_d
    tl.store(dV_ptrs, dV_acc.to(tl.bfloat16), mask=skv_mask[:, None])


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_heads    = NUM_ATTENTION_HEADS    # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS    # 8
    n_groups   = N_GROUPS               # 10
    head_dim   = HEAD_DIM               # 128

    # Pre-transpose grad_attn_output: [bs, sq, 80, 128] -> [bs, 80, sq, 128]
    dO = grad_attn_output.transpose(1, 2).contiguous()

    # Output tensors
    grad_attn_scores  = torch.empty(bs, n_heads,    seq_q,  seq_kv, dtype=torch.bfloat16, device=dO.device)
    grad_value_states = torch.empty(bs, n_kv_heads, seq_kv, head_dim, dtype=torch.bfloat16, device=dO.device)

    dropout_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Tile sizes
    BLOCK_SQ  = 32
    BLOCK_SKV = 64
    BLOCK_D   = 128  # head_dim is exactly 128

    # Kernel 1: grad_attn_scores
    grid_dS = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ))
    attn_bwd_dS_kernel[grid_dS](
        dO, attn_weights, attn_weights_dropped, value_states, dropout_mask,
        grad_attn_scores,
        # dO strides [bs, h, sq, d]
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        # P strides [bs, h, sq, skv]
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2), attn_weights.stride(3),
        # V strides [bs, kv_h, skv, d]
        value_states.stride(0), value_states.stride(1), value_states.stride(2), value_states.stride(3),
        # dims
        bs, n_heads, seq_q, seq_kv,
        n_groups, head_dim,
        dropout_scale,
        BLOCK_SQ, BLOCK_SKV, BLOCK_D,
    )

    # Kernel 2: grad_value_states
    BLOCK_SQ_V  = 32
    BLOCK_SKV_V = 64

    grid_dV = (bs * n_kv_heads, triton.cdiv(seq_kv, BLOCK_SKV_V))
    attn_bwd_dV_kernel[grid_dV](
        attn_weights_dropped, dO,
        grad_value_states,
        # Pd strides
        attn_weights_dropped.stride(0), attn_weights_dropped.stride(1),
        attn_weights_dropped.stride(2), attn_weights_dropped.stride(3),
        # dO strides
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        # dV strides
        grad_value_states.stride(0), grad_value_states.stride(1),
        grad_value_states.stride(2), grad_value_states.stride(3),
        # dims
        bs, n_heads, seq_q, seq_kv,
        n_groups, head_dim,
        BLOCK_SKV_V, BLOCK_SQ_V, BLOCK_D,
    )

    return grad_attn_scores, grad_value_states
