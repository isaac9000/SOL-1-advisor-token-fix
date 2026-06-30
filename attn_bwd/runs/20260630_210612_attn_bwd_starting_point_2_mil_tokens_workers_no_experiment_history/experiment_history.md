# Experiment History

Tracks every kernel attempt, its code, hypothesis, and result.

---

## Experiment #1 — 2026-06-30 21:06:41 UTC ✅ KEEP

**Hypothesis:** Baseline 'starting_point' — initial benchmark

**Result:** 3430.17 μs

**Kernel code:**
```python
"""
Reference attention-backward kernel — pure PyTorch baseline.

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

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # Expand value_states for GQA: [bs, 8, skv, d] → [bs, 80, skv, d]
    vs_exp = value_states[:, :, None, :, :].expand(
        bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM
    ).reshape(bs, n_heads, seq_kv, HEAD_DIM)

    # 1. Transpose grad: [bs, sq, h, d] → [bs, h, sq, d]  (cast to f32)
    dO = grad_attn_output.transpose(1, 2).to(torch.float32)

    # 2. dP̃ = dO @ V^T  →  [bs, h, sq, skv]
    dP_dropped = torch.matmul(dO, vs_exp.to(torch.float32).transpose(-2, -1))

    # 3. Dropout backward
    if attention_dropout > 0.0:
        dP = dP_dropped * dropout_mask / (1.0 - attention_dropout)
    else:
        dP = dP_dropped

    # 4. Softmax backward: dS = P ⊙ (dP − sum(dP ⊙ P))
    P = attn_weights.to(torch.float32)
    dS = P * (dP - (dP * P).sum(dim=-1, keepdim=True))
    dS = dS.to(torch.bfloat16)

    # 5. dV_exp = P̃^T @ dO  →  [bs, h, skv, d]
    dV_exp = torch.matmul(
        attn_weights_dropped.to(torch.float32).transpose(-2, -1), dO
    )

    # 6. GQA aggregation: sum over groups  →  [bs, 8, skv, d]
    dV = dV_exp.reshape(bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM).sum(dim=2)
    dV = dV.to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #2 — 2026-06-30 21:08:54 UTC ✅ KEEP

**Hypothesis:** **

**Result:** 2064.97 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using Triton fused kernels.

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


@triton.jit
def fused_dV_kernel(
    # P_dropped: [bs, 80, sq, skv]
    P_dropped_ptr, stride_pd_bs, stride_pd_h, stride_pd_sq, stride_pd_skv,
    # dO: [bs, 80, sq, d]  (already transposed from [bs,sq,80,d])
    dO_ptr, stride_do_bs, stride_do_h, stride_do_sq, stride_do_d,
    # dV: [bs, 8, skv, d]
    dV_ptr, stride_dv_bs, stride_dv_kvh, stride_dv_skv, stride_dv_d,
    bs, n_kv_heads, n_groups, sq, skv, d,
    BLOCK_SKV: tl.constexpr, BLOCK_SQ: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    Compute dV = sum_{groups} P_dropped^T @ dO, with GQA group reduction fused.
    Grid: (bs * n_kv_heads, cdiv(skv, BLOCK_SKV), cdiv(d, BLOCK_D))
    """
    # Program IDs
    pid_bkv = tl.program_id(0)
    pid_skv = tl.program_id(1)
    pid_d   = tl.program_id(2)

    batch_idx  = pid_bkv // n_kv_heads
    kv_head    = pid_bkv % n_kv_heads

    # Tile offsets
    skv_start = pid_skv * BLOCK_SKV
    d_start   = pid_d   * BLOCK_D

    skv_offs = skv_start + tl.arange(0, BLOCK_SKV)
    d_offs   = d_start   + tl.arange(0, BLOCK_D)
    sq_offs  = tl.arange(0, BLOCK_SQ)

    skv_mask = skv_offs < skv
    d_mask   = d_offs   < d

    # Accumulator for dV tile [BLOCK_SKV, BLOCK_D]
    acc = tl.zeros((BLOCK_SKV, BLOCK_D), dtype=tl.float32)

    # Loop over all 10 query groups for this KV head
    for g in range(n_groups):
        q_head = kv_head * n_groups + g

        # Base pointers for this (batch, q_head)
        P_base = P_dropped_ptr + batch_idx * stride_pd_bs + q_head * stride_pd_h
        dO_base = dO_ptr + batch_idx * stride_do_bs + q_head * stride_do_h

        # Loop over sq dimension in tiles
        num_sq_blocks = tl.cdiv(sq, BLOCK_SQ)
        for sq_tile in range(num_sq_blocks):
            sq_start = sq_tile * BLOCK_SQ
            sq_tile_offs = sq_start + sq_offs
            sq_mask = sq_tile_offs < sq

            # Load P_dropped tile: [BLOCK_SKV, BLOCK_SQ] — we want [skv, sq]
            # P_dropped is [bs, h, sq, skv], so P^T[skv, sq]
            # We need P_dropped[sq, skv] transposed to [skv, sq]
            p_ptrs = P_base + sq_tile_offs[:, None] * stride_pd_sq + skv_offs[None, :] * stride_pd_skv
            # p_tile shape: [BLOCK_SQ, BLOCK_SKV]
            p_tile = tl.load(p_ptrs, mask=sq_mask[:, None] & skv_mask[None, :], other=0.0)
            p_tile = p_tile.to(tl.float32)

            # Load dO tile: [BLOCK_SQ, BLOCK_D]
            do_ptrs = dO_base + sq_tile_offs[:, None] * stride_do_sq + d_offs[None, :] * stride_do_d
            do_tile = tl.load(do_ptrs, mask=sq_mask[:, None] & d_mask[None, :], other=0.0)
            do_tile = do_tile.to(tl.float32)

            # acc += P^T @ dO = [BLOCK_SKV, BLOCK_SQ] @ [BLOCK_SQ, BLOCK_D]
            # p_tile is [BLOCK_SQ, BLOCK_SKV], so we need p_tile^T = [BLOCK_SKV, BLOCK_SQ]
            acc += tl.dot(tl.trans(p_tile), do_tile)

    # Write result
    dV_base = dV_ptr + batch_idx * stride_dv_bs + kv_head * stride_dv_kvh
    out_ptrs = dV_base + skv_offs[:, None] * stride_dv_skv + d_offs[None, :] * stride_dv_d
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=skv_mask[:, None] & d_mask[None, :])


@triton.jit
def fused_dS_kernel(
    # dO: [bs, 80, sq, d]
    dO_ptr, stride_do_bs, stride_do_h, stride_do_sq, stride_do_d,
    # V: [bs, 8, skv, d]  (original, before GQA expand)
    V_ptr, stride_v_bs, stride_v_kvh, stride_v_skv, stride_v_d,
    # P (attn_weights): [bs, 80, sq, skv]
    P_ptr, stride_p_bs, stride_p_h, stride_p_sq, stride_p_skv,
    # dropout_mask: [bs, 80, sq, skv]
    mask_ptr, stride_m_bs, stride_m_h, stride_m_sq, stride_m_skv,
    # dS output: [bs, 80, sq, skv]
    dS_ptr, stride_ds_bs, stride_ds_h, stride_ds_sq, stride_ds_skv,
    bs, n_heads, n_groups, sq, skv, d,
    inv_keep_prob: tl.constexpr,
    BLOCK_SQ: tl.constexpr, BLOCK_SKV: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    Fused: dP = dO @ V^T, apply dropout mask, compute softmax backward dS.
    Grid: (bs * n_heads, cdiv(sq, BLOCK_SQ))
    """
    pid_bh = tl.program_id(0)
    pid_sq = tl.program_id(1)

    batch_idx = pid_bh // n_heads
    head      = pid_bh % n_heads
    kv_head   = head // n_groups

    sq_start = pid_sq * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < sq

    d_offs   = tl.arange(0, BLOCK_D)
    skv_offs = tl.arange(0, BLOCK_SKV)

    dO_base = dO_ptr  + batch_idx * stride_do_bs + head    * stride_do_h
    V_base  = V_ptr   + batch_idx * stride_v_bs  + kv_head * stride_v_kvh
    P_base  = P_ptr   + batch_idx * stride_p_bs  + head    * stride_p_h
    M_base  = mask_ptr + batch_idx * stride_m_bs + head    * stride_m_h
    dS_base = dS_ptr  + batch_idx * stride_ds_bs + head    * stride_ds_h

    # We compute dP = dO @ V^T for this (batch, head) for sq_offs rows x all skv cols
    # Result shape: [BLOCK_SQ, skv] — we tile over skv

    num_skv_blocks = tl.cdiv(skv, BLOCK_SKV)

    # We need to compute rowsum(dP * P) across all skv for softmax bwd
    # Strategy: first pass computes dP tiles and rowsum, second pass writes dS
    # But since skv can be large, do a single pass storing dP and accumulating rowsum

    # Accumulate rowsum of dP*P: [BLOCK_SQ]
    rowsum = tl.zeros((BLOCK_SQ,), dtype=tl.float32)

    # We'll need to store dP temporarily — use a loop with two passes
    # Pass 1: compute dP tiles, accumulate rowsum
    # Pass 2: compute dS and write
    # Since we can't easily store large dP in registers, do two loops over skv

    # ----- Pass 1: compute rowsum(dP * P) -----
    for skv_tile in range(num_skv_blocks):
        skv_start = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask = skv_tile_offs < skv

        # Load dO: [BLOCK_SQ, BLOCK_D]
        do_ptrs = dO_base + sq_offs[:, None] * stride_do_sq + d_offs[None, :] * stride_do_d
        do_tile = tl.load(do_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)

        # Load V: [BLOCK_SKV, BLOCK_D]
        v_ptrs = V_base + skv_tile_offs[:, None] * stride_v_skv + d_offs[None, :] * stride_v_d
        v_tile = tl.load(v_ptrs, mask=skv_mask[:, None], other=0.0).to(tl.float32)

        # dP_tile = dO @ V^T: [BLOCK_SQ, BLOCK_SKV]
        dP_tile = tl.dot(do_tile, tl.trans(v_tile))

        # Load dropout mask and apply
        m_ptrs = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile = tl.load(m_ptrs, mask=sq_mask[:, None] & skv_mask[None, :], other=0)
        dP_tile = dP_tile * m_tile.to(tl.float32) * inv_keep_prob

        # Load P: [BLOCK_SQ, BLOCK_SKV]
        p_ptrs = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile = tl.load(p_ptrs, mask=sq_mask[:, None] & skv_mask[None, :], other=0.0).to(tl.float32)

        # Accumulate rowsum(dP * P)
        rowsum += tl.sum(dP_tile * p_tile, axis=1)

    # ----- Pass 2: compute dS and store -----
    for skv_tile in range(num_skv_blocks):
        skv_start = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask = skv_tile_offs < skv

        # Recompute dP_tile
        do_ptrs = dO_base + sq_offs[:, None] * stride_do_sq + d_offs[None, :] * stride_do_d
        do_tile = tl.load(do_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)

        v_ptrs = V_base + skv_tile_offs[:, None] * stride_v_skv + d_offs[None, :] * stride_v_d
        v_tile = tl.load(v_ptrs, mask=skv_mask[:, None], other=0.0).to(tl.float32)

        dP_tile = tl.dot(do_tile, tl.trans(v_tile))

        m_ptrs = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile = tl.load(m_ptrs, mask=sq_mask[:, None] & skv_mask[None, :], other=0)
        dP_tile = dP_tile * m_tile.to(tl.float32) * inv_keep_prob

        p_ptrs = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile = tl.load(p_ptrs, mask=sq_mask[:, None] & skv_mask[None, :], other=0.0).to(tl.float32)

        # dS = P * (dP - rowsum)
        dS_tile = p_tile * (dP_tile - rowsum[:, None])

        # Store
        ds_ptrs = dS_base + sq_offs[:, None] * stride_ds_sq + skv_tile_offs[None, :] * stride_ds_skv
        tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=sq_mask[:, None] & skv_mask[None, :])


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

    # Transpose dO: [bs, sq, 80, d] -> [bs, 80, sq, d], keep bfloat16
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()  # [bs, 80, sq, d]

    # ---- Compute dV using fused kernel ----
    dV = torch.empty((bs, n_kv_heads, seq_kv, d), dtype=torch.bfloat16, device=dO.device)

    # Make P_dropped contiguous
    P_dropped = attn_weights_dropped.contiguous()  # [bs, 80, sq, skv]

    BLOCK_SKV = 64
    BLOCK_SQ  = 32
    BLOCK_D   = 128  # d=128, fits in one block

    grid_dV = (bs * n_kv_heads,
               triton.cdiv(seq_kv, BLOCK_SKV),
               triton.cdiv(d, BLOCK_D))

    fused_dV_kernel[grid_dV](
        P_dropped, P_dropped.stride(0), P_dropped.stride(1), P_dropped.stride(2), P_dropped.stride(3),
        dO, dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dV, dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        bs, n_kv_heads, n_groups, seq_q, seq_kv, d,
        BLOCK_SKV=BLOCK_SKV, BLOCK_SQ=BLOCK_SQ, BLOCK_D=BLOCK_D,
    )

    # ---- Compute dS using fused kernel ----
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    P_attn = attn_weights.contiguous()      # [bs, 80, sq, skv]
    dmask  = dropout_mask.contiguous()      # [bs, 80, sq, skv]
    V_cont = value_states.contiguous()      # [bs, 8, skv, d]

    inv_keep = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    BLOCK_SQ_DS  = 16
    BLOCK_SKV_DS = 64
    BLOCK_D_DS   = 128

    grid_dS = (bs * n_heads,
               triton.cdiv(seq_q, BLOCK_SQ_DS))

    fused_dS_kernel[grid_dS](
        dO, dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        V_cont, V_cont.stride(0), V_cont.stride(1), V_cont.stride(2), V_cont.stride(3),
        P_attn, P_attn.stride(0), P_attn.stride(1), P_attn.stride(2), P_attn.stride(3),
        dmask, dmask.stride(0), dmask.stride(1), dmask.stride(2), dmask.stride(3),
        dS, dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
        bs, n_heads, n_groups, seq_q, seq_kv, d,
        inv_keep_prob=inv_keep,
        BLOCK_SQ=BLOCK_SQ_DS, BLOCK_SKV=BLOCK_SKV_DS, BLOCK_D=BLOCK_D_DS,
    )

    return dS, dV

```

---

## Experiment #3 — 2026-06-30 21:11:29 UTC ✅ KEEP

**Hypothesis:** **

**Result:** 2007.59 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using hybrid cuBLAS + Triton approach.

Strategy:
  1. Use torch.matmul (cuBLAS) for the two large GEMMs:
     - dP_raw = dO @ V_expanded^T   [bs, 80, sq, skv]
     - dV_raw = P_dropped^T @ dO    [bs, 80, skv, d]
  2. Use a lightweight Triton kernel for softmax backward:
     - fuse dropout application + rowsum + dS = P*(dP - rowsum)
     - two passes over skv (pass1: rowsum, pass2: store) — minimal memory
  3. Use a simple Triton kernel for GQA dV reduction:
     - sum dV_raw over the 10 groups to get [bs, 8, skv, d]

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


@triton.jit
def softmax_bwd_kernel(
    # dP_raw: [bs, 80, sq, skv] float32
    dP_ptr, stride_dp_bs, stride_dp_h, stride_dp_sq, stride_dp_skv,
    # P (attn_weights): [bs, 80, sq, skv] bfloat16
    P_ptr, stride_p_bs, stride_p_h, stride_p_sq, stride_p_skv,
    # dropout_mask: [bs, 80, sq, skv] bool
    mask_ptr, stride_m_bs, stride_m_h, stride_m_sq, stride_m_skv,
    # dS output: [bs, 80, sq, skv] bfloat16
    dS_ptr, stride_ds_bs, stride_ds_h, stride_ds_sq, stride_ds_skv,
    bs, n_heads, sq, skv,
    inv_keep_prob: tl.constexpr,
    BLOCK_SQ: tl.constexpr, BLOCK_SKV: tl.constexpr,
):
    """
    Single-pass softmax backward with dropout application.
    Grid: (bs * n_heads, cdiv(sq, BLOCK_SQ))
    For each sq tile, iterate over all skv tiles twice:
      Pass 1: accumulate rowsum(dP_masked * P)
      Pass 2: compute and store dS = P * (dP_masked - rowsum)
    The dP values come already computed (from cuBLAS matmul).
    """
    pid_bh = tl.program_id(0)
    pid_sq = tl.program_id(1)

    batch_idx = pid_bh // n_heads
    head      = pid_bh % n_heads

    sq_start = pid_sq * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < sq

    skv_offs = tl.arange(0, BLOCK_SKV)

    dP_base = dP_ptr  + batch_idx * stride_dp_bs + head * stride_dp_h
    P_base  = P_ptr   + batch_idx * stride_p_bs  + head * stride_p_h
    M_base  = mask_ptr + batch_idx * stride_m_bs + head * stride_m_h
    dS_base = dS_ptr  + batch_idx * stride_ds_bs + head * stride_ds_h

    num_skv_blocks = tl.cdiv(skv, BLOCK_SKV)

    # ----- Pass 1: compute rowsum(dP_masked * P) -----
    rowsum = tl.zeros((BLOCK_SQ,), dtype=tl.float32)

    for skv_tile in range(num_skv_blocks):
        skv_start = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask = skv_tile_offs < skv

        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        # Load dP_raw tile: [BLOCK_SQ, BLOCK_SKV]
        dp_ptrs = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
        dp_tile = tl.load(dp_ptrs, mask=combined_mask, other=0.0)  # float32

        # Load dropout mask and apply
        m_ptrs = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile = tl.load(m_ptrs, mask=combined_mask, other=0)
        dp_masked = dp_tile * m_tile.to(tl.float32) * inv_keep_prob

        # Load P: [BLOCK_SQ, BLOCK_SKV]
        p_ptrs = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # Accumulate rowsum(dP_masked * P)
        rowsum += tl.sum(dp_masked * p_tile, axis=1)

    # ----- Pass 2: compute dS and store -----
    for skv_tile in range(num_skv_blocks):
        skv_start = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask = skv_tile_offs < skv

        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dp_ptrs = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
        dp_tile = tl.load(dp_ptrs, mask=combined_mask, other=0.0)

        m_ptrs = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile = tl.load(m_ptrs, mask=combined_mask, other=0)
        dp_masked = dp_tile * m_tile.to(tl.float32) * inv_keep_prob

        p_ptrs = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # dS = P * (dP_masked - rowsum)
        dS_tile = p_tile * (dp_masked - rowsum[:, None])

        ds_ptrs = dS_base + sq_offs[:, None] * stride_ds_sq + skv_tile_offs[None, :] * stride_ds_skv
        tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


@triton.jit
def gqa_dV_reduce_kernel(
    # dV_raw: [bs, 80, skv, d] float32
    dV_raw_ptr, stride_dvr_bs, stride_dvr_h, stride_dvr_skv, stride_dvr_d,
    # dV out: [bs, 8, skv, d] bfloat16
    dV_ptr, stride_dv_bs, stride_dv_kvh, stride_dv_skv, stride_dv_d,
    bs, n_kv_heads, n_groups, skv, d,
    BLOCK_SKV: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    Reduce dV_raw [bs, 80, skv, d] over groups to [bs, 8, skv, d].
    Grid: (bs * n_kv_heads, cdiv(skv, BLOCK_SKV), cdiv(d, BLOCK_D))
    """
    pid_bkv = tl.program_id(0)
    pid_skv = tl.program_id(1)
    pid_d   = tl.program_id(2)

    batch_idx = pid_bkv // n_kv_heads
    kv_head   = pid_bkv % n_kv_heads

    skv_start = pid_skv * BLOCK_SKV
    d_start   = pid_d   * BLOCK_D

    skv_offs = skv_start + tl.arange(0, BLOCK_SKV)
    d_offs   = d_start   + tl.arange(0, BLOCK_D)

    skv_mask = skv_offs < skv
    d_mask   = d_offs   < d
    combined_mask = skv_mask[:, None] & d_mask[None, :]

    acc = tl.zeros((BLOCK_SKV, BLOCK_D), dtype=tl.float32)

    for g in range(n_groups):
        q_head = kv_head * n_groups + g
        base = dV_raw_ptr + batch_idx * stride_dvr_bs + q_head * stride_dvr_h
        ptrs = base + skv_offs[:, None] * stride_dvr_skv + d_offs[None, :] * stride_dvr_d
        tile = tl.load(ptrs, mask=combined_mask, other=0.0)  # float32
        acc += tile

    dV_base = dV_ptr + batch_idx * stride_dv_bs + kv_head * stride_dv_kvh
    out_ptrs = dV_base + skv_offs[:, None] * stride_dv_skv + d_offs[None, :] * stride_dv_d
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=combined_mask)


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

    # Transpose dO: [bs, sq, 80, d] -> [bs, 80, sq, d], contiguous
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()  # [bs, 80, sq, d] bfloat16

    # ---- GEMM 1: dP_raw = dO @ V_expanded^T ----
    # V: [bs, 8, skv, d] -> expand to [bs, 80, skv, d]
    V_exp = value_states.reshape(bs, n_kv_heads, 1, seq_kv, d).expand(
        bs, n_kv_heads, n_groups, seq_kv, d
    ).reshape(bs, n_heads, seq_kv, d).contiguous()  # [bs, 80, skv, d]

    # dO: [bs, 80, sq, d], V_exp: [bs, 80, skv, d]
    # dP_raw = dO @ V_exp^T => [bs, 80, sq, skv]
    dO_f32 = dO.float()
    V_exp_f32 = V_exp.float()
    dP_raw = torch.matmul(dO_f32, V_exp_f32.transpose(-2, -1))  # [bs, 80, sq, skv] float32

    # ---- GEMM 2: dV_raw = P_dropped^T @ dO ----
    # P_dropped: [bs, 80, sq, skv], dO: [bs, 80, sq, d]
    # dV_raw = P_dropped^T @ dO => [bs, 80, skv, d]
    P_dropped_f32 = attn_weights_dropped.float()
    dV_raw = torch.matmul(P_dropped_f32.transpose(-2, -1), dO_f32)  # [bs, 80, skv, d] float32

    # ---- Triton kernel 1: softmax backward ----
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    P_attn = attn_weights.contiguous()   # [bs, 80, sq, skv] bfloat16
    dmask  = dropout_mask.contiguous()   # [bs, 80, sq, skv] bool

    inv_keep = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    BLOCK_SQ_DS  = 16
    BLOCK_SKV_DS = 64

    grid_dS = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ_DS))

    softmax_bwd_kernel[grid_dS](
        dP_raw, dP_raw.stride(0), dP_raw.stride(1), dP_raw.stride(2), dP_raw.stride(3),
        P_attn, P_attn.stride(0), P_attn.stride(1), P_attn.stride(2), P_attn.stride(3),
        dmask, dmask.stride(0), dmask.stride(1), dmask.stride(2), dmask.stride(3),
        dS, dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
        bs, n_heads, seq_q, seq_kv,
        inv_keep_prob=inv_keep,
        BLOCK_SQ=BLOCK_SQ_DS, BLOCK_SKV=BLOCK_SKV_DS,
    )

    # ---- Triton kernel 2: GQA dV reduction ----
    dV = torch.empty((bs, n_kv_heads, seq_kv, d), dtype=torch.bfloat16, device=dO.device)

    BLOCK_SKV = 64
    BLOCK_D   = 128

    grid_dV = (bs * n_kv_heads,
               triton.cdiv(seq_kv, BLOCK_SKV),
               triton.cdiv(d, BLOCK_D))

    gqa_dV_reduce_kernel[grid_dV](
        dV_raw, dV_raw.stride(0), dV_raw.stride(1), dV_raw.stride(2), dV_raw.stride(3),
        dV, dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        bs, n_kv_heads, n_groups, seq_kv, d,
        BLOCK_SKV=BLOCK_SKV, BLOCK_D=BLOCK_D,
    )

    return dS, dV

```

---

## Experiment #4 — 2026-06-30 21:13:09 UTC ✅ KEEP

**Hypothesis:** **

**Result:** 615.79 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using BF16 GEMMs + Triton softmax backward.

Strategy:
  1. Avoid materializing V_exp [bs, 80, skv, d] entirely.
     Instead reshape dO to [bs*8, 10, sq, d] and use expand+contiguous on V
     to [bs*8, 10, skv, d], then flatten to [bs*80, sq/skv, d] for bmm in BF16.

  2. For dP: bmm(dO_bf16, V^T_bf16) in BF16 — maximize B200 tensor core throughput.
     For dV: bmm(P_dropped^T_bf16, dO_bf16) in BF16.

  3. Triton softmax-backward kernel: reads BF16 dP, applies dropout + rowsum
     in float32 internally, stores BF16 dS.

  4. dV: bmm result is [bs*80, skv, d], reshape to [bs, 8, 10, skv, d] and sum dim=2.

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


@triton.jit
def softmax_bwd_kernel(
    # dP: [bs, 80, sq, skv] bfloat16
    dP_ptr, stride_dp_bs, stride_dp_h, stride_dp_sq, stride_dp_skv,
    # P (attn_weights): [bs, 80, sq, skv] bfloat16
    P_ptr, stride_p_bs, stride_p_h, stride_p_sq, stride_p_skv,
    # dropout_mask: [bs, 80, sq, skv] bool
    mask_ptr, stride_m_bs, stride_m_h, stride_m_sq, stride_m_skv,
    # dS output: [bs, 80, sq, skv] bfloat16
    dS_ptr, stride_ds_bs, stride_ds_h, stride_ds_sq, stride_ds_skv,
    bs, n_heads, sq, skv,
    inv_keep_prob: tl.constexpr,
    BLOCK_SQ: tl.constexpr, BLOCK_SKV: tl.constexpr,
):
    """
    Single-pass softmax backward with dropout application.
    Grid: (bs * n_heads, cdiv(sq, BLOCK_SQ))
    For each sq tile, iterate over all skv tiles twice:
      Pass 1: accumulate rowsum(dP_masked * P)
      Pass 2: compute and store dS = P * (dP_masked - rowsum)
    dP is now BF16 (from BF16 bmm).
    """
    pid_bh = tl.program_id(0)
    pid_sq = tl.program_id(1)

    batch_idx = pid_bh // n_heads
    head      = pid_bh % n_heads

    sq_start = pid_sq * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < sq

    skv_offs = tl.arange(0, BLOCK_SKV)

    dP_base = dP_ptr  + batch_idx * stride_dp_bs + head * stride_dp_h
    P_base  = P_ptr   + batch_idx * stride_p_bs  + head * stride_p_h
    M_base  = mask_ptr + batch_idx * stride_m_bs + head * stride_m_h
    dS_base = dS_ptr  + batch_idx * stride_ds_bs + head * stride_ds_h

    num_skv_blocks = tl.cdiv(skv, BLOCK_SKV)

    # ----- Pass 1: compute rowsum(dP_masked * P) -----
    rowsum = tl.zeros((BLOCK_SQ,), dtype=tl.float32)

    for skv_tile in range(num_skv_blocks):
        skv_start = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask = skv_tile_offs < skv

        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        # Load dP tile (BF16) and cast to float32
        dp_ptrs = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
        dp_tile = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)  # bf16 -> f32

        # Load dropout mask and apply
        m_ptrs = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile = tl.load(m_ptrs, mask=combined_mask, other=0)
        dp_masked = dp_tile * m_tile.to(tl.float32) * inv_keep_prob

        # Load P: [BLOCK_SQ, BLOCK_SKV]
        p_ptrs = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # Accumulate rowsum(dP_masked * P)
        rowsum += tl.sum(dp_masked * p_tile, axis=1)

    # ----- Pass 2: compute dS and store -----
    for skv_tile in range(num_skv_blocks):
        skv_start = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask = skv_tile_offs < skv

        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dp_ptrs = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
        dp_tile = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        m_ptrs = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile = tl.load(m_ptrs, mask=combined_mask, other=0)
        dp_masked = dp_tile * m_tile.to(tl.float32) * inv_keep_prob

        p_ptrs = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # dS = P * (dP_masked - rowsum)
        dS_tile = p_tile * (dp_masked - rowsum[:, None])

        ds_ptrs = dS_base + sq_offs[:, None] * stride_ds_sq + skv_tile_offs[None, :] * stride_ds_skv
        tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


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

    # Transpose dO: [bs, sq, 80, d] -> [bs, 80, sq, d], contiguous, keep BF16
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()  # [bs, 80, sq, d] bfloat16

    # ---- GEMM 1 (BF16): dP = dO @ V^T, avoiding V_exp materialization ----
    # Reshape dO: [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8*10, sq, d]
    dO_grouped = dO.view(bs, n_kv_heads, n_groups, seq_q, d)
    dO_bmm = dO_grouped.reshape(bs * n_kv_heads * n_groups, seq_q, d)  # [bs*80, sq, d]

    # V: [bs, 8, skv, d] -> expand to [bs, 8, 10, skv, d] (no copy) -> [bs*80, skv, d]
    # Use .expand() then .reshape() — expand is zero-copy, reshape after contiguous()
    V_grouped = value_states.unsqueeze(2).expand(bs, n_kv_heads, n_groups, seq_kv, d)
    V_bmm = V_grouped.reshape(bs * n_kv_heads * n_groups, seq_kv, d)  # triggers contiguous copy

    # BF16 bmm: dP = dO_bmm @ V_bmm^T => [bs*80, sq, skv]
    dP_raw = torch.bmm(dO_bmm, V_bmm.transpose(-2, -1))  # BF16, [bs*80, sq, skv]
    dP_raw = dP_raw.view(bs, n_heads, seq_q, seq_kv)  # [bs, 80, sq, skv] BF16

    # ---- GEMM 2 (BF16): dV = P_dropped^T @ dO ----
    # P_dropped: [bs, 80, sq, skv] -> [bs*80, sq, skv]
    P_dropped_bmm = attn_weights_dropped.reshape(bs * n_heads, seq_q, seq_kv)  # [bs*80, sq, skv]
    dO_bmm2 = dO.reshape(bs * n_heads, seq_q, d)  # [bs*80, sq, d]

    # BF16 bmm: dV_raw = P_dropped^T @ dO => [bs*80, skv, d]
    dV_raw = torch.bmm(P_dropped_bmm.transpose(-2, -1), dO_bmm2)  # BF16, [bs*80, skv, d]

    # Reshape and sum over groups: [bs*80, skv, d] -> [bs, 8, 10, skv, d] -> sum -> [bs, 8, skv, d]
    dV = dV_raw.view(bs, n_kv_heads, n_groups, seq_kv, d).sum(dim=2).to(torch.bfloat16)

    # ---- Triton kernel: softmax backward (reads BF16 dP_raw) ----
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    P_attn = attn_weights.contiguous()   # [bs, 80, sq, skv] bfloat16
    dmask  = dropout_mask.contiguous()   # [bs, 80, sq, skv] bool

    inv_keep = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    BLOCK_SQ_DS  = 16
    BLOCK_SKV_DS = 64

    grid_dS = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ_DS))

    softmax_bwd_kernel[grid_dS](
        dP_raw, dP_raw.stride(0), dP_raw.stride(1), dP_raw.stride(2), dP_raw.stride(3),
        P_attn, P_attn.stride(0), P_attn.stride(1), P_attn.stride(2), P_attn.stride(3),
        dmask, dmask.stride(0), dmask.stride(1), dmask.stride(2), dmask.stride(3),
        dS, dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
        bs, n_heads, seq_q, seq_kv,
        inv_keep_prob=inv_keep,
        BLOCK_SQ=BLOCK_SQ_DS, BLOCK_SKV=BLOCK_SKV_DS,
    )

    return dS, dV

```

---

## Experiment #5 — 2026-06-30 21:14:51 UTC ✅ KEEP

**Hypothesis:** ** Changed both GEMMs to use GQA-aware grouping along the sequence dimension instead of expanding V. dO is reshaped from `[bs, 80, sq, d]` → `[bs*8, 10*sq, d]`, V is used directly as `[bs*8, skv, d]` 

**Result:** 487.13 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using BF16 GEMMs + Triton softmax backward.

Strategy:
  1. Avoid materializing V_exp [bs, 80, skv, d] entirely.
     For dP: reshape dO from [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
             use V as [bs*8, skv, d], compute bmm -> [bs*8, 10*sq, skv]
             reshape to [bs, 80, sq, skv]. No V copy needed!

  2. For dV: reshape P_dropped from [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
             reshape dO from [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
             compute [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
             This directly gives dV summed over groups -- no separate reduction!

  3. Triton softmax-backward kernel: reads BF16 dP, applies dropout + rowsum
     in float32 internally, stores BF16 dS.

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


@triton.jit
def softmax_bwd_kernel(
    # dP: [bs, 80, sq, skv] bfloat16
    dP_ptr, stride_dp_bs, stride_dp_h, stride_dp_sq, stride_dp_skv,
    # P (attn_weights): [bs, 80, sq, skv] bfloat16
    P_ptr, stride_p_bs, stride_p_h, stride_p_sq, stride_p_skv,
    # dropout_mask: [bs, 80, sq, skv] bool
    mask_ptr, stride_m_bs, stride_m_h, stride_m_sq, stride_m_skv,
    # dS output: [bs, 80, sq, skv] bfloat16
    dS_ptr, stride_ds_bs, stride_ds_h, stride_ds_sq, stride_ds_skv,
    bs, n_heads, sq, skv,
    inv_keep_prob: tl.constexpr,
    BLOCK_SQ: tl.constexpr, BLOCK_SKV: tl.constexpr,
):
    """
    Single-pass softmax backward with dropout application.
    Grid: (bs * n_heads, cdiv(sq, BLOCK_SQ))
    For each sq tile, iterate over all skv tiles twice:
      Pass 1: accumulate rowsum(dP_masked * P)
      Pass 2: compute and store dS = P * (dP_masked - rowsum)
    dP is BF16 (from BF16 bmm).
    """
    pid_bh = tl.program_id(0)
    pid_sq = tl.program_id(1)

    batch_idx = pid_bh // n_heads
    head      = pid_bh % n_heads

    sq_start = pid_sq * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < sq

    skv_offs = tl.arange(0, BLOCK_SKV)

    dP_base = dP_ptr  + batch_idx * stride_dp_bs + head * stride_dp_h
    P_base  = P_ptr   + batch_idx * stride_p_bs  + head * stride_p_h
    M_base  = mask_ptr + batch_idx * stride_m_bs + head * stride_m_h
    dS_base = dS_ptr  + batch_idx * stride_ds_bs + head * stride_ds_h

    num_skv_blocks = tl.cdiv(skv, BLOCK_SKV)

    # ----- Pass 1: compute rowsum(dP_masked * P) -----
    rowsum = tl.zeros((BLOCK_SQ,), dtype=tl.float32)

    for skv_tile in range(num_skv_blocks):
        skv_start = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask = skv_tile_offs < skv

        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        # Load dP tile (BF16) and cast to float32
        dp_ptrs = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
        dp_tile = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # Load dropout mask and apply
        m_ptrs = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile = tl.load(m_ptrs, mask=combined_mask, other=0)
        dp_masked = dp_tile * m_tile.to(tl.float32) * inv_keep_prob

        # Load P: [BLOCK_SQ, BLOCK_SKV]
        p_ptrs = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # Accumulate rowsum(dP_masked * P)
        rowsum += tl.sum(dp_masked * p_tile, axis=1)

    # ----- Pass 2: compute dS and store -----
    for skv_tile in range(num_skv_blocks):
        skv_start = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask = skv_tile_offs < skv

        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dp_ptrs = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
        dp_tile = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        m_ptrs = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile = tl.load(m_ptrs, mask=combined_mask, other=0)
        dp_masked = dp_tile * m_tile.to(tl.float32) * inv_keep_prob

        p_ptrs = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # dS = P * (dP_masked - rowsum)
        dS_tile = p_tile * (dp_masked - rowsum[:, None])

        ds_ptrs = dS_base + sq_offs[:, None] * stride_ds_sq + skv_tile_offs[None, :] * stride_ds_skv
        tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


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

    # Transpose dO: [bs, sq, 80, d] -> [bs, 80, sq, d] contiguous, keep BF16
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()  # [bs, 80, sq, d] bfloat16

    # Reshape dO for GQA-aware GEMMs:
    # [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    dO_grouped = dO.view(bs, n_kv_heads, n_groups, seq_q, d)
    dO_gqa = dO_grouped.reshape(bs * n_kv_heads, n_groups * seq_q, d)  # [bs*8, 10*sq, d]

    # V: [bs, 8, skv, d] -> [bs*8, skv, d] — NO expansion needed!
    V_gqa = value_states.reshape(bs * n_kv_heads, seq_kv, d)  # [bs*8, skv, d]

    # ---- GEMM 1 (BF16): dP = dO_gqa @ V_gqa^T ----
    # [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
    dP_gqa = torch.bmm(dO_gqa, V_gqa.transpose(-2, -1))  # [bs*8, 10*sq, skv] BF16
    dP_raw = dP_gqa.view(bs, n_kv_heads, n_groups, seq_q, seq_kv) \
                   .reshape(bs, n_heads, seq_q, seq_kv)  # [bs, 80, sq, skv] BF16

    # ---- GEMM 2 (BF16): dV = P_dropped_gqa^T @ dO_gqa ----
    # P_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
    P_dropped_gqa = attn_weights_dropped.view(bs, n_kv_heads, n_groups, seq_q, seq_kv) \
                                        .reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
    # This directly sums over all 10 groups for each KV head — no extra reduction needed!
    dV_gqa = torch.bmm(P_dropped_gqa.transpose(-2, -1), dO_gqa)  # [bs*8, skv, d] BF16

    # Reshape to [bs, 8, skv, d]
    dV = dV_gqa.view(bs, n_kv_heads, seq_kv, d).to(torch.bfloat16)

    # ---- Triton kernel: softmax backward (reads BF16 dP_raw) ----
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    P_attn = attn_weights.contiguous()   # [bs, 80, sq, skv] bfloat16
    dmask  = dropout_mask.contiguous()   # [bs, 80, sq, skv] bool

    inv_keep = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    BLOCK_SQ_DS  = 16
    BLOCK_SKV_DS = 64

    grid_dS = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ_DS))

    softmax_bwd_kernel[grid_dS](
        dP_raw, dP_raw.stride(0), dP_raw.stride(1), dP_raw.stride(2), dP_raw.stride(3),
        P_attn, P_attn.stride(0), P_attn.stride(1), P_attn.stride(2), P_attn.stride(3),
        dmask, dmask.stride(0), dmask.stride(1), dmask.stride(2), dmask.stride(3),
        dS, dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
        bs, n_heads, seq_q, seq_kv,
        inv_keep_prob=inv_keep,
        BLOCK_SQ=BLOCK_SQ_DS, BLOCK_SKV=BLOCK_SKV_DS,
    )

    return dS, dV

```

---

## Experiment #6 — 2026-06-30 21:16:41 UTC ❌ DISCARD

**Hypothesis:** ** Replaced the two-step (BMM → Triton softmax-bwd) pipeline with a single `fused_softmax_bwd_kernel` that:

**Result:** 1833.86 μs

---

## Experiment #7 — 2026-06-30 21:18:46 UTC 💥 CRASH

**Hypothesis:** **

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #8 — 2026-06-30 21:20:54 UTC ❌ DISCARD

**Hypothesis:** ** Added a new `softmax_bwd_single_pass_kernel` that launches one Triton program per `(batch, head, sq_row)` — eliminating the `BLOCK_SQ` dimension entirely. This kernel handles one row at a time with

**Result:** 601.48 μs

---

## Experiment #9 — 2026-06-30 21:23:11 UTC ✅ KEEP

**Hypothesis:** ** Replaced the two separate kernels (`softmax_bwd_single_pass_kernel` and `softmax_bwd_large_block_kernel`) with a single `@triton.autotune`-decorated `softmax_bwd_kernel`. The autotune explores 12 c

**Result:** 464.59 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using BF16 GEMMs + Triton softmax backward.

Strategy:
  1. Avoid materializing V_exp [bs, 80, skv, d] entirely.
     For dP: reshape dO from [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
             use V as [bs*8, skv, d], compute bmm -> [bs*8, 10*sq, skv]
             reshape to [bs, 80, sq, skv]. No V copy needed!

  2. For dV: reshape P_dropped from [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
             reshape dO from [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
             compute [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
             This directly gives dV summed over groups -- no separate reduction!

  3. Triton softmax-backward kernel: autotuned single-pass-when-possible approach.
     - For dO: avoid permute+contiguous by passing non-contiguous strides.
     - Autotune (BLOCK_SQ, BLOCK_SKV) combos for best throughput.

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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 256}),
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 512}),
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 1024}),
        triton.Config({'BLOCK_SQ': 2,  'BLOCK_SKV': 256}),
        triton.Config({'BLOCK_SQ': 2,  'BLOCK_SKV': 512}),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 128}),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 256}),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 512}),
        triton.Config({'BLOCK_SQ': 8,  'BLOCK_SKV': 128}),
        triton.Config({'BLOCK_SQ': 8,  'BLOCK_SKV': 256}),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 128}),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 64}),
    ],
    key=['sq', 'skv'],
)
@triton.jit
def softmax_bwd_kernel(
    # dP: [bs, 80, sq, skv] bfloat16
    dP_ptr, stride_dp_bs, stride_dp_h, stride_dp_sq, stride_dp_skv,
    # P (attn_weights): [bs, 80, sq, skv] bfloat16
    P_ptr, stride_p_bs, stride_p_h, stride_p_sq, stride_p_skv,
    # dropout_mask: [bs, 80, sq, skv] bool
    mask_ptr, stride_m_bs, stride_m_h, stride_m_sq, stride_m_skv,
    # dS output: [bs, 80, sq, skv] bfloat16
    dS_ptr, stride_ds_bs, stride_ds_h, stride_ds_sq, stride_ds_skv,
    bs, n_heads, sq, skv,
    inv_keep_prob: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """
    Autotuned two-pass softmax backward.
    Grid: (bs * n_heads, cdiv(sq, BLOCK_SQ))
    Processes BLOCK_SQ rows simultaneously for better memory efficiency.
    Uses large BLOCK_SKV to minimize loop iterations.
    """
    pid_bh = tl.program_id(0)
    pid_sq = tl.program_id(1)

    batch_idx = pid_bh // n_heads
    head      = pid_bh % n_heads

    sq_start = pid_sq * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < sq

    skv_offs = tl.arange(0, BLOCK_SKV)

    dP_base = dP_ptr   + batch_idx * stride_dp_bs + head * stride_dp_h
    P_base  = P_ptr    + batch_idx * stride_p_bs  + head * stride_p_h
    M_base  = mask_ptr + batch_idx * stride_m_bs  + head * stride_m_h
    dS_base = dS_ptr   + batch_idx * stride_ds_bs + head * stride_ds_h

    num_skv_blocks = tl.cdiv(skv, BLOCK_SKV)

    # ----- Pass 1: compute rowsum(dP_masked * P) -----
    rowsum = tl.zeros((BLOCK_SQ,), dtype=tl.float32)

    for skv_tile in range(num_skv_blocks):
        skv_start     = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask      = skv_tile_offs < skv

        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dp_ptrs  = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
        dp_tile  = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        m_ptrs   = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile   = tl.load(m_ptrs, mask=combined_mask, other=0).to(tl.float32)
        dp_masked = dp_tile * m_tile * inv_keep_prob

        p_ptrs   = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile   = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        rowsum  += tl.sum(dp_masked * p_tile, axis=1)

    # ----- Pass 2: compute dS and store -----
    for skv_tile in range(num_skv_blocks):
        skv_start     = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask      = skv_tile_offs < skv

        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dp_ptrs   = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
        dp_tile   = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        m_ptrs    = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile    = tl.load(m_ptrs, mask=combined_mask, other=0).to(tl.float32)
        dp_masked = dp_tile * m_tile * inv_keep_prob

        p_ptrs    = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile    = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        dS_tile = p_tile * (dp_masked - rowsum[:, None])

        ds_ptrs = dS_base + sq_offs[:, None] * stride_ds_sq + skv_tile_offs[None, :] * stride_ds_skv
        tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


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

    # Avoid permute+contiguous for dO — pass non-contiguous strides to GEMM.
    # grad_attn_output: [bs, sq, 80, d]
    # We need it as [bs, 80, sq, d] — use .permute() without .contiguous()
    # Then view it as [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d] for GQA-aware GEMM.
    # However, bmm requires contiguous input, so we still need contiguous for GEMMs.
    # Use contiguous() here but avoid it for the softmax kernel.
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()  # [bs, 80, sq, d] bfloat16

    # Reshape dO for GQA-aware GEMMs:
    # [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    dO_grouped = dO.view(bs, n_kv_heads, n_groups, seq_q, d)
    dO_gqa = dO_grouped.reshape(bs * n_kv_heads, n_groups * seq_q, d)  # [bs*8, 10*sq, d]

    # V: [bs, 8, skv, d] -> [bs*8, skv, d] — NO expansion needed!
    V_gqa = value_states.reshape(bs * n_kv_heads, seq_kv, d)  # [bs*8, skv, d]

    # ---- GEMM 1 (BF16): dP = dO_gqa @ V_gqa^T ----
    # [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
    dP_gqa = torch.bmm(dO_gqa, V_gqa.transpose(-2, -1))  # [bs*8, 10*sq, skv] BF16
    dP_raw = dP_gqa.view(bs, n_kv_heads, n_groups, seq_q, seq_kv) \
                   .reshape(bs, n_heads, seq_q, seq_kv)  # [bs, 80, sq, skv] BF16

    # ---- GEMM 2 (BF16): dV = P_dropped_gqa^T @ dO_gqa ----
    # P_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
    P_dropped_gqa = attn_weights_dropped.view(bs, n_kv_heads, n_groups, seq_q, seq_kv) \
                                        .reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
    dV_gqa = torch.bmm(P_dropped_gqa.transpose(-2, -1), dO_gqa)  # [bs*8, skv, d] BF16

    # Reshape to [bs, 8, skv, d]
    dV = dV_gqa.view(bs, n_kv_heads, seq_kv, d).to(torch.bfloat16)

    # ---- Triton kernel: softmax backward ----
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    P_attn = attn_weights.contiguous()   # [bs, 80, sq, skv] bfloat16
    dmask  = dropout_mask.contiguous()   # [bs, 80, sq, skv] bool

    inv_keep = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Use single autotuned kernel for all sizes
    grid_dS = lambda meta: (bs * n_heads, triton.cdiv(seq_q, meta['BLOCK_SQ']))
    softmax_bwd_kernel[grid_dS](
        dP_raw, dP_raw.stride(0), dP_raw.stride(1), dP_raw.stride(2), dP_raw.stride(3),
        P_attn, P_attn.stride(0), P_attn.stride(1), P_attn.stride(2), P_attn.stride(3),
        dmask,  dmask.stride(0),  dmask.stride(1),  dmask.stride(2),  dmask.stride(3),
        dS,     dS.stride(0),     dS.stride(1),     dS.stride(2),     dS.stride(3),
        bs, n_heads, seq_q, seq_kv,
        inv_keep_prob=inv_keep,
    )

    return dS, dV

```

---

## Experiment #10 — 2026-06-30 21:24:56 UTC 💥 CRASH

**Hypothesis:** ** Replaced the `softmax_bwd_kernel` (which read dP from HBM) with a new `fused_softmax_bwd_kernel` that:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #11 — 2026-06-30 21:27:07 UTC ❌ DISCARD

**Hypothesis:** Added CUDA stream parallelism — GEMM1 (dP = dO_gqa @ V_gqa^T) runs on stream1 and GEMM2 (dV = P_dropped_gqa^T @ dO_gqa) runs on stream2 simultaneously. Both streams first wait on the current stream (e

**Result:** 518.09 μs

---

## Experiment #12 — 2026-06-30 21:29:30 UTC ✅ KEEP

**Hypothesis:** Worker implementation

**Result:** 413.40 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using BF16 GEMMs + Triton softmax backward.

Strategy:
  1. Avoid materializing V_exp [bs, 80, skv, d] entirely.
     For dP: reshape dO from [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
             use V as [bs*8, skv, d], compute bmm -> [bs*8, 10*sq, skv]
             reshape to [bs, 80, sq, skv]. No V copy needed!

  2. For dV: reshape P_dropped from [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
             reshape dO from [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
             compute [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
             This directly gives dV summed over groups -- no separate reduction!
             Avoid transpose copy by computing bmm(P^T, dO) via
             bmm(P_dropped_gqa.transpose(-2,-1), dO_gqa) on stream2.

  3. Stream parallelism: launch GEMM1 (dP) and GEMM2 (dV) on separate CUDA streams
     so they overlap in execution. Both are independent computations.

  4. Triton softmax-backward kernel: autotuned single-pass-when-possible approach.

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
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 256}),
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 512}),
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 1024}),
        triton.Config({'BLOCK_SQ': 2,  'BLOCK_SKV': 256}),
        triton.Config({'BLOCK_SQ': 2,  'BLOCK_SKV': 512}),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 128}),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 256}),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 512}),
        triton.Config({'BLOCK_SQ': 8,  'BLOCK_SKV': 128}),
        triton.Config({'BLOCK_SQ': 8,  'BLOCK_SKV': 256}),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 128}),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 64}),
    ],
    key=['sq', 'skv'],
)
@triton.jit
def softmax_bwd_kernel(
    # dP: [bs, 80, sq, skv] bfloat16
    dP_ptr, stride_dp_bs, stride_dp_h, stride_dp_sq, stride_dp_skv,
    # P (attn_weights): [bs, 80, sq, skv] bfloat16
    P_ptr, stride_p_bs, stride_p_h, stride_p_sq, stride_p_skv,
    # dropout_mask: [bs, 80, sq, skv] bool
    mask_ptr, stride_m_bs, stride_m_h, stride_m_sq, stride_m_skv,
    # dS output: [bs, 80, sq, skv] bfloat16
    dS_ptr, stride_ds_bs, stride_ds_h, stride_ds_sq, stride_ds_skv,
    bs, n_heads, sq, skv,
    inv_keep_prob: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """
    Autotuned two-pass softmax backward.
    Grid: (bs * n_heads, cdiv(sq, BLOCK_SQ))
    Processes BLOCK_SQ rows simultaneously for better memory efficiency.
    Uses large BLOCK_SKV to minimize loop iterations.
    """
    pid_bh = tl.program_id(0)
    pid_sq = tl.program_id(1)

    batch_idx = pid_bh // n_heads
    head      = pid_bh % n_heads

    sq_start = pid_sq * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < sq

    skv_offs = tl.arange(0, BLOCK_SKV)

    dP_base = dP_ptr   + batch_idx * stride_dp_bs + head * stride_dp_h
    P_base  = P_ptr    + batch_idx * stride_p_bs  + head * stride_p_h
    M_base  = mask_ptr + batch_idx * stride_m_bs  + head * stride_m_h
    dS_base = dS_ptr   + batch_idx * stride_ds_bs + head * stride_ds_h

    num_skv_blocks = tl.cdiv(skv, BLOCK_SKV)

    # ----- Pass 1: compute rowsum(dP_masked * P) -----
    rowsum = tl.zeros((BLOCK_SQ,), dtype=tl.float32)

    for skv_tile in range(num_skv_blocks):
        skv_start     = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask      = skv_tile_offs < skv

        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dp_ptrs  = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
        dp_tile  = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        m_ptrs   = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile   = tl.load(m_ptrs, mask=combined_mask, other=0).to(tl.float32)
        dp_masked = dp_tile * m_tile * inv_keep_prob

        p_ptrs   = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile   = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        rowsum  += tl.sum(dp_masked * p_tile, axis=1)

    # ----- Pass 2: compute dS and store -----
    for skv_tile in range(num_skv_blocks):
        skv_start     = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask      = skv_tile_offs < skv

        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dp_ptrs   = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
        dp_tile   = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        m_ptrs    = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile    = tl.load(m_ptrs, mask=combined_mask, other=0).to(tl.float32)
        dp_masked = dp_tile * m_tile * inv_keep_prob

        p_ptrs    = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile    = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        dS_tile = p_tile * (dp_masked - rowsum[:, None])

        ds_ptrs = dS_base + sq_offs[:, None] * stride_ds_sq + skv_tile_offs[None, :] * stride_ds_skv
        tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


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

    # Reshape dO for GQA-aware GEMMs:
    # [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    dO_grouped = dO.view(bs, n_kv_heads, n_groups, seq_q, d)
    dO_gqa = dO_grouped.reshape(bs * n_kv_heads, n_groups * seq_q, d)  # [bs*8, 10*sq, d]

    # V: [bs, 8, skv, d] -> [bs*8, skv, d] — NO expansion needed!
    V_gqa = value_states.reshape(bs * n_kv_heads, seq_kv, d)  # [bs*8, skv, d]

    # P_dropped for dV: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
    P_dropped_gqa = attn_weights_dropped.view(bs, n_kv_heads, n_groups, seq_q, seq_kv) \
                                        .reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # Pre-allocate output tensors
    dP_gqa = torch.empty((bs * n_kv_heads, n_groups * seq_q, seq_kv), dtype=torch.bfloat16, device=device)
    dV_gqa = torch.empty((bs * n_kv_heads, seq_kv, d), dtype=torch.bfloat16, device=device)

    # Get the two streams
    stream1, stream2 = _get_streams()
    current_stream = torch.cuda.current_stream(device)

    # ---- GEMM 1 (BF16): dP = dO_gqa @ V_gqa^T on stream1 ----
    with torch.cuda.stream(stream1):
        # Wait for current stream work (dO_gqa, V_gqa) to finish
        stream1.wait_stream(current_stream)
        # [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
        torch.bmm(dO_gqa, V_gqa.transpose(-2, -1), out=dP_gqa)

    # ---- GEMM 2 (BF16): dV = P_dropped_gqa^T @ dO_gqa on stream2 ----
    with torch.cuda.stream(stream2):
        # Wait for current stream work to finish
        stream2.wait_stream(current_stream)
        # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
        # Use contiguous P_dropped_gqa.transpose to avoid implicit copy in bmm
        # by using the non-transposed form: compute dO^T @ P first approach
        # Actually bmm needs contiguous; transpose(-2,-1) creates non-contiguous view
        # so bmm will handle it. This is the same cost as before but now overlapped.
        torch.bmm(P_dropped_gqa.transpose(-2, -1), dO_gqa, out=dV_gqa)

    # Synchronize both streams back to current stream before using results
    current_stream.wait_stream(stream1)
    current_stream.wait_stream(stream2)

    # Reshape dP_gqa back to [bs, 80, sq, skv]
    dP_raw = dP_gqa.view(bs, n_kv_heads, n_groups, seq_q, seq_kv) \
                   .reshape(bs, n_heads, seq_q, seq_kv)  # [bs, 80, sq, skv] BF16

    # Reshape dV to [bs, 8, skv, d]
    dV = dV_gqa.view(bs, n_kv_heads, seq_kv, d).to(torch.bfloat16)

    # ---- Triton kernel: softmax backward ----
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=device)

    P_attn = attn_weights.contiguous()   # [bs, 80, sq, skv] bfloat16
    dmask  = dropout_mask.contiguous()   # [bs, 80, sq, skv] bool

    inv_keep = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Use single autotuned kernel for all sizes
    grid_dS = lambda meta: (bs * n_heads, triton.cdiv(seq_q, meta['BLOCK_SQ']))
    softmax_bwd_kernel[grid_dS](
        dP_raw, dP_raw.stride(0), dP_raw.stride(1), dP_raw.stride(2), dP_raw.stride(3),
        P_attn, P_attn.stride(0), P_attn.stride(1), P_attn.stride(2), P_attn.stride(3),
        dmask,  dmask.stride(0),  dmask.stride(1),  dmask.stride(2),  dmask.stride(3),
        dS,     dS.stride(0),     dS.stride(1),     dS.stride(2),     dS.stride(3),
        bs, n_heads, seq_q, seq_kv,
        inv_keep_prob=inv_keep,
    )

    return dS, dV

```

---

## Experiment #13 — 2026-06-30 21:32:27 UTC 💥 CRASH

**Hypothesis:** Replaced the two-kernel approach (cuBLAS BMM for dP → HBM → Triton softmax-bwd) with a single `fused_dS_kernel` that:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #14 — 2026-06-30 21:35:14 UTC ❌ DISCARD

**Hypothesis:** ** Replaced the two-kernel approach (GEMM→dP_HBM→softmax_bwd) with a single `fused_ds_kernel` Triton kernel that:

**Result:** 713.14 μs

---

## Experiment #15 — 2026-06-30 21:40:18 UTC ❌ DISCARD

**Hypothesis:** Expanded the @triton.autotune configs from 12 to 40 configurations, adding: (a) num_warps=16 configs for B200 which supports wider warps, (b) num_stages=4 configs for better HBM latency hiding on the 

**Result:** 680.62 μs

---

## Experiment #16 — 2026-06-30 21:45:52 UTC ❌ DISCARD

**Hypothesis:** ** Replaced the `mask_ptr` parameter (bool tensor) with `Pd_ptr` (P_dropped, bf16 tensor) in `fused_ds_kernel`. In Pass 1 (rowsum computation), instead of `dp_masked = dP_raw * mask * inv_keep` follow

**Result:** 764.63 μs

---

## Experiment #17 — 2026-06-30 21:50:54 UTC 💥 CRASH

**Hypothesis:** ** Added a `SINGLE_PASS: tl.constexpr` parameter to `fused_ds_kernel` that branches between:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #18 — 2026-06-30 21:53:47 UTC ❌ DISCARD

**Hypothesis:** Replaced the two-kernel approach (separate dP GEMM + softmax_bwd_kernel) with a single fused_softmax_bwd_kernel that: (1) loads the dO row once into registers, (2) in pass 1 iterates over KV blocks lo

**Result:** 4880.70 μs

---

## Experiment #19 — 2026-06-30 21:58:02 UTC ❌ DISCARD

**Hypothesis:** ** Three targeted changes to the Triton kernel and host code:

**Result:** 3453.73 μs

---

## Experiment #20 — 2026-06-30 22:01:46 UTC 💥 CRASH

**Hypothesis:** **

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #21 — 2026-06-30 22:04:21 UTC ✅ KEEP

**Hypothesis:** **

**Result:** 408.02 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using BF16 GEMMs + Triton softmax backward.

Strategy:
  1. Avoid materializing V_exp [bs, 80, skv, d] entirely.
     For dP: reshape dO from [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
             use V as [bs*8, skv, d], compute bmm -> [bs*8, 10*sq, skv]
             reshape to [bs, 80, sq, skv]. No V copy needed!

  2. For dV: reshape P_dropped from [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
             reshape dO from [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
             compute [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
             This directly gives dV summed over groups -- no separate reduction!

  3. Stream parallelism: launch GEMM1 (dP) and GEMM2 (dV) on separate CUDA streams
     so they overlap in execution. Both are independent computations.

  4. Triton softmax-backward kernel: single-pass when entire skv row fits in SRAM,
     otherwise two-pass. Large BLOCK_SKV configs with num_warps=8/16 for B200.
     Uses attn_weights_dropped directly to avoid dropout mask load when possible.

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
        # Single-pass configs (large BLOCK_SKV to cover full skv in one tile)
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 512},  num_warps=8),
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 1024}, num_warps=8),
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 1024}, num_warps=16),
        triton.Config({'BLOCK_SQ': 2,  'BLOCK_SKV': 512},  num_warps=8),
        triton.Config({'BLOCK_SQ': 2,  'BLOCK_SKV': 1024}, num_warps=8),
        triton.Config({'BLOCK_SQ': 2,  'BLOCK_SKV': 1024}, num_warps=16),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 512},  num_warps=8),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 1024}, num_warps=8),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 512},  num_warps=16),
        # Two-pass fallback configs (smaller tiles, also with higher warps)
        triton.Config({'BLOCK_SQ': 1,  'BLOCK_SKV': 256},  num_warps=8),
        triton.Config({'BLOCK_SQ': 2,  'BLOCK_SKV': 256},  num_warps=8),
        triton.Config({'BLOCK_SQ': 4,  'BLOCK_SKV': 256},  num_warps=8),
        triton.Config({'BLOCK_SQ': 8,  'BLOCK_SKV': 128},  num_warps=8),
        triton.Config({'BLOCK_SQ': 8,  'BLOCK_SKV': 256},  num_warps=8),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 128},  num_warps=8),
        triton.Config({'BLOCK_SQ': 16, 'BLOCK_SKV': 64},   num_warps=8),
    ],
    key=['sq', 'skv'],
)
@triton.jit
def softmax_bwd_kernel(
    # dP: [bs, 80, sq, skv] bfloat16
    dP_ptr, stride_dp_bs, stride_dp_h, stride_dp_sq, stride_dp_skv,
    # P (attn_weights): [bs, 80, sq, skv] bfloat16
    P_ptr, stride_p_bs, stride_p_h, stride_p_sq, stride_p_skv,
    # dropout_mask: [bs, 80, sq, skv] bool
    mask_ptr, stride_m_bs, stride_m_h, stride_m_sq, stride_m_skv,
    # dS output: [bs, 80, sq, skv] bfloat16
    dS_ptr, stride_ds_bs, stride_ds_h, stride_ds_sq, stride_ds_skv,
    bs, n_heads, sq, skv,
    inv_keep_prob: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """
    Single-pass softmax backward when BLOCK_SKV >= skv (entire row fits in one tile),
    otherwise falls back to two-pass. Large BLOCK_SKV configs with more warps
    maximize memory bandwidth on B200.

    Grid: (bs * n_heads, cdiv(sq, BLOCK_SQ))
    """
    pid_bh = tl.program_id(0)
    pid_sq = tl.program_id(1)

    batch_idx = pid_bh // n_heads
    head      = pid_bh % n_heads

    sq_start = pid_sq * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < sq

    skv_offs = tl.arange(0, BLOCK_SKV)

    dP_base = dP_ptr   + batch_idx * stride_dp_bs + head * stride_dp_h
    P_base  = P_ptr    + batch_idx * stride_p_bs  + head * stride_p_h
    M_base  = mask_ptr + batch_idx * stride_m_bs  + head * stride_m_h
    dS_base = dS_ptr   + batch_idx * stride_ds_bs + head * stride_ds_h

    num_skv_blocks = tl.cdiv(skv, BLOCK_SKV)

    # ----- Single-pass: when the entire skv dimension fits in one BLOCK_SKV tile -----
    # tl.constexpr comparison: if BLOCK_SKV covers all of skv in one block
    if BLOCK_SKV >= skv:
        # Single tile covers entire row — compute rowsum and dS in one pass
        skv_tile_offs = skv_offs  # just the first (and only) tile
        skv_msk       = skv_tile_offs < skv
        combined_mask = sq_mask[:, None] & skv_msk[None, :]

        dp_ptrs  = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
        dp_tile  = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        m_ptrs   = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile   = tl.load(m_ptrs, mask=combined_mask, other=0).to(tl.float32)
        dp_masked = dp_tile * m_tile * inv_keep_prob

        p_ptrs   = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile   = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # Compute rowsum in registers — no second HBM pass needed
        rowsum = tl.sum(dp_masked * p_tile, axis=1)

        dS_tile = p_tile * (dp_masked - rowsum[:, None])

        ds_ptrs = dS_base + sq_offs[:, None] * stride_ds_sq + skv_tile_offs[None, :] * stride_ds_skv
        tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)

    else:
        # ----- Two-pass fallback: skv is larger than BLOCK_SKV -----

        # Pass 1: compute rowsum(dP_masked * P)
        rowsum = tl.zeros((BLOCK_SQ,), dtype=tl.float32)

        for skv_tile in range(num_skv_blocks):
            skv_start     = skv_tile * BLOCK_SKV
            skv_tile_offs = skv_start + skv_offs
            skv_msk       = skv_tile_offs < skv

            combined_mask = sq_mask[:, None] & skv_msk[None, :]

            dp_ptrs  = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
            dp_tile  = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

            m_ptrs   = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
            m_tile   = tl.load(m_ptrs, mask=combined_mask, other=0).to(tl.float32)
            dp_masked = dp_tile * m_tile * inv_keep_prob

            p_ptrs   = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
            p_tile   = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

            rowsum  += tl.sum(dp_masked * p_tile, axis=1)

        # Pass 2: compute dS and store
        for skv_tile in range(num_skv_blocks):
            skv_start     = skv_tile * BLOCK_SKV
            skv_tile_offs = skv_start + skv_offs
            skv_msk       = skv_tile_offs < skv

            combined_mask = sq_mask[:, None] & skv_msk[None, :]

            dp_ptrs   = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
            dp_tile   = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

            m_ptrs    = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
            m_tile    = tl.load(m_ptrs, mask=combined_mask, other=0).to(tl.float32)
            dp_masked = dp_tile * m_tile * inv_keep_prob

            p_ptrs    = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
            p_tile    = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

            dS_tile = p_tile * (dp_masked - rowsum[:, None])

            ds_ptrs = dS_base + sq_offs[:, None] * stride_ds_sq + skv_tile_offs[None, :] * stride_ds_skv
            tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


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

    # Reshape dO for GQA-aware GEMMs:
    # [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    dO_grouped = dO.view(bs, n_kv_heads, n_groups, seq_q, d)
    dO_gqa = dO_grouped.reshape(bs * n_kv_heads, n_groups * seq_q, d)  # [bs*8, 10*sq, d]

    # V: [bs, 8, skv, d] -> [bs*8, skv, d] — NO expansion needed!
    V_gqa = value_states.reshape(bs * n_kv_heads, seq_kv, d)  # [bs*8, skv, d]

    # P_dropped for dV: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
    P_dropped_gqa = attn_weights_dropped.view(bs, n_kv_heads, n_groups, seq_q, seq_kv) \
                                        .reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # Pre-allocate output tensors
    dP_gqa = torch.empty((bs * n_kv_heads, n_groups * seq_q, seq_kv), dtype=torch.bfloat16, device=device)
    dV_gqa = torch.empty((bs * n_kv_heads, seq_kv, d), dtype=torch.bfloat16, device=device)

    # Get the two streams
    stream1, stream2 = _get_streams()
    current_stream = torch.cuda.current_stream(device)

    # ---- GEMM 1 (BF16): dP = dO_gqa @ V_gqa^T on stream1 ----
    with torch.cuda.stream(stream1):
        stream1.wait_stream(current_stream)
        # [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
        torch.bmm(dO_gqa, V_gqa.transpose(-2, -1), out=dP_gqa)

    # ---- GEMM 2 (BF16): dV = P_dropped_gqa^T @ dO_gqa on stream2 ----
    with torch.cuda.stream(stream2):
        stream2.wait_stream(current_stream)
        # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
        torch.bmm(P_dropped_gqa.transpose(-2, -1), dO_gqa, out=dV_gqa)

    # Synchronize both streams back to current stream before using results
    current_stream.wait_stream(stream1)
    current_stream.wait_stream(stream2)

    # Reshape dP_gqa back to [bs, 80, sq, skv]
    dP_raw = dP_gqa.view(bs, n_kv_heads, n_groups, seq_q, seq_kv) \
                   .reshape(bs, n_heads, seq_q, seq_kv)  # [bs, 80, sq, skv] BF16

    # Reshape dV to [bs, 8, skv, d]
    dV = dV_gqa.view(bs, n_kv_heads, seq_kv, d).to(torch.bfloat16)

    # ---- Triton kernel: softmax backward ----
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=device)

    P_attn = attn_weights.contiguous()   # [bs, 80, sq, skv] bfloat16
    dmask  = dropout_mask.contiguous()   # [bs, 80, sq, skv] bool

    inv_keep = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Use single autotuned kernel for all sizes
    grid_dS = lambda meta: (bs * n_heads, triton.cdiv(seq_q, meta['BLOCK_SQ']))
    softmax_bwd_kernel[grid_dS](
        dP_raw, dP_raw.stride(0), dP_raw.stride(1), dP_raw.stride(2), dP_raw.stride(3),
        P_attn, P_attn.stride(0), P_attn.stride(1), P_attn.stride(2), P_attn.stride(3),
        dmask,  dmask.stride(0),  dmask.stride(1),  dmask.stride(2),  dmask.stride(3),
        dS,     dS.stride(0),     dS.stride(1),     dS.stride(2),     dS.stride(3),
        bs, n_heads, seq_q, seq_kv,
        inv_keep_prob=inv_keep,
    )

    return dS, dV

```

---

## Experiment #22 — 2026-06-30 22:07:39 UTC ❌ DISCARD

**Hypothesis:** ** Added a `_make_dO_gqa_contiguous()` helper that uses `torch.as_strided` to create a 5D non-contiguous view `[bs, 8, 10, sq, d]` directly from `grad_attn_output`'s `[bs, sq, 80, d]` layout, with cus

**Result:** 421.82 μs

---

## Experiment #23 — 2026-06-30 22:10:45 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 436.34 μs

---

## Experiment #24 — 2026-06-30 22:14:26 UTC ❌ DISCARD

**Hypothesis:** ** Replaced the two-stage approach (cuBLAS GEMM1 for dP → Triton softmax_bwd) with a single `fused_dp_softmax_bwd_kernel` that:

**Result:** 770.47 μs

---

## Experiment #25 — 2026-06-30 22:17:01 UTC 💥 CRASH

**Hypothesis:** ** Rewrote the fused Triton kernel to:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #26 — 2026-06-30 22:19:29 UTC ❌ DISCARD

**Hypothesis:** ** Rewrote `softmax_bwd_kernel` to accept `Pd_ptr` (attn_weights_dropped, bf16) instead of `mask_ptr` (bool). Changed the rowsum computation to `sum(dP_tile * pd_tile)` and the dS computation to `pd_t

**Result:** 421.75 μs

---

## Experiment #27 — 2026-06-30 22:22:21 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 498.79 μs

---

## Experiment #28 — 2026-06-30 22:25:27 UTC ❌ DISCARD

**Hypothesis:** ** Restructured GEMM1 (computing dP) to use `dO_5d` in its native `[bs, seq_q, n_kv_heads, n_groups, d]` layout with `torch.matmul` broadcasting against `V_T_5d = [bs, 1, n_kv_heads, d, seq_kv]` → pro

**Result:** 5730.47 μs

---

## Experiment #29 — 2026-06-30 22:28:20 UTC ❌ DISCARD

**Hypothesis:** ** Replaced the separate GEMM1 + `softmax_bwd_kernel` with a single `fused_softmax_bwd_kernel` that:

**Result:** 690.75 μs

---

## Experiment #30 — 2026-06-30 22:31:50 UTC ❌ DISCARD

**Hypothesis:** ** Rewrote `submission.py` to:

**Result:** 852.35 μs

---

## Experiment #31 — 2026-06-30 22:35:13 UTC ❌ DISCARD

**Hypothesis:** ** Replaced `grad_attn_output.permute(0, 2, 1, 3).contiguous()` with a custom Triton `transpose_dO_kernel` that directly reads `[bs,sq,n_heads,d]` and writes `[bs,n_heads,sq,d]` using vectorized 8-ele

**Result:** 744.50 μs

---

## Experiment #32 — 2026-06-30 22:37:32 UTC 💥 CRASH

**Hypothesis:** ** Replaced the separate GEMM1 (cuBLAS BMM writing `dP_raw` to HBM) + `softmax_bwd_kernel` (reading `dP_raw` back from HBM) with a single fused `fused_dP_softmax_bwd_kernel` that:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #33 — 2026-06-30 22:40:00 UTC 💥 CRASH

**Hypothesis:** ** Split the single combined `softmax_bwd_kernel` into two separate kernels:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #34 — 2026-06-30 22:43:11 UTC ❌ DISCARD

**Hypothesis:** ** Added `attn_weights_dropped` (`P_drop`/`Pd`) as a new kernel parameter. In pass 1 (rowsum), the kernel now loads `dP_raw` and `P_dropped` only — no dropout mask needed. In pass 2, it loads `dP_raw`

**Result:** 436.07 μs

---

## Experiment #35 — 2026-06-30 22:44:54 UTC 💥 CRASH

**Hypothesis:** ** Replaced the separate `softmax_bwd_kernel` (which consumed pre-computed `dP_raw` from HBM) with a new `fused_dP_softmax_bwd_kernel` that takes `dO` and `V` as inputs, computes `dP_raw = dO @ V^T` i

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #36 — 2026-06-30 22:46:26 UTC ❌ DISCARD

**Hypothesis:** ** Removed the entire `softmax_bwd_kernel` Triton kernel and its autotune configuration. Added a `_softmax_bwd_fn` Python function that performs: `dP_masked = dP_raw.float() * dropout_mask.float() * i

**Result:** 910.23 μs

---

## Experiment #37 — 2026-06-30 22:47:56 UTC ❌ DISCARD

**Hypothesis:** ** Changed the stream scheduling order and structure. GEMM2 is launched first on stream2 (it's independent of everything), then GEMM1 + softmax-bwd are chained together on stream1 within the same `wit

**Result:** 882.62 μs

---

## Experiment #38 — 2026-06-30 22:49:27 UTC ❌ DISCARD

**Hypothesis:** Changed `mode="reduce-overhead"` to `mode="max-autotune"` in the `torch.compile` call for `_softmax_bwd_compiled`. All other code (GQA-aware cuBLAS BMMs, dual-stream overlap, stream scheduling) remain

**Result:** 884.03 μs

