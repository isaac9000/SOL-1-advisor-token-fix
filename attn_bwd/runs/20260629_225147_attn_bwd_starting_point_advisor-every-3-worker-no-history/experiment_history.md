# Experiment History

Tracks every kernel attempt, its code, hypothesis, and result.

---

## Experiment #1 — 2026-06-29 22:51:58 UTC ✅ KEEP

**Hypothesis:** Baseline 'starting_point' — initial benchmark

**Result:** 3428.46 μs

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

## Experiment #2 — 2026-06-29 22:53:59 UTC ✅ KEEP

**Hypothesis:** ** Two Triton kernels — `attn_bwd_ds_kernel` and `attn_bwd_dv_kernel` — replacing the pure PyTorch reference.

**Result:** 2404.77 μs

**Kernel code:**
```python
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
# Kernel 1: Fused dS = P*(dP - sum(dP*P)) where dP = dO @ V^T
# Each program handles one (bs_idx, head_idx, sq_tile) block
# ---------------------------------------------------------------------------
@triton.jit
def attn_bwd_ds_kernel(
    # pointers
    dO_ptr,            # [bs, n_heads, sq, d]   float32
    V_ptr,             # [bs, 8, skv, d]         bfloat16  (KV heads, unexpanded)
    P_ptr,             # [bs, n_heads, sq, skv]  bfloat16
    mask_ptr,          # [bs, n_heads, sq, skv]  bool
    dS_ptr,            # [bs, n_heads, sq, skv]  bfloat16  (output)
    # dims
    bs, n_heads, sq, skv,
    n_kv_heads, n_groups, head_dim,
    inv_scale,         # 1/(1-dropout)
    # strides for dO [bs, n_heads, sq, d]
    dO_stride_bs, dO_stride_h, dO_stride_sq, dO_stride_d,
    # strides for V [bs, 8, skv, d]
    V_stride_bs, V_stride_h, V_stride_skv, V_stride_d,
    # strides for P [bs, n_heads, sq, skv]
    P_stride_bs, P_stride_h, P_stride_sq, P_stride_skv,
    # strides for mask [bs, n_heads, sq, skv]
    M_stride_bs, M_stride_h, M_stride_sq, M_stride_skv,
    # strides for dS [bs, n_heads, sq, skv]
    dS_stride_bs, dS_stride_h, dS_stride_sq, dS_stride_skv,
    # tile sizes
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # program id
    pid_bh = tl.program_id(0)   # batch * n_heads
    pid_sq = tl.program_id(1)   # sq tile index

    bs_idx = pid_bh // n_heads
    h_idx  = pid_bh % n_heads
    kv_idx = h_idx // n_groups  # which KV head

    sq_start = pid_sq * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < sq

    d_offs = tl.arange(0, BLOCK_D)

    # Base pointer for dO: [bs_idx, h_idx, sq_offs, :]
    dO_base = (bs_idx * dO_stride_bs
               + h_idx * dO_stride_h)
    # Load dO tile [BLOCK_SQ, BLOCK_D]
    dO = tl.load(
        dO_ptr + dO_base
        + sq_offs[:, None] * dO_stride_sq
        + d_offs[None, :] * dO_stride_d,
        mask=sq_mask[:, None] & (d_offs[None, :] < head_dim),
        other=0.0,
    )  # float32

    # We will accumulate dP over skv tiles,
    # but since dS output size is [sq, skv] and we do softmax bwd,
    # we need all skv values at once (for the row-sum).
    # Strategy: compute dP_full row by row with a loop over skv blocks,
    # store to dS temporarily in float32, then apply softmax bwd in a second pass.
    # But that requires temporary storage. Instead, we use a two-pass approach
    # over skv blocks: first compute the row sum, then write final dS.
    # However this doubles the work. Given correctness requirements, do it properly.

    # Pass 1: accumulate row_sum = sum_skv(dP * P) for each sq row
    row_sum = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    # We also need to accumulate dP values. Since skv can be large, store in HBM
    # via dS_ptr as a temporary float32 buffer — but dS_ptr is bf16 output.
    # So do 2 passes: pass1 compute row_sum, pass2 write dS.

    # Base pointer for V: [bs_idx, kv_idx, :, :]
    V_base = bs_idx * V_stride_bs + kv_idx * V_stride_h

    # Pass 1: compute dP = dO @ V^T, apply mask, compute row sum of dP*P
    skv_offs_start = tl.arange(0, BLOCK_SKV)
    n_skv_blocks = tl.cdiv(skv, BLOCK_SKV)

    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_offs_start
        skv_mask = skv_offs < skv

        # Load V tile [BLOCK_SKV, BLOCK_D]
        V_tile = tl.load(
            V_ptr + V_base
            + skv_offs[:, None] * V_stride_skv
            + d_offs[None, :] * V_stride_d,
            mask=skv_mask[:, None] & (d_offs[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)  # [BLOCK_SKV, BLOCK_D]

        # dP_tile = dO @ V^T  [BLOCK_SQ, BLOCK_SKV]
        dP_tile = tl.dot(dO, tl.trans(V_tile))  # [BLOCK_SQ, BLOCK_SKV]

        # Load dropout mask [BLOCK_SQ, BLOCK_SKV]
        M_base = (bs_idx * M_stride_bs + h_idx * M_stride_h)
        drop_mask = tl.load(
            mask_ptr + M_base
            + sq_offs[:, None] * M_stride_sq
            + skv_offs[None, :] * M_stride_skv,
            mask=sq_mask[:, None] & skv_mask[None, :],
            other=0,
        )  # bool

        # Dropout backward: dP = dP_dropped * mask / (1-p)
        dP_tile = dP_tile * drop_mask.to(tl.float32) * inv_scale

        # Load P tile [BLOCK_SQ, BLOCK_SKV]
        P_base = (bs_idx * P_stride_bs + h_idx * P_stride_h)
        P_tile = tl.load(
            P_ptr + P_base
            + sq_offs[:, None] * P_stride_sq
            + skv_offs[None, :] * P_stride_skv,
            mask=sq_mask[:, None] & skv_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Accumulate row sum: sum(dP * P) across skv
        row_sum += tl.sum(dP_tile * P_tile, axis=1)  # [BLOCK_SQ]

    # Pass 2: recompute dP and write final dS = P * (dP - row_sum)
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_offs_start
        skv_mask = skv_offs < skv

        # Load V tile [BLOCK_SKV, BLOCK_D]
        V_tile = tl.load(
            V_ptr + V_base
            + skv_offs[:, None] * V_stride_skv
            + d_offs[None, :] * V_stride_d,
            mask=skv_mask[:, None] & (d_offs[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)

        # dP_tile = dO @ V^T  [BLOCK_SQ, BLOCK_SKV]
        dP_tile = tl.dot(dO, tl.trans(V_tile))

        # Load dropout mask
        M_base = bs_idx * M_stride_bs + h_idx * M_stride_h
        drop_mask = tl.load(
            mask_ptr + M_base
            + sq_offs[:, None] * M_stride_sq
            + skv_offs[None, :] * M_stride_skv,
            mask=sq_mask[:, None] & skv_mask[None, :],
            other=0,
        )
        dP_tile = dP_tile * drop_mask.to(tl.float32) * inv_scale

        # Load P tile
        P_base = bs_idx * P_stride_bs + h_idx * P_stride_h
        P_tile = tl.load(
            P_ptr + P_base
            + sq_offs[:, None] * P_stride_sq
            + skv_offs[None, :] * P_stride_skv,
            mask=sq_mask[:, None] & skv_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Softmax backward
        dS_tile = P_tile * (dP_tile - row_sum[:, None])  # [BLOCK_SQ, BLOCK_SKV]

        # Write to dS
        dS_base = bs_idx * dS_stride_bs + h_idx * dS_stride_h
        tl.store(
            dS_ptr + dS_base
            + sq_offs[:, None] * dS_stride_sq
            + skv_offs[None, :] * dS_stride_skv,
            dS_tile.to(tl.bfloat16),
            mask=sq_mask[:, None] & skv_mask[None, :],
        )


# ---------------------------------------------------------------------------
# Kernel 2: Fused dV with GQA reduction
# Each program handles one (bs_idx, kv_head_idx, skv_tile) block
# Accumulates over 10 GQA groups and all sq positions
# ---------------------------------------------------------------------------
@triton.jit
def attn_bwd_dv_kernel(
    # pointers
    P_drop_ptr,        # [bs, n_heads, sq, skv]   bfloat16
    dO_ptr,            # [bs, n_heads, sq, d]      float32
    dV_ptr,            # [bs, 8, skv, d]            bfloat16  (output)
    # dims
    bs, n_heads, sq, skv, n_kv_heads, n_groups, head_dim,
    # strides for P_drop [bs, n_heads, sq, skv]
    Pd_stride_bs, Pd_stride_h, Pd_stride_sq, Pd_stride_skv,
    # strides for dO [bs, n_heads, sq, d]
    dO_stride_bs, dO_stride_h, dO_stride_sq, dO_stride_d,
    # strides for dV [bs, 8, skv, d]
    dV_stride_bs, dV_stride_h, dV_stride_skv, dV_stride_d,
    # tile sizes
    BLOCK_SKV: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bkv = tl.program_id(0)   # batch * n_kv_heads
    pid_skv = tl.program_id(1)   # skv tile index

    bs_idx  = pid_bkv // n_kv_heads
    kv_idx  = pid_bkv % n_kv_heads

    skv_start = pid_skv * BLOCK_SKV
    skv_offs  = skv_start + tl.arange(0, BLOCK_SKV)
    skv_mask  = skv_offs < skv

    d_offs = tl.arange(0, BLOCK_D)
    sq_offs_start = tl.arange(0, BLOCK_SQ)

    # Accumulate dV [BLOCK_SKV, BLOCK_D]
    dV_acc = tl.zeros([BLOCK_SKV, BLOCK_D], dtype=tl.float32)

    # Loop over 10 GQA groups
    for g in range(0, n_groups):
        h_idx = kv_idx * n_groups + g

        # Loop over sq blocks
        for sq_block in range(0, tl.cdiv(sq, BLOCK_SQ)):
            sq_offs = sq_block * BLOCK_SQ + sq_offs_start
            sq_mask = sq_offs < sq

            # Load P_drop tile [BLOCK_SQ, BLOCK_SKV]  (transposed view: [BLOCK_SKV, BLOCK_SQ] for mm)
            Pd_base = bs_idx * Pd_stride_bs + h_idx * Pd_stride_h
            P_tile = tl.load(
                P_drop_ptr + Pd_base
                + sq_offs[:, None] * Pd_stride_sq
                + skv_offs[None, :] * Pd_stride_skv,
                mask=sq_mask[:, None] & skv_mask[None, :],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_SQ, BLOCK_SKV]

            # Load dO tile [BLOCK_SQ, BLOCK_D]
            dO_base = bs_idx * dO_stride_bs + h_idx * dO_stride_h
            dO_tile = tl.load(
                dO_ptr + dO_base
                + sq_offs[:, None] * dO_stride_sq
                + d_offs[None, :] * dO_stride_d,
                mask=sq_mask[:, None] & (d_offs[None, :] < head_dim),
                other=0.0,
            )  # float32 [BLOCK_SQ, BLOCK_D]

            # dV += P_drop^T @ dO  ->  [BLOCK_SKV, BLOCK_D]
            dV_acc += tl.dot(tl.trans(P_tile), dO_tile)

    # Write dV
    dV_base = bs_idx * dV_stride_bs + kv_idx * dV_stride_h
    tl.store(
        dV_ptr + dV_base
        + skv_offs[:, None] * dV_stride_skv
        + d_offs[None, :] * dV_stride_d,
        dV_acc.to(tl.bfloat16),
        mask=skv_mask[:, None] & (d_offs[None, :] < head_dim),
    )


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = N_GROUPS  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # Transpose dO to [bs, n_heads, sq, d] in float32 (contiguous)
    dO = grad_attn_output.transpose(1, 2).contiguous().to(torch.float32)
    # dO: [bs, 80, sq, 128]

    # Ensure inputs are contiguous
    attn_weights_c = attn_weights.contiguous()
    attn_weights_dropped_c = attn_weights_dropped.contiguous()
    value_states_c = value_states.contiguous()
    dropout_mask_c = dropout_mask.contiguous()

    # Output tensors
    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=dO.device)
    dV = torch.empty(bs, n_kv_heads, seq_kv, HEAD_DIM, dtype=torch.bfloat16, device=dO.device)

    inv_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Tile sizes for kernel 1 (dS)
    BLOCK_SQ_DS  = 16
    BLOCK_SKV_DS = 64
    BLOCK_D_DS   = 128  # head_dim is always 128

    # Grid for kernel 1: (bs * n_heads, ceil(sq / BLOCK_SQ))
    grid_ds = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ_DS))

    attn_bwd_ds_kernel[grid_ds](
        dO, value_states_c, attn_weights_c, dropout_mask_c, dS,
        bs, n_heads, seq_q, seq_kv,
        n_kv_heads, n_groups, HEAD_DIM,
        inv_scale,
        # dO strides [bs, n_heads, sq, d]
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        # V strides [bs, 8, skv, d]
        value_states_c.stride(0), value_states_c.stride(1),
        value_states_c.stride(2), value_states_c.stride(3),
        # P strides [bs, n_heads, sq, skv]
        attn_weights_c.stride(0), attn_weights_c.stride(1),
        attn_weights_c.stride(2), attn_weights_c.stride(3),
        # mask strides [bs, n_heads, sq, skv]
        dropout_mask_c.stride(0), dropout_mask_c.stride(1),
        dropout_mask_c.stride(2), dropout_mask_c.stride(3),
        # dS strides [bs, n_heads, sq, skv]
        dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
        BLOCK_SQ=BLOCK_SQ_DS,
        BLOCK_SKV=BLOCK_SKV_DS,
        BLOCK_D=BLOCK_D_DS,
        num_warps=4,
        num_stages=2,
    )

    # Tile sizes for kernel 2 (dV)
    BLOCK_SKV_DV = 64
    BLOCK_SQ_DV  = 16
    BLOCK_D_DV   = 128

    # Grid for kernel 2: (bs * n_kv_heads, ceil(skv / BLOCK_SKV))
    grid_dv = (bs * n_kv_heads, triton.cdiv(seq_kv, BLOCK_SKV_DV))

    attn_bwd_dv_kernel[grid_dv](
        attn_weights_dropped_c, dO, dV,
        bs, n_heads, seq_q, seq_kv, n_kv_heads, n_groups, HEAD_DIM,
        # P_drop strides [bs, n_heads, sq, skv]
        attn_weights_dropped_c.stride(0), attn_weights_dropped_c.stride(1),
        attn_weights_dropped_c.stride(2), attn_weights_dropped_c.stride(3),
        # dO strides [bs, n_heads, sq, d]
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        # dV strides [bs, 8, skv, d]
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        BLOCK_SKV=BLOCK_SKV_DV,
        BLOCK_SQ=BLOCK_SQ_DV,
        BLOCK_D=BLOCK_D_DV,
        num_warps=4,
        num_stages=2,
    )

    return dS, dV

```

---

## Experiment #3 — 2026-06-29 22:55:31 UTC ❌ DISCARD

**Hypothesis:** Refined the existing two-kernel approach with several targeted improvements:

**Result:** 2535.17 μs

---

## Experiment #4 — 2026-06-29 22:58:27 UTC ❌ DISCARD

**Hypothesis:** ** Replaced the two-pass Triton kernel for `dS` with a **single-pass Triton kernel** that reads precomputed `row_sum` from memory. The `row_sum = sum_skv(dP * P)` is now computed via PyTorch BMM opera

**Result:** 3808.15 μs

---

## Experiment #5 — 2026-06-29 22:59:49 UTC 💥 CRASH

**Hypothesis:** 1. Removed `attn_bwd_ds_kernel` and `attn_bwd_dv_kernel` (the two complex Triton GEMM kernels).

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #6 — 2026-06-29 23:01:06 UTC ✅ KEEP

**Hypothesis:** 1. **Step 1**: `dO = grad_attn_output.transpose(1,2).contiguous().to(float32)` — [bs, 80, sq, 128]

**Result:** 2021.69 μs

**Kernel code:**
```python
"""
Attention backward: cuBLAS batched GEMMs for matrix multiplications +
fused Triton pointwise kernel for softmax backward.

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
# Fused pointwise kernel: given dP_raw [bs, 80, sq, skv] (float32),
# dropout_mask [bs, 80, sq, skv] (bool), P [bs, 80, sq, skv] (bfloat16),
# compute dS = P * (dP - sum_skv(dP * P)) in one pass.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,      # [bs, n_heads, sq, skv]  float32
    P_ptr,       # [bs, n_heads, sq, skv]  bfloat16
    mask_ptr,    # [bs, n_heads, sq, skv]  bool
    dS_ptr,      # [bs, n_heads, sq, skv]  bfloat16  (output)
    inv_scale,   # scalar float
    sq, skv,
    # strides for all 4D tensors (same layout)
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SKV: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
):
    """
    Each program handles one (bs_idx, head_idx, sq_block) stripe.
    We load the full skv dimension (or BLOCK_SKV tiles) to compute row_sum,
    then write dS. Since skv fits in SRAM with tiling, do one pass accumulating
    row_sum and then a second pass writing output — but keep both in registers
    if possible.
    
    Actually for large skv we do two sub-passes within the kernel (no HBM spill).
    """
    pid_bh = tl.program_id(0)   # batch * n_heads flattened
    pid_sq = tl.program_id(1)   # sq tile

    # Decompose pid_bh into bs_idx and h_idx
    # (n_heads passed as a constexpr-friendly value via grid)
    n_heads = 80
    bs_idx = pid_bh // n_heads
    h_idx  = pid_bh % n_heads

    sq_start = pid_sq * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < sq

    # Base offset for this (bs, head) pair
    base = bs_idx * stride_bs + h_idx * stride_h

    # Accumulate row_sum = sum_skv(dP * P) for each sq row
    row_sum = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    skv_arange = tl.arange(0, BLOCK_SKV)

    # Pass 1: compute row_sum
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv
        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        # Load dP_raw (float32)
        dP_tile = tl.load(
            dP_ptr + base + sq_offs[:, None] * stride_sq + skv_offs[None, :] * stride_skv,
            mask=combined_mask, other=0.0,
        )  # float32

        # Load dropout mask (bool) and apply
        drop = tl.load(
            mask_ptr + base + sq_offs[:, None] * stride_sq + skv_offs[None, :] * stride_skv,
            mask=combined_mask, other=0,
        )
        dP_tile = dP_tile * drop.to(tl.float32) * inv_scale

        # Load P (bfloat16) -> float32
        P_tile = tl.load(
            P_ptr + base + sq_offs[:, None] * stride_sq + skv_offs[None, :] * stride_skv,
            mask=combined_mask, other=0.0,
        ).to(tl.float32)

        row_sum += tl.sum(dP_tile * P_tile, axis=1)

    # Pass 2: compute and store dS
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv
        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dP_tile = tl.load(
            dP_ptr + base + sq_offs[:, None] * stride_sq + skv_offs[None, :] * stride_skv,
            mask=combined_mask, other=0.0,
        )

        drop = tl.load(
            mask_ptr + base + sq_offs[:, None] * stride_sq + skv_offs[None, :] * stride_skv,
            mask=combined_mask, other=0,
        )
        dP_tile = dP_tile * drop.to(tl.float32) * inv_scale

        P_tile = tl.load(
            P_ptr + base + sq_offs[:, None] * stride_sq + skv_offs[None, :] * stride_skv,
            mask=combined_mask, other=0.0,
        ).to(tl.float32)

        dS_tile = P_tile * (dP_tile - row_sum[:, None])

        tl.store(
            dS_ptr + base + sq_offs[:, None] * stride_sq + skv_offs[None, :] * stride_skv,
            dS_tile.to(tl.bfloat16),
            mask=combined_mask,
        )


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # ----------------------------------------------------------------
    # Step 1: Prepare dO as [bs, 80, sq, 128] float32, contiguous
    # ----------------------------------------------------------------
    dO = grad_attn_output.transpose(1, 2).contiguous().to(torch.float32)
    # dO: [bs, 80, sq, 128]

    # ----------------------------------------------------------------
    # Step 2: Expand V from [bs, 8, skv, 128] -> [bs, 80, skv, 128] float32
    # Use expand (zero-copy) then contiguous to help cuBLAS
    # ----------------------------------------------------------------
    V_exp = (value_states
             .view(bs, n_kv_heads, 1, seq_kv, HEAD_DIM)
             .expand(bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM)
             .reshape(bs, n_heads, seq_kv, HEAD_DIM)
             .contiguous()
             .to(torch.float32))
    # V_exp: [bs, 80, skv, 128]

    # ----------------------------------------------------------------
    # Step 3: dP_raw = dO @ V_exp^T  -> [bs, 80, sq, skv]
    # cuBLAS batched GEMM: [bs*80, sq, 128] x [bs*80, 128, skv]
    # ----------------------------------------------------------------
    dP_raw = torch.matmul(dO, V_exp.transpose(-2, -1))
    # dP_raw: [bs, 80, sq, skv]  float32

    # ----------------------------------------------------------------
    # Step 4: Fused Triton pointwise kernel for softmax backward
    # dP = dP_raw * mask * inv_scale
    # row_sum = sum(dP * P, dim=-1)
    # dS = P * (dP - row_sum)
    # ----------------------------------------------------------------
    P = attn_weights.contiguous()
    mask_c = dropout_mask.contiguous()
    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=dO.device)

    inv_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # All tensors have the same [bs, 80, sq, skv] layout
    stride_bs  = dP_raw.stride(0)
    stride_h   = dP_raw.stride(1)
    stride_sq  = dP_raw.stride(2)
    stride_skv = dP_raw.stride(3)

    # Tile sizes: BLOCK_SQ rows processed together, BLOCK_SKV for inner loop
    BLOCK_SQ_K  = 16
    # For skv, use a tile that fits well
    # skv is typically 4096 for long sequences; loop in blocks of 256
    BLOCK_SKV_K = 256

    grid_softmax = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ_K))

    softmax_bwd_kernel[grid_softmax](
        dP_raw, P, mask_c, dS,
        inv_scale,
        seq_q, seq_kv,
        stride_bs, stride_h, stride_sq, stride_skv,
        BLOCK_SKV=BLOCK_SKV_K,
        BLOCK_SQ=BLOCK_SQ_K,
        num_warps=8,
        num_stages=3,
    )

    # ----------------------------------------------------------------
    # Step 5: dV_exp = P_drop^T @ dO  -> [bs, 80, skv, 128]
    # cuBLAS batched GEMM: [bs*80, skv, sq] x [bs*80, sq, 128]
    # Then GQA reduce: reshape to [bs, 8, 10, skv, 128] and sum over dim=2
    # ----------------------------------------------------------------
    P_drop = attn_weights_dropped.contiguous().to(torch.float32)

    dV_exp = torch.matmul(P_drop.transpose(-2, -1), dO)
    # dV_exp: [bs, 80, skv, 128]

    dV = (dV_exp
          .reshape(bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM)
          .sum(dim=2)
          .to(torch.bfloat16))
    # dV: [bs, 8, skv, 128]

    return dS, dV

```

---

## Experiment #7 — 2026-06-29 23:02:20 UTC ❌ DISCARD

**Hypothesis:** Restructured the `softmax_bwd_kernel` to use **one program per row** (bs × n_heads × seq_q) instead of the previous `(bs*n_heads, seq_q_tile)` 2D grid with BLOCK_SQ rows per program. This changes the 

**Result:** 2046.72 μs

---

## Experiment #8 — 2026-06-29 23:03:38 UTC ✅ KEEP

**Hypothesis:** **

**Result:** 851.07 μs

**Kernel code:**
```python
"""
Attention backward: GQA-native cuBLAS batched GEMMs (no V expansion) +
fused Triton pointwise kernel for softmax backward.

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
# Fused pointwise kernel: given dP_raw [bs, 80, sq, skv] (float32),
# dropout_mask [bs, 80, sq, skv] (bool), P [bs, 80, sq, skv] (bfloat16),
# compute dS = P * (dP - sum_skv(dP * P)) in one pass over rows.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,      # [bs, n_heads, sq, skv]  float32
    P_ptr,       # [bs, n_heads, sq, skv]  bfloat16
    mask_ptr,    # [bs, n_heads, sq, skv]  bool
    dS_ptr,      # [bs, n_heads, sq, skv]  bfloat16  (output)
    inv_scale,   # scalar float
    sq, skv,
    # strides for all 4D tensors (same layout)
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SKV: tl.constexpr,
):
    """
    Each program handles ONE row: (bs_idx, head_idx, sq_idx).
    We process the skv dimension in BLOCK_SKV tiles.
    Pass 1: accumulate row_sum = sum_skv(dP * P) in registers.
    Pass 2: compute dS = P * (dP - row_sum) and store.
    Both passes are sequential over HBM — but for a single row, the working
    set is small enough to stay in L2 cache between the two passes.
    """
    pid = tl.program_id(0)   # flattened (bs, head, sq) index
    n_heads = 80

    # Decompose pid into (bs_idx, h_idx, sq_idx)
    total_heads = 80
    bh_idx = pid // sq
    sq_idx = pid % sq
    bs_idx = bh_idx // total_heads
    h_idx  = bh_idx % total_heads

    # Base offset for this row
    base = bs_idx * stride_bs + h_idx * stride_h + sq_idx * stride_sq

    skv_arange = tl.arange(0, BLOCK_SKV)

    # Pass 1: compute row_sum = sum_skv(dP * P)
    row_sum = tl.zeros([1], dtype=tl.float32)

    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        dP_tile = tl.load(
            dP_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        )  # float32

        drop = tl.load(
            mask_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0,
        )
        dP_tile = dP_tile * drop.to(tl.float32) * inv_scale

        P_tile = tl.load(
            P_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        ).to(tl.float32)

        row_sum += tl.sum(dP_tile * P_tile, axis=0)

    # Pass 2: compute and store dS
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        dP_tile = tl.load(
            dP_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        )

        drop = tl.load(
            mask_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0,
        )
        dP_tile = dP_tile * drop.to(tl.float32) * inv_scale

        P_tile = tl.load(
            P_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        ).to(tl.float32)

        dS_tile = P_tile * (dP_tile - row_sum)

        tl.store(
            dS_ptr + base + skv_offs * stride_skv,
            dS_tile.to(tl.bfloat16),
            mask=skv_mask,
        )


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # ----------------------------------------------------------------
    # Step 1: Prepare dO as [bs, 80, sq, 128] bfloat16, contiguous
    # Reshape to [bs, 8, 10*sq, 128] for GQA-native BMMs
    # ----------------------------------------------------------------
    dO = grad_attn_output.transpose(1, 2).contiguous()
    # dO: [bs, 80, sq, 128]  bfloat16

    # Reshape dO to group by KV head: [bs, 8, 10*sq, 128]
    dO_grouped = dO.reshape(bs, n_kv_heads, n_groups * seq_q, HEAD_DIM)
    # dO_grouped: [bs, 8, 10*sq, 128]  bfloat16

    # ----------------------------------------------------------------
    # Step 2: dP_raw = dO_grouped @ V^T  -> [bs, 8, 10*sq, skv]
    # GQA-native: avoids 10x V expansion entirely!
    # V: [bs, 8, skv, 128]  bfloat16
    # dP_raw: [bs, 8, 10*sq, skv]  bfloat16 -> float32
    # ----------------------------------------------------------------
    V = value_states  # [bs, 8, skv, 128]  bfloat16, already contiguous

    # BMM in bfloat16: [bs, 8, 10*sq, 128] x [bs, 8, 128, skv] -> [bs, 8, 10*sq, skv]
    dP_raw_grouped = torch.matmul(dO_grouped, V.transpose(-2, -1)).to(torch.float32)
    # Reshape back to [bs, 80, sq, skv]
    dP_raw = dP_raw_grouped.reshape(bs, n_heads, seq_q, seq_kv)
    # dP_raw: [bs, 80, sq, skv]  float32

    # ----------------------------------------------------------------
    # Step 3: Fused Triton pointwise kernel for softmax backward
    # Each program handles one (bs, head, sq_row):
    #   dP = dP_raw * mask * inv_scale
    #   row_sum = sum(dP * P, dim=-1)
    #   dS = P * (dP - row_sum)
    # ----------------------------------------------------------------
    P = attn_weights.contiguous()
    mask_c = dropout_mask.contiguous()
    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=dO.device)

    inv_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # All tensors have the same [bs, 80, sq, skv] layout
    stride_bs  = dP_raw.stride(0)
    stride_h   = dP_raw.stride(1)
    stride_sq_s  = dP_raw.stride(2)
    stride_skv = dP_raw.stride(3)

    # One program per row: grid = bs * n_heads * seq_q
    BLOCK_SKV_K = 512  # 512 elements × 4 bytes = 2KB per tile, good L2 reuse

    grid_softmax = (bs * n_heads * seq_q,)

    softmax_bwd_kernel[grid_softmax](
        dP_raw, P, mask_c, dS,
        inv_scale,
        seq_q, seq_kv,
        stride_bs, stride_h, stride_sq_s, stride_skv,
        BLOCK_SKV=BLOCK_SKV_K,
        num_warps=4,
        num_stages=3,
    )

    # ----------------------------------------------------------------
    # Step 4: dV = P_drop_grouped^T @ dO_grouped  -> [bs, 8, skv, 128]
    # GQA-native: reshape P_drop to [bs, 8, 10*sq, skv], no separate sum needed!
    # cuBLAS: [bs, 8, skv, 10*sq] x [bs, 8, 10*sq, 128] -> [bs, 8, skv, 128]
    # ----------------------------------------------------------------
    # Reshape attn_weights_dropped from [bs, 80, sq, skv] -> [bs, 8, 10*sq, skv]
    P_drop_grouped = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups * seq_q, seq_kv)
    # P_drop_grouped: [bs, 8, 10*sq, skv]  bfloat16

    # BMM in bfloat16: [bs, 8, skv, 10*sq] x [bs, 8, 10*sq, 128] -> [bs, 8, skv, 128]
    dV = torch.matmul(P_drop_grouped.transpose(-2, -1), dO_grouped).to(torch.bfloat16)
    # dV: [bs, 8, skv, 128]  bfloat16

    return dS, dV

```

---

## Experiment #9 — 2026-06-29 23:05:01 UTC ✅ KEEP

**Hypothesis:** Added a single-pass Triton softmax backward kernel (`softmax_bwd_kernel_single`) that loads each data tile exactly once, computes the row_sum, and writes dS in a single sweep — eliminating the 2× HBM 

**Result:** 788.02 μs

**Kernel code:**
```python
"""
Attention backward: GQA-native cuBLAS batched GEMMs (no V expansion) +
fused Triton pointwise kernel for softmax backward.

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
N_GROUPS = 10


# ---------------------------------------------------------------------------
# Fused pointwise kernel: given dP_raw [bs, 80, sq, skv] (float32),
# dropout_mask [bs, 80, sq, skv] (bool), P [bs, 80, sq, skv] (bfloat16),
# compute dS = P * (dP - sum_skv(dP * P)) in one pass over rows.
#
# Single-pass variant: when BLOCK_SKV >= skv, we load everything once
# into registers, compute the row_sum, then compute dS in the same pass.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_kernel_single(
    dP_ptr,      # [bs, n_heads, sq, skv]  float32
    P_ptr,       # [bs, n_heads, sq, skv]  bfloat16
    mask_ptr,    # [bs, n_heads, sq, skv]  bool
    dS_ptr,      # [bs, n_heads, sq, skv]  bfloat16  (output)
    inv_scale,   # scalar float
    sq, skv,
    # strides for all 4D tensors (same layout)
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SKV: tl.constexpr,
):
    """
    Single-pass variant: load all tiles once, accumulate row_sum in registers,
    then compute dS in a second sweep over the already-loaded tiles.
    Each program handles ONE row: (bs_idx, head_idx, sq_idx).
    """
    pid = tl.program_id(0)   # flattened (bs, head, sq) index
    n_heads = 80

    bh_idx = pid // sq
    sq_idx = pid % sq
    bs_idx = bh_idx // n_heads
    h_idx  = bh_idx % n_heads

    # Base offset for this row
    base = bs_idx * stride_bs + h_idx * stride_h + sq_idx * stride_sq

    # Single tile covers the full skv dimension
    skv_offs = tl.arange(0, BLOCK_SKV)
    skv_mask = skv_offs < skv

    # Load all data once
    dP_tile = tl.load(
        dP_ptr + base + skv_offs * stride_skv,
        mask=skv_mask, other=0.0,
    )  # float32

    drop = tl.load(
        mask_ptr + base + skv_offs * stride_skv,
        mask=skv_mask, other=0,
    )
    dP_tile = dP_tile * drop.to(tl.float32) * inv_scale

    P_tile = tl.load(
        P_ptr + base + skv_offs * stride_skv,
        mask=skv_mask, other=0.0,
    ).to(tl.float32)

    # Compute row_sum and dS in single pass
    row_sum = tl.sum(dP_tile * P_tile, axis=0)
    dS_tile = P_tile * (dP_tile - row_sum)

    tl.store(
        dS_ptr + base + skv_offs * stride_skv,
        dS_tile.to(tl.bfloat16),
        mask=skv_mask,
    )


@triton.jit
def softmax_bwd_kernel(
    dP_ptr,      # [bs, n_heads, sq, skv]  float32
    P_ptr,       # [bs, n_heads, sq, skv]  bfloat16
    mask_ptr,    # [bs, n_heads, sq, skv]  bool
    dS_ptr,      # [bs, n_heads, sq, skv]  bfloat16  (output)
    inv_scale,   # scalar float
    sq, skv,
    # strides for all 4D tensors (same layout)
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SKV: tl.constexpr,
):
    """
    Two-pass variant: used when skv > BLOCK_SKV (needs multiple tiles).
    Each program handles ONE row: (bs_idx, head_idx, sq_idx).
    """
    pid = tl.program_id(0)
    n_heads = 80

    bh_idx = pid // sq
    sq_idx = pid % sq
    bs_idx = bh_idx // n_heads
    h_idx  = bh_idx % n_heads

    base = bs_idx * stride_bs + h_idx * stride_h + sq_idx * stride_sq

    skv_arange = tl.arange(0, BLOCK_SKV)

    # Pass 1: accumulate row_sum = sum_skv(dP * P)
    row_sum = tl.zeros([1], dtype=tl.float32)

    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        dP_tile = tl.load(
            dP_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        )

        drop = tl.load(
            mask_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0,
        )
        dP_tile = dP_tile * drop.to(tl.float32) * inv_scale

        P_tile = tl.load(
            P_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        ).to(tl.float32)

        row_sum += tl.sum(dP_tile * P_tile, axis=0)

    # Pass 2: compute and store dS
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        dP_tile = tl.load(
            dP_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        )

        drop = tl.load(
            mask_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0,
        )
        dP_tile = dP_tile * drop.to(tl.float32) * inv_scale

        P_tile = tl.load(
            P_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        ).to(tl.float32)

        dS_tile = P_tile * (dP_tile - row_sum)

        tl.store(
            dS_ptr + base + skv_offs * stride_skv,
            dS_tile.to(tl.bfloat16),
            mask=skv_mask,
        )


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # ----------------------------------------------------------------
    # Step 1: Prepare dO as [bs, 80, sq, 128] bfloat16, contiguous
    # Reshape to [bs, 8, 10*sq, 128] for GQA-native BMMs
    # ----------------------------------------------------------------
    dO = grad_attn_output.transpose(1, 2).contiguous()
    # dO: [bs, 80, sq, 128]  bfloat16

    # Reshape dO to group by KV head: [bs, 8, 10*sq, 128]
    dO_grouped = dO.reshape(bs, n_kv_heads, n_groups * seq_q, HEAD_DIM)
    # dO_grouped: [bs, 8, 10*sq, 128]  bfloat16

    # ----------------------------------------------------------------
    # Step 2: dP_raw = dO_grouped @ V^T  -> [bs, 8, 10*sq, skv]
    # GQA-native: avoids 10x V expansion entirely!
    # V: [bs, 8, skv, 128]  bfloat16
    # ----------------------------------------------------------------
    V = value_states  # [bs, 8, skv, 128]  bfloat16, already contiguous

    # BMM in bfloat16: [bs, 8, 10*sq, 128] x [bs, 8, 128, skv] -> [bs, 8, 10*sq, skv]
    dP_raw_grouped = torch.matmul(dO_grouped, V.transpose(-2, -1)).to(torch.float32)
    # Reshape back to [bs, 80, sq, skv] — free view since memory is contiguous
    dP_raw = dP_raw_grouped.reshape(bs, n_heads, seq_q, seq_kv)
    # dP_raw: [bs, 80, sq, skv]  float32

    # ----------------------------------------------------------------
    # Step 3: Fused Triton pointwise kernel for softmax backward
    # ----------------------------------------------------------------
    P = attn_weights  # already contiguous [bs, 80, sq, skv] bfloat16
    mask_c = dropout_mask  # already contiguous [bs, 80, sq, skv] bool
    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=dO.device)

    inv_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    stride_bs  = dP_raw.stride(0)
    stride_h   = dP_raw.stride(1)
    stride_sq_s  = dP_raw.stride(2)
    stride_skv = dP_raw.stride(3)

    grid_softmax = (bs * n_heads * seq_q,)

    # Use single-pass kernel when seq_kv fits in one tile (power-of-2 block)
    # For typical seq_kv up to 4096, use BLOCK_SKV=4096 for single-pass
    # For larger seq_kv, fall back to two-pass with BLOCK_SKV=512
    if seq_kv <= 1024:
        BLOCK_SKV_K = 1024
        softmax_bwd_kernel_single[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=4,
            num_stages=1,
        )
    elif seq_kv <= 2048:
        BLOCK_SKV_K = 2048
        softmax_bwd_kernel_single[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=8,
            num_stages=1,
        )
    elif seq_kv <= 4096:
        BLOCK_SKV_K = 4096
        softmax_bwd_kernel_single[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=16,
            num_stages=1,
        )
    else:
        BLOCK_SKV_K = 512
        softmax_bwd_kernel[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=4,
            num_stages=3,
        )

    # ----------------------------------------------------------------
    # Step 4: dV = P_drop_grouped^T @ dO_grouped  -> [bs, 8, skv, 128]
    # GQA-native: reshape P_drop to [bs, 8, 10*sq, skv], no separate sum needed!
    # cuBLAS: [bs, 8, skv, 10*sq] x [bs, 8, 10*sq, 128] -> [bs, 8, skv, 128]
    # ----------------------------------------------------------------
    # Reshape attn_weights_dropped from [bs, 80, sq, skv] -> [bs, 8, 10*sq, skv]
    P_drop_grouped = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups * seq_q, seq_kv)
    # P_drop_grouped: [bs, 8, 10*sq, skv]  bfloat16

    # BMM in bfloat16: [bs, 8, skv, 10*sq] x [bs, 8, 10*sq, 128] -> [bs, 8, skv, 128]
    dV = torch.matmul(P_drop_grouped.transpose(-2, -1), dO_grouped).to(torch.bfloat16)
    # dV: [bs, 8, skv, 128]  bfloat16

    return dS, dV

```

---

## Experiment #10 — 2026-06-29 23:06:34 UTC ✅ KEEP

**Hypothesis:** ** A targeted refinement of the existing GQA-native approach. The key change is how `dO_grouped` is constructed: instead of doing `transpose(1,2).contiguous()` to get `[bs,80,sq,128]` then reshaping t

**Result:** 786.71 μs

**Kernel code:**
```python
"""
Attention backward: GQA-native cuBLAS batched GEMMs (no V expansion) +
fused Triton pointwise kernel for softmax backward.

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
N_GROUPS = 10


# ---------------------------------------------------------------------------
# Fused pointwise kernel: given dP_raw [bs, 80, sq, skv] (float32),
# dropout_mask [bs, 80, sq, skv] (bool), P [bs, 80, sq, skv] (bfloat16),
# compute dS = P * (dP - sum_skv(dP * P)) in one pass over rows.
#
# Single-pass variant: when BLOCK_SKV >= skv, we load everything once
# into registers, compute the row_sum, then compute dS in the same pass.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_kernel_single(
    dP_ptr,      # [bs, n_heads, sq, skv]  float32
    P_ptr,       # [bs, n_heads, sq, skv]  bfloat16
    mask_ptr,    # [bs, n_heads, sq, skv]  bool
    dS_ptr,      # [bs, n_heads, sq, skv]  bfloat16  (output)
    inv_scale,   # scalar float
    sq, skv,
    # strides for all 4D tensors (same layout)
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SKV: tl.constexpr,
):
    """
    Single-pass variant: load all tiles once, accumulate row_sum in registers,
    then compute dS in a second sweep over the already-loaded tiles.
    Each program handles ONE row: (bs_idx, head_idx, sq_idx).
    """
    pid = tl.program_id(0)   # flattened (bs, head, sq) index
    n_heads = 80

    bh_idx = pid // sq
    sq_idx = pid % sq
    bs_idx = bh_idx // n_heads
    h_idx  = bh_idx % n_heads

    # Base offset for this row
    base = bs_idx * stride_bs + h_idx * stride_h + sq_idx * stride_sq

    # Single tile covers the full skv dimension
    skv_offs = tl.arange(0, BLOCK_SKV)
    skv_mask = skv_offs < skv

    # Load all data once
    dP_tile = tl.load(
        dP_ptr + base + skv_offs * stride_skv,
        mask=skv_mask, other=0.0,
    )  # float32

    drop = tl.load(
        mask_ptr + base + skv_offs * stride_skv,
        mask=skv_mask, other=0,
    )
    dP_tile = dP_tile * drop.to(tl.float32) * inv_scale

    P_tile = tl.load(
        P_ptr + base + skv_offs * stride_skv,
        mask=skv_mask, other=0.0,
    ).to(tl.float32)

    # Compute row_sum and dS in single pass
    row_sum = tl.sum(dP_tile * P_tile, axis=0)
    dS_tile = P_tile * (dP_tile - row_sum)

    tl.store(
        dS_ptr + base + skv_offs * stride_skv,
        dS_tile.to(tl.bfloat16),
        mask=skv_mask,
    )


@triton.jit
def softmax_bwd_kernel(
    dP_ptr,      # [bs, n_heads, sq, skv]  float32
    P_ptr,       # [bs, n_heads, sq, skv]  bfloat16
    mask_ptr,    # [bs, n_heads, sq, skv]  bool
    dS_ptr,      # [bs, n_heads, sq, skv]  bfloat16  (output)
    inv_scale,   # scalar float
    sq, skv,
    # strides for all 4D tensors (same layout)
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SKV: tl.constexpr,
):
    """
    Two-pass variant: used when skv > BLOCK_SKV (needs multiple tiles).
    Each program handles ONE row: (bs_idx, head_idx, sq_idx).
    """
    pid = tl.program_id(0)
    n_heads = 80

    bh_idx = pid // sq
    sq_idx = pid % sq
    bs_idx = bh_idx // n_heads
    h_idx  = bh_idx % n_heads

    base = bs_idx * stride_bs + h_idx * stride_h + sq_idx * stride_sq

    skv_arange = tl.arange(0, BLOCK_SKV)

    # Pass 1: accumulate row_sum = sum_skv(dP * P)
    row_sum = tl.zeros([1], dtype=tl.float32)

    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        dP_tile = tl.load(
            dP_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        )

        drop = tl.load(
            mask_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0,
        )
        dP_tile = dP_tile * drop.to(tl.float32) * inv_scale

        P_tile = tl.load(
            P_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        ).to(tl.float32)

        row_sum += tl.sum(dP_tile * P_tile, axis=0)

    # Pass 2: compute and store dS
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        dP_tile = tl.load(
            dP_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        )

        drop = tl.load(
            mask_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0,
        )
        dP_tile = dP_tile * drop.to(tl.float32) * inv_scale

        P_tile = tl.load(
            P_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        ).to(tl.float32)

        dS_tile = P_tile * (dP_tile - row_sum)

        tl.store(
            dS_ptr + base + skv_offs * stride_skv,
            dS_tile.to(tl.bfloat16),
            mask=skv_mask,
        )


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # ----------------------------------------------------------------
    # Step 1: Prepare dO as [bs, 8, 10*sq, 128] bfloat16, contiguous
    # Go directly from [bs, sq, 80, 128] -> [bs, 8, 10*sq, 128]
    # avoiding intermediate [bs, 80, sq, 128] materialization
    # ----------------------------------------------------------------
    # grad_attn_output: [bs, sq, 80, 128]
    # Permute to [bs, 80, sq, 128] then reshape to [bs, 8, 10*sq, 128]
    # Use permute+contiguous to get [bs, 8, 10, sq, 128] then merge dims
    dO_5d = grad_attn_output.permute(0, 2, 1, 3)  # [bs, 80, sq, 128] non-contiguous view
    # Reshape to [bs, 8, 10*sq, 128] — need contiguous first
    dO_grouped = dO_5d.contiguous().reshape(bs, n_kv_heads, n_groups * seq_q, HEAD_DIM)
    # dO_grouped: [bs, 8, 10*sq, 128]  bfloat16, contiguous

    # ----------------------------------------------------------------
    # Step 2: dP_raw = dO_grouped @ V^T  -> [bs, 8, 10*sq, skv] -> [bs, 80, sq, skv]
    # GQA-native: avoids 10x V expansion entirely!
    # V: [bs, 8, skv, 128]  bfloat16
    # ----------------------------------------------------------------
    V = value_states  # [bs, 8, skv, 128]  bfloat16, already contiguous

    # BMM: [bs, 8, 10*sq, 128] x [bs, 8, 128, skv] -> [bs, 8, 10*sq, skv]
    # Keep in bfloat16, convert to float32 after
    dP_raw_grouped = torch.matmul(dO_grouped, V.transpose(-2, -1))
    # float32 conversion via reshape: [bs, 8, 10*sq, skv] -> [bs, 80, sq, skv]
    dP_raw = dP_raw_grouped.to(torch.float32).reshape(bs, n_heads, seq_q, seq_kv)
    # dP_raw: [bs, 80, sq, skv]  float32, contiguous

    # ----------------------------------------------------------------
    # Step 3: Fused Triton pointwise kernel for softmax backward
    # ----------------------------------------------------------------
    P = attn_weights  # [bs, 80, sq, skv] bfloat16
    mask_c = dropout_mask  # [bs, 80, sq, skv] bool
    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=dO_grouped.device)

    inv_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    stride_bs    = dP_raw.stride(0)
    stride_h     = dP_raw.stride(1)
    stride_sq_s  = dP_raw.stride(2)
    stride_skv   = dP_raw.stride(3)

    grid_softmax = (bs * n_heads * seq_q,)

    # Use single-pass kernel when seq_kv fits in one tile (power-of-2 block)
    if seq_kv <= 1024:
        BLOCK_SKV_K = 1024
        softmax_bwd_kernel_single[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=4,
            num_stages=1,
        )
    elif seq_kv <= 2048:
        BLOCK_SKV_K = 2048
        softmax_bwd_kernel_single[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=8,
            num_stages=1,
        )
    elif seq_kv <= 4096:
        BLOCK_SKV_K = 4096
        softmax_bwd_kernel_single[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=16,
            num_stages=1,
        )
    else:
        BLOCK_SKV_K = 512
        softmax_bwd_kernel[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=4,
            num_stages=3,
        )

    # ----------------------------------------------------------------
    # Step 4: dV = P_drop_grouped^T @ dO_grouped  -> [bs, 8, skv, 128]
    # GQA-native: reshape P_drop to [bs, 8, 10*sq, skv], no separate sum needed!
    # cuBLAS: [bs, 8, skv, 10*sq] x [bs, 8, 10*sq, 128] -> [bs, 8, skv, 128]
    # ----------------------------------------------------------------
    # Reshape attn_weights_dropped from [bs, 80, sq, skv] -> [bs, 8, 10*sq, skv]
    # attn_weights_dropped is contiguous [bs, 80, sq, skv], reshape is a view
    P_drop_grouped = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups * seq_q, seq_kv)
    # P_drop_grouped: [bs, 8, 10*sq, skv]  bfloat16

    # BMM in bfloat16: [bs, 8, skv, 10*sq] x [bs, 8, 10*sq, 128] -> [bs, 8, skv, 128]
    dV = torch.matmul(P_drop_grouped.transpose(-2, -1), dO_grouped).to(torch.bfloat16)
    # dV: [bs, 8, skv, 128]  bfloat16

    return dS, dV

```

---

## Experiment #11 — 2026-06-29 23:08:24 UTC ❌ DISCARD

**Hypothesis:** Replaced the two-step approach (BMM matmul → float32 dP_raw → Triton softmax kernel) with two new fused Triton kernels (`fused_bwd_dS_kernel` for multi-pass and `fused_bwd_dS_kernel_single` for single

**Result:** 3353.91 μs

---

## Experiment #12 — 2026-06-29 23:09:54 UTC ❌ DISCARD

**Hypothesis:** 1. **Replaced the large BLOCK_SKV=1024 single-pass kernel** for seq_kv ≤ 1024 with a new `fused_bwd_dS_kernel_twopass` kernel that uses `BLOCK_SKV=256, BLOCK_D=32` — chunking the head_dim dimension to

**Result:** 3381.65 μs

---

## Experiment #13 — 2026-06-29 23:11:06 UTC 💥 CRASH

**Hypothesis:** ** Replaced the per-row kernel (one program per sq row) with a **multi-row program kernel** (`fused_bwd_dS_multirow`). Each program now handles `ROWS_PER_PROG` consecutive sq-rows simultaneously. The 

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #14 — 2026-06-29 23:12:42 UTC 💥 CRASH

**Hypothesis:** Replaced the two-step (BMM + Triton softmax_bwd) approach with a single `fused_ds_kernel` Triton kernel. The kernel:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #15 — 2026-06-29 23:14:17 UTC ❌ DISCARD

**Hypothesis:** ** Replaced the two-step approach (cuBLAS BMM for `dP_raw` + Triton pointwise softmax kernel) with a single `fused_ds_kernel` Triton kernel that:

**Result:** 6197.73 μs

---

## Experiment #16 — 2026-06-29 23:15:36 UTC ❌ DISCARD

**Hypothesis:** ** Replaced the per-row fused kernel (one program per sq_row using `tl.sum(dO_row * V_tile, axis=1)`) with a block-based kernel using `tl.dot` (one program per BLOCK_SQ=16 rows, using `tl.dot(dO_block

**Result:** 2246.39 μs

---

## Experiment #17 — 2026-06-29 23:17:12 UTC 💥 CRASH

**Hypothesis:** ** Modified `fused_ds_kernel` to:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #18 — 2026-06-29 23:18:26 UTC ✅ KEEP

**Hypothesis:** Changed `dP_raw` from float32 to bfloat16 throughout. The BMM result `dP_raw_grouped` now stays as bfloat16 (no `.to(torch.float32)` after the matmul). The reshape to `[bs, 80, sq, skv]` is now a bflo

**Result:** 466.23 μs

**Kernel code:**
```python
"""
Attention backward: GQA-native cuBLAS batched GEMMs (no V expansion) +
fused Triton pointwise kernel for softmax backward.

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
N_GROUPS = 10


# ---------------------------------------------------------------------------
# Fused pointwise kernel: given dP_raw [bs, 80, sq, skv] (bfloat16),
# dropout_mask [bs, 80, sq, skv] (bool), P [bs, 80, sq, skv] (bfloat16),
# compute dS = P * (dP - sum_skv(dP * P)) in one pass over rows.
#
# Single-pass variant: when BLOCK_SKV >= skv, we load everything once
# into registers, compute the row_sum, then compute dS in the same pass.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_kernel_single(
    dP_ptr,      # [bs, n_heads, sq, skv]  bfloat16
    P_ptr,       # [bs, n_heads, sq, skv]  bfloat16
    mask_ptr,    # [bs, n_heads, sq, skv]  bool
    dS_ptr,      # [bs, n_heads, sq, skv]  bfloat16  (output)
    inv_scale,   # scalar float
    sq, skv,
    # strides for all 4D tensors (same layout)
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SKV: tl.constexpr,
):
    """
    Single-pass variant: load all tiles once, accumulate row_sum in registers,
    then compute dS in a second sweep over the already-loaded tiles.
    Each program handles ONE row: (bs_idx, head_idx, sq_idx).
    dP_ptr is bfloat16 (halved bandwidth vs float32).
    """
    pid = tl.program_id(0)   # flattened (bs, head, sq) index
    n_heads = 80

    bh_idx = pid // sq
    sq_idx = pid % sq
    bs_idx = bh_idx // n_heads
    h_idx  = bh_idx % n_heads

    # Base offset for this row
    base = bs_idx * stride_bs + h_idx * stride_h + sq_idx * stride_sq

    # Single tile covers the full skv dimension
    skv_offs = tl.arange(0, BLOCK_SKV)
    skv_mask = skv_offs < skv

    # Load dP as bfloat16, convert to float32 for computation
    dP_tile = tl.load(
        dP_ptr + base + skv_offs * stride_skv,
        mask=skv_mask, other=0.0,
    ).to(tl.float32)  # bfloat16 -> float32

    drop = tl.load(
        mask_ptr + base + skv_offs * stride_skv,
        mask=skv_mask, other=0,
    )
    dP_tile = dP_tile * drop.to(tl.float32) * inv_scale

    P_tile = tl.load(
        P_ptr + base + skv_offs * stride_skv,
        mask=skv_mask, other=0.0,
    ).to(tl.float32)

    # Compute row_sum and dS in single pass
    row_sum = tl.sum(dP_tile * P_tile, axis=0)
    dS_tile = P_tile * (dP_tile - row_sum)

    tl.store(
        dS_ptr + base + skv_offs * stride_skv,
        dS_tile.to(tl.bfloat16),
        mask=skv_mask,
    )


@triton.jit
def softmax_bwd_kernel(
    dP_ptr,      # [bs, n_heads, sq, skv]  bfloat16
    P_ptr,       # [bs, n_heads, sq, skv]  bfloat16
    mask_ptr,    # [bs, n_heads, sq, skv]  bool
    dS_ptr,      # [bs, n_heads, sq, skv]  bfloat16  (output)
    inv_scale,   # scalar float
    sq, skv,
    # strides for all 4D tensors (same layout)
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SKV: tl.constexpr,
):
    """
    Two-pass variant: used when skv > BLOCK_SKV (needs multiple tiles).
    Each program handles ONE row: (bs_idx, head_idx, sq_idx).
    dP_ptr is bfloat16 (halved bandwidth vs float32).
    """
    pid = tl.program_id(0)
    n_heads = 80

    bh_idx = pid // sq
    sq_idx = pid % sq
    bs_idx = bh_idx // n_heads
    h_idx  = bh_idx % n_heads

    base = bs_idx * stride_bs + h_idx * stride_h + sq_idx * stride_sq

    skv_arange = tl.arange(0, BLOCK_SKV)

    # Pass 1: accumulate row_sum = sum_skv(dP * P)
    row_sum = tl.zeros([1], dtype=tl.float32)

    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        dP_tile = tl.load(
            dP_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        ).to(tl.float32)  # bfloat16 -> float32

        drop = tl.load(
            mask_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0,
        )
        dP_tile = dP_tile * drop.to(tl.float32) * inv_scale

        P_tile = tl.load(
            P_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        ).to(tl.float32)

        row_sum += tl.sum(dP_tile * P_tile, axis=0)

    # Pass 2: compute and store dS
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        dP_tile = tl.load(
            dP_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        ).to(tl.float32)  # bfloat16 -> float32

        drop = tl.load(
            mask_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0,
        )
        dP_tile = dP_tile * drop.to(tl.float32) * inv_scale

        P_tile = tl.load(
            P_ptr + base + skv_offs * stride_skv,
            mask=skv_mask, other=0.0,
        ).to(tl.float32)

        dS_tile = P_tile * (dP_tile - row_sum)

        tl.store(
            dS_ptr + base + skv_offs * stride_skv,
            dS_tile.to(tl.bfloat16),
            mask=skv_mask,
        )


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # ----------------------------------------------------------------
    # Step 1: Prepare dO as [bs, 8, 10*sq, 128] bfloat16, contiguous
    # Go directly from [bs, sq, 80, 128] -> [bs, 8, 10*sq, 128]
    # avoiding intermediate [bs, 80, sq, 128] materialization
    # ----------------------------------------------------------------
    # grad_attn_output: [bs, sq, 80, 128]
    # Permute to [bs, 80, sq, 128] then reshape to [bs, 8, 10*sq, 128]
    # Use permute+contiguous to get [bs, 8, 10, sq, 128] then merge dims
    dO_5d = grad_attn_output.permute(0, 2, 1, 3)  # [bs, 80, sq, 128] non-contiguous view
    # Reshape to [bs, 8, 10*sq, 128] — need contiguous first
    dO_grouped = dO_5d.contiguous().reshape(bs, n_kv_heads, n_groups * seq_q, HEAD_DIM)
    # dO_grouped: [bs, 8, 10*sq, 128]  bfloat16, contiguous

    # ----------------------------------------------------------------
    # Step 2: dP_raw = dO_grouped @ V^T  -> [bs, 8, 10*sq, skv] -> [bs, 80, sq, skv]
    # GQA-native: avoids 10x V expansion entirely!
    # V: [bs, 8, skv, 128]  bfloat16
    # Keep result in bfloat16 to halve the bandwidth for softmax backward.
    # ----------------------------------------------------------------
    V = value_states  # [bs, 8, skv, 128]  bfloat16, already contiguous

    # BMM: [bs, 8, 10*sq, 128] x [bs, 8, 128, skv] -> [bs, 8, 10*sq, skv]  bfloat16
    dP_raw_grouped = torch.matmul(dO_grouped, V.transpose(-2, -1))
    # Stay in bfloat16 — reshape is a view: [bs, 8, 10*sq, skv] -> [bs, 80, sq, skv]
    dP_raw = dP_raw_grouped.reshape(bs, n_heads, seq_q, seq_kv)
    # dP_raw: [bs, 80, sq, skv]  bfloat16, contiguous

    # ----------------------------------------------------------------
    # Step 3: Fused Triton pointwise kernel for softmax backward
    # dP_raw is now bfloat16 — halved memory bandwidth for reads
    # ----------------------------------------------------------------
    P = attn_weights  # [bs, 80, sq, skv] bfloat16
    mask_c = dropout_mask  # [bs, 80, sq, skv] bool
    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=dO_grouped.device)

    inv_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    stride_bs    = dP_raw.stride(0)
    stride_h     = dP_raw.stride(1)
    stride_sq_s  = dP_raw.stride(2)
    stride_skv   = dP_raw.stride(3)

    grid_softmax = (bs * n_heads * seq_q,)

    # Use single-pass kernel when seq_kv fits in one tile (power-of-2 block)
    if seq_kv <= 1024:
        BLOCK_SKV_K = 1024
        softmax_bwd_kernel_single[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=4,
            num_stages=1,
        )
    elif seq_kv <= 2048:
        BLOCK_SKV_K = 2048
        softmax_bwd_kernel_single[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=8,
            num_stages=1,
        )
    elif seq_kv <= 4096:
        BLOCK_SKV_K = 4096
        softmax_bwd_kernel_single[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=16,
            num_stages=1,
        )
    else:
        BLOCK_SKV_K = 512
        softmax_bwd_kernel[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=4,
            num_stages=3,
        )

    # ----------------------------------------------------------------
    # Step 4: dV = P_drop_grouped^T @ dO_grouped  -> [bs, 8, skv, 128]
    # GQA-native: reshape P_drop to [bs, 8, 10*sq, skv], no separate sum needed!
    # cuBLAS: [bs, 8, skv, 10*sq] x [bs, 8, 10*sq, 128] -> [bs, 8, skv, 128]
    # ----------------------------------------------------------------
    # Reshape attn_weights_dropped from [bs, 80, sq, skv] -> [bs, 8, 10*sq, skv]
    # attn_weights_dropped is contiguous [bs, 80, sq, skv], reshape is a view
    P_drop_grouped = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups * seq_q, seq_kv)
    # P_drop_grouped: [bs, 8, 10*sq, skv]  bfloat16

    # BMM in bfloat16: [bs, 8, skv, 10*sq] x [bs, 8, 10*sq, 128] -> [bs, 8, skv, 128]
    dV = torch.matmul(P_drop_grouped.transpose(-2, -1), dO_grouped).to(torch.bfloat16)
    # dV: [bs, 8, skv, 128]  bfloat16

    return dS, dV

```

---

## Experiment #19 — 2026-06-29 23:19:51 UTC ❌ DISCARD

**Hypothesis:** ** Two changes combined:

**Result:** 470.33 μs

---

## Experiment #20 — 2026-06-29 23:21:15 UTC ✅ KEEP

**Hypothesis:** Replaced both `softmax_bwd_kernel_single` and `softmax_bwd_kernel` with a single `softmax_bwd_multirow_kernel`. The grid is now 2D: `(bs * n_heads, cdiv(seq_q, BLOCK_SQ_K))` with `BLOCK_SQ_K=4`. Each 

**Result:** 445.16 μs

**Kernel code:**
```python
"""
Attention backward: GQA-native cuBLAS batched GEMMs (no V expansion) +
fused Triton multi-row softmax backward kernel.

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
N_GROUPS = 10


# ---------------------------------------------------------------------------
# Multi-row Triton softmax backward kernel.
# Grid: (bs * n_heads, cdiv(seq_q, BLOCK_SQ))
# Each program handles BLOCK_SQ rows of one (batch, head) pair.
# Uses tl.dot for computing row_sum = sum(dP * P, axis=skv) across BLOCK_SKV tiles.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_multirow_kernel(
    dP_ptr,      # [bs, n_heads, sq, skv]  bfloat16
    P_ptr,       # [bs, n_heads, sq, skv]  bfloat16
    mask_ptr,    # [bs, n_heads, sq, skv]  bool
    dS_ptr,      # [bs, n_heads, sq, skv]  bfloat16  (output)
    inv_scale,   # scalar float
    sq, skv,
    # strides for all 4D tensors (same layout)
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """
    Multi-row variant: each program handles BLOCK_SQ rows together.
    Grid: (bs * n_heads, cdiv(sq, BLOCK_SQ))
    
    For each block of rows, we:
    1. Pass 1: load dP and P in BLOCK_SKV tiles, compute per-row sums using tl.sum(axis=1)
    2. Pass 2: re-load tiles, compute dS = P * (dP - row_sum), store output
    """
    bh_idx  = tl.program_id(0)   # (batch, head) index
    sq_blk  = tl.program_id(1)   # which block of sq rows

    # Base offset for this (batch, head)
    bs_idx = bh_idx // 80
    h_idx  = bh_idx % 80
    base_bh = bs_idx * stride_bs + h_idx * stride_h

    # Row offsets for this program
    sq_start = sq_blk * BLOCK_SQ
    sq_offs = sq_start + tl.arange(0, BLOCK_SQ)   # [BLOCK_SQ]
    sq_mask = sq_offs < sq                          # [BLOCK_SQ]

    skv_arange = tl.arange(0, BLOCK_SKV)           # [BLOCK_SKV]

    # Pass 1: accumulate per-row sum = sum_skv(dP * P)
    # row_sum shape: [BLOCK_SQ]
    row_sum = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange   # [BLOCK_SKV]
        skv_mask = skv_offs < skv                        # [BLOCK_SKV]

        # Compute pointers: [BLOCK_SQ, BLOCK_SKV]
        # offset = base_bh + sq_offs[:, None] * stride_sq + skv_offs[None, :] * stride_skv
        ptrs = (base_bh
                + sq_offs[:, None] * stride_sq
                + skv_offs[None, :] * stride_skv)
        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dP_tile = tl.load(dP_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)
        drop    = tl.load(mask_ptr + ptrs, mask=combined_mask, other=0).to(tl.float32)
        dP_tile = dP_tile * drop * inv_scale

        P_tile  = tl.load(P_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # Per-row sum: sum along skv axis -> [BLOCK_SQ]
        row_sum += tl.sum(dP_tile * P_tile, axis=1)

    # Pass 2: compute dS and store
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        ptrs = (base_bh
                + sq_offs[:, None] * stride_sq
                + skv_offs[None, :] * stride_skv)
        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dP_tile = tl.load(dP_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)
        drop    = tl.load(mask_ptr + ptrs, mask=combined_mask, other=0).to(tl.float32)
        dP_tile = dP_tile * drop * inv_scale

        P_tile  = tl.load(P_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # dS = P * (dP - row_sum[:, None])  broadcast row_sum over skv
        dS_tile = P_tile * (dP_tile - row_sum[:, None])

        tl.store(dS_ptr + ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # ----------------------------------------------------------------
    # Step 1: Prepare dO as [bs, 8, 10*sq, 128] bfloat16, contiguous
    # grad_attn_output: [bs, sq, 80, 128]
    # ----------------------------------------------------------------
    dO_5d = grad_attn_output.reshape(bs, seq_q, n_kv_heads, n_groups, HEAD_DIM)
    dO_5d_perm2 = dO_5d.permute(0, 2, 3, 1, 4)  # [bs, 8, 10, sq, 128]
    dO_grouped = dO_5d_perm2.contiguous().reshape(bs, n_kv_heads, n_groups * seq_q, HEAD_DIM)
    # dO_grouped: [bs, 8, 10*sq, 128]  bfloat16, contiguous

    # ----------------------------------------------------------------
    # Step 2: dP_raw = dO_grouped @ V^T  -> [bs, 8, 10*sq, skv] -> [bs, 80, sq, skv]
    # GQA-native: avoids 10x V expansion entirely!
    # ----------------------------------------------------------------
    V = value_states  # [bs, 8, skv, 128]  bfloat16, already contiguous

    # BMM: [bs, 8, 10*sq, 128] x [bs, 8, 128, skv] -> [bs, 8, 10*sq, skv]  bfloat16
    dP_raw_grouped = torch.matmul(dO_grouped, V.transpose(-2, -1))
    # Reshape is a view: [bs, 8, 10*sq, skv] -> [bs, 80, sq, skv]
    dP_raw = dP_raw_grouped.reshape(bs, n_heads, seq_q, seq_kv)
    # dP_raw: [bs, 80, sq, skv]  bfloat16, contiguous

    # ----------------------------------------------------------------
    # Step 3: Multi-row Triton kernel for softmax backward
    # Grid: (bs * n_heads, cdiv(seq_q, BLOCK_SQ))
    # Each program handles BLOCK_SQ rows for one (batch, head) pair.
    # ----------------------------------------------------------------
    P = attn_weights          # [bs, 80, sq, skv] bfloat16
    mask_c = dropout_mask     # [bs, 80, sq, skv] bool
    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=dO_grouped.device)

    inv_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    stride_bs    = dP_raw.stride(0)
    stride_h     = dP_raw.stride(1)
    stride_sq_s  = dP_raw.stride(2)
    stride_skv   = dP_raw.stride(3)

    # Choose BLOCK_SQ and BLOCK_SKV based on seq_kv
    # Larger BLOCK_SQ = better SM utilization, but registers scale as BLOCK_SQ * BLOCK_SKV
    BLOCK_SQ_K = 4
    if seq_kv <= 512:
        BLOCK_SKV_K = 512
        NW = 4
    elif seq_kv <= 1024:
        BLOCK_SKV_K = 1024
        NW = 8
    elif seq_kv <= 2048:
        BLOCK_SKV_K = 2048
        NW = 16
    else:
        BLOCK_SKV_K = 2048
        NW = 16

    grid_softmax = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ_K))

    softmax_bwd_multirow_kernel[grid_softmax](
        dP_raw, P, mask_c, dS,
        inv_scale,
        seq_q, seq_kv,
        stride_bs, stride_h, stride_sq_s, stride_skv,
        BLOCK_SQ=BLOCK_SQ_K,
        BLOCK_SKV=BLOCK_SKV_K,
        num_warps=NW,
        num_stages=1,
    )

    # ----------------------------------------------------------------
    # Step 4: dV = P_drop_grouped^T @ dO_grouped  -> [bs, 8, skv, 128]
    # GQA-native: reshape P_drop to [bs, 8, 10*sq, skv], no separate sum needed!
    # ----------------------------------------------------------------
    P_drop_grouped = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups * seq_q, seq_kv)
    # BMM in bfloat16: [bs, 8, skv, 10*sq] x [bs, 8, 10*sq, 128] -> [bs, 8, skv, 128]
    dV = torch.matmul(P_drop_grouped.transpose(-2, -1), dO_grouped).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #21 — 2026-06-29 23:22:24 UTC ❌ DISCARD

**Hypothesis:** Tuned `BLOCK_SQ` from a fixed `4` to adaptive values based on `seq_kv`: use `BLOCK_SQ=8` for smaller seq_kv (≤1024) and `BLOCK_SQ=4` for larger (≤2048+). This doubles the number of rows processed per 

**Result:** 449.48 μs

---

## Experiment #22 — 2026-06-29 23:23:48 UTC ✅ KEEP

**Hypothesis:** Added a **single-pass softmax backward kernel** (`softmax_bwd_singlepass_kernel`) that handles the entire `skv` dimension in one tile when `seq_kv <= 4096`. This eliminates the two-pass structure (loa

**Result:** 439.52 μs

**Kernel code:**
```python
"""
Attention backward: GQA-native cuBLAS batched GEMMs (no V expansion) +
fused Triton multi-row softmax backward kernel with tl.dot-based accumulation.

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
N_GROUPS = 10


# ---------------------------------------------------------------------------
# Multi-row Triton softmax backward kernel.
# Grid: (bs * n_heads, cdiv(seq_q, BLOCK_SQ))
# Each program handles BLOCK_SQ rows of one (batch, head) pair.
# Two-pass: pass1 accumulates per-row sums, pass2 computes dS and stores.
# Uses tl.dot for higher arithmetic intensity on the inner product computation.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_multirow_kernel(
    dP_ptr,      # [bs, n_heads, sq, skv]  bfloat16
    P_ptr,       # [bs, n_heads, sq, skv]  bfloat16
    mask_ptr,    # [bs, n_heads, sq, skv]  bool
    dS_ptr,      # [bs, n_heads, sq, skv]  bfloat16  (output)
    inv_scale,   # scalar float
    sq, skv,
    # strides for all 4D tensors (same layout)
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    bh_idx  = tl.program_id(0)   # (batch, head) index
    sq_blk  = tl.program_id(1)   # which block of sq rows

    # Base offset for this (batch, head)
    bs_idx = bh_idx // 80
    h_idx  = bh_idx % 80
    base_bh = bs_idx * stride_bs + h_idx * stride_h

    # Row offsets for this program
    sq_start = sq_blk * BLOCK_SQ
    sq_offs = sq_start + tl.arange(0, BLOCK_SQ)   # [BLOCK_SQ]
    sq_mask = sq_offs < sq                          # [BLOCK_SQ]

    skv_arange = tl.arange(0, BLOCK_SKV)           # [BLOCK_SKV]

    # Pass 1: accumulate per-row sum = sum_skv(dP * P) using tl.dot
    # dP_tile [BLOCK_SQ, BLOCK_SKV], P_tile [BLOCK_SQ, BLOCK_SKV]
    # row_sum[i] = sum_j (dP[i,j] * P[i,j])
    # We compute this as: (dP * P) @ ones_vec, i.e., row reduce
    # Use tl.dot with P_tile.T: [BLOCK_SKV, BLOCK_SQ] x ones would be col-sum.
    # Instead, use element-wise * then tl.sum(..., axis=1).
    row_sum = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange   # [BLOCK_SKV]
        skv_mask = skv_offs < skv                        # [BLOCK_SKV]

        ptrs = (base_bh
                + sq_offs[:, None] * stride_sq
                + skv_offs[None, :] * stride_skv)
        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dP_tile = tl.load(dP_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)
        drop    = tl.load(mask_ptr + ptrs, mask=combined_mask, other=0).to(tl.float32)
        dP_tile = dP_tile * drop * inv_scale

        P_tile  = tl.load(P_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # Per-row sum: use tl.dot to compute [BLOCK_SQ, BLOCK_SKV] x [BLOCK_SKV, 1]
        # but instead use tl.sum for correctness with masking
        row_sum += tl.sum(dP_tile * P_tile, axis=1)

    # Pass 2: compute dS and store
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        ptrs = (base_bh
                + sq_offs[:, None] * stride_sq
                + skv_offs[None, :] * stride_skv)
        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dP_tile = tl.load(dP_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)
        drop    = tl.load(mask_ptr + ptrs, mask=combined_mask, other=0).to(tl.float32)
        dP_tile = dP_tile * drop * inv_scale

        P_tile  = tl.load(P_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # dS = P * (dP - row_sum[:, None])  broadcast row_sum over skv
        dS_tile = P_tile * (dP_tile - row_sum[:, None])

        tl.store(dS_ptr + ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


# ---------------------------------------------------------------------------
# Single-pass softmax backward kernel: accumulate row_sum and write dS
# in a single loop by buffering all tiles.
# Only feasible when BLOCK_SKV covers the entire skv dimension.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_singlepass_kernel(
    dP_ptr,
    P_ptr,
    mask_ptr,
    dS_ptr,
    inv_scale,
    sq, skv,
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """Single-pass version when skv fits in BLOCK_SKV (power-of-2, <= 4096)."""
    bh_idx  = tl.program_id(0)
    sq_blk  = tl.program_id(1)

    bs_idx = bh_idx // 80
    h_idx  = bh_idx % 80
    base_bh = bs_idx * stride_bs + h_idx * stride_h

    sq_start = sq_blk * BLOCK_SQ
    sq_offs = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask = sq_offs < sq

    skv_offs = tl.arange(0, BLOCK_SKV)
    skv_mask = skv_offs < skv

    ptrs = (base_bh
            + sq_offs[:, None] * stride_sq
            + skv_offs[None, :] * stride_skv)
    combined_mask = sq_mask[:, None] & skv_mask[None, :]

    # Load dP_raw and apply dropout mask + scale
    dP_tile = tl.load(dP_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)
    drop    = tl.load(mask_ptr + ptrs, mask=combined_mask, other=0).to(tl.float32)
    dP_tile = dP_tile * drop * inv_scale

    # Load P
    P_tile  = tl.load(P_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)

    # Compute row sum and dS in one pass
    row_sum = tl.sum(dP_tile * P_tile, axis=1)   # [BLOCK_SQ]
    dS_tile = P_tile * (dP_tile - row_sum[:, None])

    tl.store(dS_ptr + ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # ----------------------------------------------------------------
    # Step 1: Prepare dO as [bs, 8, 10*sq, 128] bfloat16, contiguous
    # grad_attn_output: [bs, sq, 80, 128]
    # ----------------------------------------------------------------
    dO_5d = grad_attn_output.reshape(bs, seq_q, n_kv_heads, n_groups, HEAD_DIM)
    dO_5d_perm2 = dO_5d.permute(0, 2, 3, 1, 4)  # [bs, 8, 10, sq, 128]
    dO_grouped = dO_5d_perm2.contiguous().reshape(bs, n_kv_heads, n_groups * seq_q, HEAD_DIM)
    # dO_grouped: [bs, 8, 10*sq, 128]  bfloat16, contiguous

    # ----------------------------------------------------------------
    # Step 2: dP_raw = dO_grouped @ V^T  -> [bs, 8, 10*sq, skv] -> [bs, 80, sq, skv]
    # GQA-native: avoids 10x V expansion entirely!
    # ----------------------------------------------------------------
    V = value_states  # [bs, 8, skv, 128]  bfloat16, already contiguous

    # BMM: [bs, 8, 10*sq, 128] x [bs, 8, 128, skv] -> [bs, 8, 10*sq, skv]  bfloat16
    dP_raw_grouped = torch.matmul(dO_grouped, V.transpose(-2, -1))
    # Reshape is a view: [bs, 8, 10*sq, skv] -> [bs, 80, sq, skv]
    dP_raw = dP_raw_grouped.reshape(bs, n_heads, seq_q, seq_kv)
    # dP_raw: [bs, 80, sq, skv]  bfloat16, contiguous

    # ----------------------------------------------------------------
    # Step 3: Triton kernel for softmax backward
    # ----------------------------------------------------------------
    P = attn_weights          # [bs, 80, sq, skv] bfloat16
    mask_c = dropout_mask     # [bs, 80, sq, skv] bool
    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=dO_grouped.device)

    inv_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    stride_bs    = dP_raw.stride(0)
    stride_h     = dP_raw.stride(1)
    stride_sq_s  = dP_raw.stride(2)
    stride_skv_s = dP_raw.stride(3)

    # Use single-pass kernel when skv fits in a power-of-2 block (most common cases)
    # Otherwise fall back to multi-row two-pass kernel
    if seq_kv <= 512:
        BLOCK_SQ_K  = 16
        BLOCK_SKV_K = 512
        NW = 4
        use_single = True
    elif seq_kv <= 1024:
        BLOCK_SQ_K  = 16
        BLOCK_SKV_K = 1024
        NW = 8
        use_single = True
    elif seq_kv <= 2048:
        BLOCK_SQ_K  = 8
        BLOCK_SKV_K = 2048
        NW = 16
        use_single = True
    elif seq_kv <= 4096:
        BLOCK_SQ_K  = 4
        BLOCK_SKV_K = 4096
        NW = 16
        use_single = True
    else:
        BLOCK_SQ_K  = 4
        BLOCK_SKV_K = 2048
        NW = 16
        use_single = False

    grid_softmax = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ_K))

    if use_single:
        softmax_bwd_singlepass_kernel[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv_s,
            BLOCK_SQ=BLOCK_SQ_K,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=NW,
            num_stages=1,
        )
    else:
        softmax_bwd_multirow_kernel[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv_s,
            BLOCK_SQ=BLOCK_SQ_K,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=NW,
            num_stages=1,
        )

    # ----------------------------------------------------------------
    # Step 4: dV = P_drop_grouped^T @ dO_grouped  -> [bs, 8, skv, 128]
    # GQA-native: reshape P_drop to [bs, 8, 10*sq, skv], no separate sum needed!
    # ----------------------------------------------------------------
    P_drop_grouped = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups * seq_q, seq_kv)
    # BMM in bfloat16: [bs, 8, skv, 10*sq] x [bs, 8, 10*sq, 128] -> [bs, 8, skv, 128]
    dV = torch.matmul(P_drop_grouped.transpose(-2, -1), dO_grouped).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #23 — 2026-06-29 23:25:25 UTC ✅ KEEP

**Hypothesis:** ** Added two persistent CUDA streams (`_stream1`, `_stream2`) created once via `_get_streams()`. The BMM for `dP_raw` runs on `stream1` and the BMM for `dV` runs on `stream2` simultaneously. The curre

**Result:** 432.00 μs

**Kernel code:**
```python
"""
Attention backward: GQA-native cuBLAS batched GEMMs (no V expansion) +
fused Triton multi-row softmax backward kernel with tl.dot-based accumulation.

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
N_GROUPS = 10

# Pre-create persistent CUDA streams for concurrent BMM execution
_stream1 = None
_stream2 = None

def _get_streams():
    global _stream1, _stream2
    if _stream1 is None:
        _stream1 = torch.cuda.Stream()
        _stream2 = torch.cuda.Stream()
    return _stream1, _stream2


# ---------------------------------------------------------------------------
# Multi-row Triton softmax backward kernel.
# Grid: (bs * n_heads, cdiv(seq_q, BLOCK_SQ))
# Each program handles BLOCK_SQ rows of one (batch, head) pair.
# Two-pass: pass1 accumulates per-row sums, pass2 computes dS and stores.
# Uses tl.dot for higher arithmetic intensity on the inner product computation.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_multirow_kernel(
    dP_ptr,      # [bs, n_heads, sq, skv]  bfloat16
    P_ptr,       # [bs, n_heads, sq, skv]  bfloat16
    mask_ptr,    # [bs, n_heads, sq, skv]  bool
    dS_ptr,      # [bs, n_heads, sq, skv]  bfloat16  (output)
    inv_scale,   # scalar float
    sq, skv,
    # strides for all 4D tensors (same layout)
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    bh_idx  = tl.program_id(0)   # (batch, head) index
    sq_blk  = tl.program_id(1)   # which block of sq rows

    # Base offset for this (batch, head)
    bs_idx = bh_idx // 80
    h_idx  = bh_idx % 80
    base_bh = bs_idx * stride_bs + h_idx * stride_h

    # Row offsets for this program
    sq_start = sq_blk * BLOCK_SQ
    sq_offs = sq_start + tl.arange(0, BLOCK_SQ)   # [BLOCK_SQ]
    sq_mask = sq_offs < sq                          # [BLOCK_SQ]

    skv_arange = tl.arange(0, BLOCK_SKV)           # [BLOCK_SKV]

    # Pass 1: accumulate per-row sum = sum_skv(dP * P) using tl.dot
    # dP_tile [BLOCK_SQ, BLOCK_SKV], P_tile [BLOCK_SQ, BLOCK_SKV]
    # row_sum[i] = sum_j (dP[i,j] * P[i,j])
    # We compute this as: (dP * P) @ ones_vec, i.e., row reduce
    # Use tl.dot with P_tile.T: [BLOCK_SKV, BLOCK_SQ] x ones would be col-sum.
    # Instead, use element-wise * then tl.sum(..., axis=1).
    row_sum = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange   # [BLOCK_SKV]
        skv_mask = skv_offs < skv                        # [BLOCK_SKV]

        ptrs = (base_bh
                + sq_offs[:, None] * stride_sq
                + skv_offs[None, :] * stride_skv)
        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dP_tile = tl.load(dP_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)
        drop    = tl.load(mask_ptr + ptrs, mask=combined_mask, other=0).to(tl.float32)
        dP_tile = dP_tile * drop * inv_scale

        P_tile  = tl.load(P_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # Per-row sum: use tl.dot to compute [BLOCK_SQ, BLOCK_SKV] x [BLOCK_SKV, 1]
        # but instead use tl.sum for correctness with masking
        row_sum += tl.sum(dP_tile * P_tile, axis=1)

    # Pass 2: compute dS and store
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        ptrs = (base_bh
                + sq_offs[:, None] * stride_sq
                + skv_offs[None, :] * stride_skv)
        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dP_tile = tl.load(dP_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)
        drop    = tl.load(mask_ptr + ptrs, mask=combined_mask, other=0).to(tl.float32)
        dP_tile = dP_tile * drop * inv_scale

        P_tile  = tl.load(P_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # dS = P * (dP - row_sum[:, None])  broadcast row_sum over skv
        dS_tile = P_tile * (dP_tile - row_sum[:, None])

        tl.store(dS_ptr + ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


# ---------------------------------------------------------------------------
# Single-pass softmax backward kernel: accumulate row_sum and write dS
# in a single loop by buffering all tiles.
# Only feasible when BLOCK_SKV covers the entire skv dimension.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_singlepass_kernel(
    dP_ptr,
    P_ptr,
    mask_ptr,
    dS_ptr,
    inv_scale,
    sq, skv,
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """Single-pass version when skv fits in BLOCK_SKV (power-of-2, <= 4096)."""
    bh_idx  = tl.program_id(0)
    sq_blk  = tl.program_id(1)

    bs_idx = bh_idx // 80
    h_idx  = bh_idx % 80
    base_bh = bs_idx * stride_bs + h_idx * stride_h

    sq_start = sq_blk * BLOCK_SQ
    sq_offs = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask = sq_offs < sq

    skv_offs = tl.arange(0, BLOCK_SKV)
    skv_mask = skv_offs < skv

    ptrs = (base_bh
            + sq_offs[:, None] * stride_sq
            + skv_offs[None, :] * stride_skv)
    combined_mask = sq_mask[:, None] & skv_mask[None, :]

    # Load dP_raw and apply dropout mask + scale
    dP_tile = tl.load(dP_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)
    drop    = tl.load(mask_ptr + ptrs, mask=combined_mask, other=0).to(tl.float32)
    dP_tile = dP_tile * drop * inv_scale

    # Load P
    P_tile  = tl.load(P_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)

    # Compute row sum and dS in one pass
    row_sum = tl.sum(dP_tile * P_tile, axis=1)   # [BLOCK_SQ]
    dS_tile = P_tile * (dP_tile - row_sum[:, None])

    tl.store(dS_ptr + ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # ----------------------------------------------------------------
    # Step 1: Prepare dO as [bs, 8, 10*sq, 128] bfloat16, contiguous
    # grad_attn_output: [bs, sq, 80, 128]
    # ----------------------------------------------------------------
    dO_5d = grad_attn_output.reshape(bs, seq_q, n_kv_heads, n_groups, HEAD_DIM)
    dO_5d_perm2 = dO_5d.permute(0, 2, 3, 1, 4)  # [bs, 8, 10, sq, 128]
    dO_grouped = dO_5d_perm2.contiguous().reshape(bs, n_kv_heads, n_groups * seq_q, HEAD_DIM)
    # dO_grouped: [bs, 8, 10*sq, 128]  bfloat16, contiguous

    # ----------------------------------------------------------------
    # Steps 2 & 4 (concurrent on separate CUDA streams):
    #   Stream 1: dP_raw = dO_grouped @ V^T  -> [bs, 8, 10*sq, skv] -> [bs, 80, sq, skv]
    #   Stream 2: dV = P_drop_grouped^T @ dO_grouped  -> [bs, 8, skv, 128]
    # ----------------------------------------------------------------
    V = value_states  # [bs, 8, skv, 128]  bfloat16
    P_drop_grouped = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups * seq_q, seq_kv)

    stream1, stream2 = _get_streams()
    current_stream = torch.cuda.current_stream()

    # Both streams must wait for the current stream to finish producing inputs
    stream1.wait_stream(current_stream)
    stream2.wait_stream(current_stream)

    # Launch BMM for dP_raw on stream 1
    with torch.cuda.stream(stream1):
        dP_raw_grouped = torch.matmul(dO_grouped, V.transpose(-2, -1))
        dP_raw = dP_raw_grouped.reshape(bs, n_heads, seq_q, seq_kv)

    # Launch BMM for dV on stream 2
    with torch.cuda.stream(stream2):
        dV = torch.matmul(P_drop_grouped.transpose(-2, -1), dO_grouped).to(torch.bfloat16)

    # Wait for stream 1 (dP_raw needed for softmax kernel on current stream)
    current_stream.wait_stream(stream1)

    # ----------------------------------------------------------------
    # Step 3: Triton kernel for softmax backward (on current stream)
    # dP_raw: [bs, 80, sq, skv]  bfloat16
    # ----------------------------------------------------------------
    P = attn_weights          # [bs, 80, sq, skv] bfloat16
    mask_c = dropout_mask     # [bs, 80, sq, skv] bool
    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=dO_grouped.device)

    inv_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    stride_bs    = dP_raw.stride(0)
    stride_h     = dP_raw.stride(1)
    stride_sq_s  = dP_raw.stride(2)
    stride_skv_s = dP_raw.stride(3)

    # Use single-pass kernel when skv fits in a power-of-2 block (most common cases)
    # Otherwise fall back to multi-row two-pass kernel
    if seq_kv <= 512:
        BLOCK_SQ_K  = 16
        BLOCK_SKV_K = 512
        NW = 4
        use_single = True
    elif seq_kv <= 1024:
        BLOCK_SQ_K  = 16
        BLOCK_SKV_K = 1024
        NW = 8
        use_single = True
    elif seq_kv <= 2048:
        BLOCK_SQ_K  = 8
        BLOCK_SKV_K = 2048
        NW = 16
        use_single = True
    elif seq_kv <= 4096:
        BLOCK_SQ_K  = 4
        BLOCK_SKV_K = 4096
        NW = 16
        use_single = True
    else:
        BLOCK_SQ_K  = 4
        BLOCK_SKV_K = 2048
        NW = 16
        use_single = False

    grid_softmax = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ_K))

    if use_single:
        softmax_bwd_singlepass_kernel[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv_s,
            BLOCK_SQ=BLOCK_SQ_K,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=NW,
            num_stages=1,
        )
    else:
        softmax_bwd_multirow_kernel[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv_s,
            BLOCK_SQ=BLOCK_SQ_K,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=NW,
            num_stages=1,
        )

    # Wait for stream 2 (dV) to finish before returning
    current_stream.wait_stream(stream2)

    return dS, dV

```

---

## Experiment #24 — 2026-06-29 23:27:00 UTC ✅ KEEP

**Hypothesis:** ** Replaced the 5D permute path (`reshape → permute(0,2,3,1,4) → contiguous → reshape`) with a simpler 4D path (`permute(0,2,1,3) → contiguous → view`). This achieves the same result — `dO_grouped` as

**Result:** 431.32 μs

**Kernel code:**
```python
"""
Attention backward: GQA-native cuBLAS batched GEMMs (no V expansion) +
fused Triton multi-row softmax backward kernel with tl.dot-based accumulation.

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
N_GROUPS = 10

# Pre-create persistent CUDA streams for concurrent BMM execution
_stream1 = None
_stream2 = None

def _get_streams():
    global _stream1, _stream2
    if _stream1 is None:
        _stream1 = torch.cuda.Stream()
        _stream2 = torch.cuda.Stream()
    return _stream1, _stream2


# ---------------------------------------------------------------------------
# Multi-row Triton softmax backward kernel.
# Grid: (bs * n_heads, cdiv(seq_q, BLOCK_SQ))
# Each program handles BLOCK_SQ rows of one (batch, head) pair.
# Two-pass: pass1 accumulates per-row sums, pass2 computes dS and stores.
# Uses tl.dot for higher arithmetic intensity on the inner product computation.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_multirow_kernel(
    dP_ptr,      # [bs, n_heads, sq, skv]  bfloat16
    P_ptr,       # [bs, n_heads, sq, skv]  bfloat16
    mask_ptr,    # [bs, n_heads, sq, skv]  bool
    dS_ptr,      # [bs, n_heads, sq, skv]  bfloat16  (output)
    inv_scale,   # scalar float
    sq, skv,
    # strides for all 4D tensors (same layout)
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    bh_idx  = tl.program_id(0)   # (batch, head) index
    sq_blk  = tl.program_id(1)   # which block of sq rows

    # Base offset for this (batch, head)
    bs_idx = bh_idx // 80
    h_idx  = bh_idx % 80
    base_bh = bs_idx * stride_bs + h_idx * stride_h

    # Row offsets for this program
    sq_start = sq_blk * BLOCK_SQ
    sq_offs = sq_start + tl.arange(0, BLOCK_SQ)   # [BLOCK_SQ]
    sq_mask = sq_offs < sq                          # [BLOCK_SQ]

    skv_arange = tl.arange(0, BLOCK_SKV)           # [BLOCK_SKV]

    # Pass 1: accumulate per-row sum = sum_skv(dP * P) using tl.dot
    # dP_tile [BLOCK_SQ, BLOCK_SKV], P_tile [BLOCK_SQ, BLOCK_SKV]
    # row_sum[i] = sum_j (dP[i,j] * P[i,j])
    # We compute this as: (dP * P) @ ones_vec, i.e., row reduce
    # Use tl.dot with P_tile.T: [BLOCK_SKV, BLOCK_SQ] x ones would be col-sum.
    # Instead, use element-wise * then tl.sum(..., axis=1).
    row_sum = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange   # [BLOCK_SKV]
        skv_mask = skv_offs < skv                        # [BLOCK_SKV]

        ptrs = (base_bh
                + sq_offs[:, None] * stride_sq
                + skv_offs[None, :] * stride_skv)
        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dP_tile = tl.load(dP_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)
        drop    = tl.load(mask_ptr + ptrs, mask=combined_mask, other=0).to(tl.float32)
        dP_tile = dP_tile * drop * inv_scale

        P_tile  = tl.load(P_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # Per-row sum: use tl.dot to compute [BLOCK_SQ, BLOCK_SKV] x [BLOCK_SKV, 1]
        # but instead use tl.sum for correctness with masking
        row_sum += tl.sum(dP_tile * P_tile, axis=1)

    # Pass 2: compute dS and store
    for skv_block in range(0, tl.cdiv(skv, BLOCK_SKV)):
        skv_offs = skv_block * BLOCK_SKV + skv_arange
        skv_mask = skv_offs < skv

        ptrs = (base_bh
                + sq_offs[:, None] * stride_sq
                + skv_offs[None, :] * stride_skv)
        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dP_tile = tl.load(dP_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)
        drop    = tl.load(mask_ptr + ptrs, mask=combined_mask, other=0).to(tl.float32)
        dP_tile = dP_tile * drop * inv_scale

        P_tile  = tl.load(P_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # dS = P * (dP - row_sum[:, None])  broadcast row_sum over skv
        dS_tile = P_tile * (dP_tile - row_sum[:, None])

        tl.store(dS_ptr + ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


# ---------------------------------------------------------------------------
# Single-pass softmax backward kernel: accumulate row_sum and write dS
# in a single loop by buffering all tiles.
# Only feasible when BLOCK_SKV covers the entire skv dimension.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_singlepass_kernel(
    dP_ptr,
    P_ptr,
    mask_ptr,
    dS_ptr,
    inv_scale,
    sq, skv,
    stride_bs, stride_h, stride_sq, stride_skv,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """Single-pass version when skv fits in BLOCK_SKV (power-of-2, <= 4096)."""
    bh_idx  = tl.program_id(0)
    sq_blk  = tl.program_id(1)

    bs_idx = bh_idx // 80
    h_idx  = bh_idx % 80
    base_bh = bs_idx * stride_bs + h_idx * stride_h

    sq_start = sq_blk * BLOCK_SQ
    sq_offs = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask = sq_offs < sq

    skv_offs = tl.arange(0, BLOCK_SKV)
    skv_mask = skv_offs < skv

    ptrs = (base_bh
            + sq_offs[:, None] * stride_sq
            + skv_offs[None, :] * stride_skv)
    combined_mask = sq_mask[:, None] & skv_mask[None, :]

    # Load dP_raw and apply dropout mask + scale
    dP_tile = tl.load(dP_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)
    drop    = tl.load(mask_ptr + ptrs, mask=combined_mask, other=0).to(tl.float32)
    dP_tile = dP_tile * drop * inv_scale

    # Load P
    P_tile  = tl.load(P_ptr + ptrs, mask=combined_mask, other=0.0).to(tl.float32)

    # Compute row sum and dS in one pass
    row_sum = tl.sum(dP_tile * P_tile, axis=1)   # [BLOCK_SQ]
    dS_tile = P_tile * (dP_tile - row_sum[:, None])

    tl.store(dS_ptr + ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # ----------------------------------------------------------------
    # Step 1: Prepare dO as [bs, 8, 10*sq, 128] bfloat16
    # grad_attn_output: [bs, sq, 80, 128]  (contiguous)
    # We do a single .contiguous() on the 4D transposed layout [bs, 80, sq, 128]
    # then reinterpret as [bs, 8, 10*sq, 128] with a zero-copy .view().
    # This avoids the 5D permute path's extra complexity.
    # ----------------------------------------------------------------
    # [bs, sq, 80, 128] -> [bs, 80, sq, 128] (non-contiguous) -> contiguous
    dO_4d = grad_attn_output.permute(0, 2, 1, 3).contiguous()  # [bs, 80, sq, 128]
    # Reinterpret as [bs, 8, 10*sq, 128] zero-copy view
    dO_grouped = dO_4d.view(bs, n_kv_heads, n_groups * seq_q, HEAD_DIM)
    # dO_grouped: [bs, 8, 10*sq, 128]  bfloat16, contiguous

    # ----------------------------------------------------------------
    # Steps 2 & 4 (concurrent on separate CUDA streams):
    #   Stream 1: dP_raw = dO_grouped @ V^T  -> [bs, 8, 10*sq, skv] -> [bs, 80, sq, skv]
    #   Stream 2: dV = P_drop_grouped^T @ dO_grouped  -> [bs, 8, skv, 128]
    # ----------------------------------------------------------------
    V = value_states  # [bs, 8, skv, 128]  bfloat16
    P_drop_grouped = attn_weights_dropped.view(bs, n_kv_heads, n_groups * seq_q, seq_kv)

    stream1, stream2 = _get_streams()
    current_stream = torch.cuda.current_stream()

    # Both streams must wait for the current stream to finish producing inputs
    stream1.wait_stream(current_stream)
    stream2.wait_stream(current_stream)

    # Launch BMM for dP_raw on stream 1
    with torch.cuda.stream(stream1):
        dP_raw_grouped = torch.matmul(dO_grouped, V.transpose(-2, -1))
        dP_raw = dP_raw_grouped.view(bs, n_heads, seq_q, seq_kv)

    # Launch BMM for dV on stream 2
    with torch.cuda.stream(stream2):
        dV = torch.matmul(P_drop_grouped.transpose(-2, -1), dO_grouped).to(torch.bfloat16)

    # Wait for stream 1 (dP_raw needed for softmax kernel on current stream)
    current_stream.wait_stream(stream1)

    # ----------------------------------------------------------------
    # Step 3: Triton kernel for softmax backward (on current stream)
    # dP_raw: [bs, 80, sq, skv]  bfloat16
    # ----------------------------------------------------------------
    P = attn_weights          # [bs, 80, sq, skv] bfloat16
    mask_c = dropout_mask     # [bs, 80, sq, skv] bool
    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=dO_grouped.device)

    inv_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    stride_bs    = dP_raw.stride(0)
    stride_h     = dP_raw.stride(1)
    stride_sq_s  = dP_raw.stride(2)
    stride_skv_s = dP_raw.stride(3)

    # Use single-pass kernel when skv fits in a power-of-2 block (most common cases)
    # Otherwise fall back to multi-row two-pass kernel
    if seq_kv <= 512:
        BLOCK_SQ_K  = 16
        BLOCK_SKV_K = 512
        NW = 4
        use_single = True
    elif seq_kv <= 1024:
        BLOCK_SQ_K  = 16
        BLOCK_SKV_K = 1024
        NW = 8
        use_single = True
    elif seq_kv <= 2048:
        BLOCK_SQ_K  = 8
        BLOCK_SKV_K = 2048
        NW = 16
        use_single = True
    elif seq_kv <= 4096:
        BLOCK_SQ_K  = 4
        BLOCK_SKV_K = 4096
        NW = 16
        use_single = True
    else:
        BLOCK_SQ_K  = 4
        BLOCK_SKV_K = 2048
        NW = 16
        use_single = False

    grid_softmax = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ_K))

    if use_single:
        softmax_bwd_singlepass_kernel[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv_s,
            BLOCK_SQ=BLOCK_SQ_K,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=NW,
            num_stages=1,
        )
    else:
        softmax_bwd_multirow_kernel[grid_softmax](
            dP_raw, P, mask_c, dS,
            inv_scale,
            seq_q, seq_kv,
            stride_bs, stride_h, stride_sq_s, stride_skv_s,
            BLOCK_SQ=BLOCK_SQ_K,
            BLOCK_SKV=BLOCK_SKV_K,
            num_warps=NW,
            num_stages=1,
        )

    # Wait for stream 2 (dV) to finish before returning
    current_stream.wait_stream(stream2)

    return dS, dV

```

---

## Experiment #25 — 2026-06-29 23:28:16 UTC ❌ DISCARD

**Hypothesis:** Two targeted changes from the previous iteration's version:

**Result:** 433.42 μs

---

## Experiment #26 — 2026-06-29 23:30:06 UTC 💥 CRASH

**Hypothesis:** ** Added `fused_ds_kernel` — a new Triton kernel that operates on GQA-native grouped layout `[bs, 8, 10*sq, skv]`. The grid is `(bs * n_kv_heads, cdiv(n_groups * seq_q, BLOCK_SQ))`. Each program loads

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

