# Experiment History

Tracks every kernel attempt, its code, hypothesis, and result.

---

## Experiment #1 — 2026-06-29 18:12:41 UTC ✅ KEEP

**Hypothesis:** Baseline 'starting_point' — initial benchmark

**Result:** 3483.45 μs

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

## Experiment #2 — 2026-06-29 18:14:35 UTC ✅ KEEP

**Hypothesis:** ** Two Triton kernels:

**Result:** 1274.79 μs

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

```

---

## Experiment #3 — 2026-06-29 18:16:21 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 1570.66 μs

---

## Experiment #4 — 2026-06-29 18:17:52 UTC ✅ KEEP

**Hypothesis:** **

**Result:** 1079.61 μs

**Kernel code:**
```python
"""
Hybrid attention-backward kernel: cuBLAS BMMs + fused Triton elementwise softmax-bwd.

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
# Fused Triton elementwise kernel:
#   Input:  dP [bs, 80, sq, skv] bfloat16 (result of dO @ V^T with dropout applied)
#           P  [bs, 80, sq, skv] bfloat16
#   Output: dS [bs, 80, sq, skv] bfloat16
#   
#   For each row (bs, h, sq): dS = P * (dP - sum(dP * P))
#   Single pass: read P and dP once (via row tiles), compute row sum, write dS.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,   # [bs, 80, sq, skv]  bfloat16
    P_ptr,    # [bs, 80, sq, skv]  bfloat16
    dS_ptr,   # [bs, 80, sq, skv]  bfloat16  output
    # strides [bs, h, sq, skv]
    stride_b, stride_h, stride_sq, stride_skv,
    # dims
    n_heads: tl.constexpr,
    skv,
    BLOCK_SKV: tl.constexpr,
):
    # Grid: (bs * n_heads * sq,)  — one program per row
    pid = tl.program_id(0)

    # Decode (b, h, sq_idx) from flat pid
    # total rows = bs * n_heads * sq
    row_base = pid * stride_sq  # offset to the start of this row
    # (we'll compute base as pid * stride_sq assuming stride_sq = skv for contiguous)

    skv_tiles = tl.cdiv(skv, BLOCK_SKV)

    # Pass 1: compute row_sum = sum_k(dP[row, k] * P[row, k])
    row_sum = tl.zeros([1], dtype=tl.float32)

    for tile_idx in range(skv_tiles):
        k_offs = tile_idx * BLOCK_SKV + tl.arange(0, BLOCK_SKV)
        k_mask = k_offs < skv

        dP_tile = tl.load(dP_ptr + row_base + k_offs * stride_skv, mask=k_mask, other=0.0).to(tl.float32)
        P_tile  = tl.load(P_ptr  + row_base + k_offs * stride_skv, mask=k_mask, other=0.0).to(tl.float32)
        row_sum += tl.sum(dP_tile * P_tile, axis=0)

    # Pass 2: write dS = P * (dP - row_sum)
    rs = row_sum  # scalar

    for tile_idx in range(skv_tiles):
        k_offs = tile_idx * BLOCK_SKV + tl.arange(0, BLOCK_SKV)
        k_mask = k_offs < skv

        dP_tile = tl.load(dP_ptr + row_base + k_offs * stride_skv, mask=k_mask, other=0.0).to(tl.float32)
        P_tile  = tl.load(P_ptr  + row_base + k_offs * stride_skv, mask=k_mask, other=0.0).to(tl.float32)

        dS_tile = P_tile * (dP_tile - rs)
        tl.store(dS_ptr + row_base + k_offs * stride_skv, dS_tile.to(tl.bfloat16), mask=k_mask)


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

    # --- Step 1: PyTorch transpose (view, no copy) ---
    # grad_attn_output: [bs, sq, 80, 128] -> [bs, 80, sq, 128]
    dO = grad_attn_output.transpose(1, 2)  # non-contiguous view, no copy

    # --- Step 2: GQA expand value_states (view only, no copy) ---
    # [bs, 8, skv, 128] -> [bs, 80, skv, 128]
    vs_exp = value_states[:, :, None, :, :].expand(
        bs, n_kv_heads, n_groups, seq_kv, head_dim
    ).reshape(bs, n_heads, seq_kv, head_dim)
    # reshape of non-contiguous may need contiguous — use as_strided instead
    # Actually expand+reshape on non-contiguous can fail; let's make contiguous for BMM
    # but do it in bf16 to avoid the fp32 upcast cost
    vs_exp_c = vs_exp.contiguous()  # [bs, 80, skv, 128] bf16

    # Make dO contiguous for BMM
    dO_c = dO.contiguous()  # [bs, 80, sq, 128] bf16

    # --- Step 3: cuBLAS BMM: dP = dO @ V^T -> [bs, 80, sq, skv] ---
    # Using bf16 throughout — B200 tensor cores handle bf16 natively
    dP = torch.matmul(dO_c, vs_exp_c.transpose(-2, -1))  # [bs, 80, sq, skv] bf16

    # --- Step 4: Dropout backward ---
    dropout_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0
    # Apply dropout mask: dP = dP * mask * scale (where mask=True means kept)
    dP = dP * dropout_mask * dropout_scale  # bool * bf16 auto-promotes

    # --- Step 5: Fused Triton softmax backward ---
    # dS = P * (dP - sum(dP * P, dim=-1, keepdim=True))
    # dP and attn_weights must be contiguous for the Triton kernel
    dP_c = dP.contiguous()      # should already be contiguous
    P_c  = attn_weights.contiguous()
    grad_attn_scores = torch.empty_like(dP_c)

    # Each program handles one row (bs, h, sq)
    n_rows = bs * n_heads * seq_q
    BLOCK_SKV = 512  # process row in chunks; tune as needed

    softmax_bwd_kernel[(n_rows,)](
        dP_c, P_c, grad_attn_scores,
        # strides
        dP_c.stride(0), dP_c.stride(1), dP_c.stride(2), dP_c.stride(3),
        # dims
        n_heads, seq_kv,
        BLOCK_SKV,
    )

    # --- Step 6: cuBLAS BMM for dV: Pd^T @ dO -> [bs, 80, skv, 128] ---
    attn_weights_dropped_c = attn_weights_dropped.contiguous()
    dV_exp = torch.matmul(
        attn_weights_dropped_c.transpose(-2, -1),  # [bs, 80, skv, sq]
        dO_c                                        # [bs, 80, sq, 128]
    )  # -> [bs, 80, skv, 128]

    # --- Step 7: GQA reduction: sum over groups -> [bs, 8, skv, 128] ---
    grad_value_states = dV_exp.reshape(bs, n_kv_heads, n_groups, seq_kv, head_dim).sum(dim=2).to(torch.bfloat16)

    return grad_attn_scores, grad_value_states

```

---

## Experiment #5 — 2026-06-29 18:19:08 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 1147.01 μs

---

## Experiment #6 — 2026-06-29 18:20:59 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 1570.06 μs

