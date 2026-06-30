# Experiment History

Tracks every kernel attempt, its code, hypothesis, and result.

---

## Experiment #1 — 2026-06-29 21:44:00 UTC ✅ KEEP

**Hypothesis:** Baseline 'starting_point' — initial benchmark

**Result:** 3430.44 μs

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

## Experiment #2 — 2026-06-29 21:45:57 UTC 💥 CRASH

**Hypothesis:** ** A Triton-based fused implementation with two kernels:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #3 — 2026-06-29 21:47:33 UTC ✅ KEEP

**Hypothesis:** ** Replaced `custom_kernel` with a `@torch.compile(mode="max-autotune", fullgraph=True)` decorated inner function `_attn_bwd_compiled` that:

**Result:** 1926.35 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel — torch.compile + GQA-avoiding dV path.

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
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10


@torch.compile(mode="max-autotune", fullgraph=True)
def _attn_bwd_compiled(dO_in, attn_weights, attn_weights_dropped,
                        value_states, dropout_mask, attention_dropout):
    """
    Optimized attention backward:
    - GQA dV: reshape + einsum avoids full [bs,80,skv,d] expansion
    - Contiguous tensors for BLAS efficiency
    - torch.compile fuses elementwise ops
    """
    bs     = dO_in.shape[0]
    seq_q  = dO_in.shape[1]
    seq_kv = value_states.shape[2]
    n_kv   = NUM_KEY_VALUE_HEADS
    n_g    = N_GROUPS
    d      = HEAD_DIM

    # [bs, sq, 80, d] -> [bs, 80, sq, d], float32 for numerical stability
    dO = dO_in.transpose(1, 2).contiguous().to(torch.float32)  # [bs, 80, sq, d]

    # ── dP computation: dO @ V^T ──────────────────────────────────────────────
    # Expand value_states: [bs, 8, skv, d] -> [bs, 80, skv, d]
    vs_exp = value_states[:, :, None, :, :].expand(
        bs, n_kv, n_g, seq_kv, d
    ).reshape(bs, n_kv * n_g, seq_kv, d).contiguous().to(torch.float32)

    # dP_dropped: [bs, 80, sq, skv]
    dP_dropped = torch.matmul(dO, vs_exp.transpose(-2, -1))

    # Dropout backward
    scale = 1.0 / (1.0 - attention_dropout)
    dP = dP_dropped * dropout_mask.to(torch.float32) * scale

    # ── Softmax backward ──────────────────────────────────────────────────────
    P = attn_weights.to(torch.float32)
    dPP = dP * P
    dS = P * (dP - dPP.sum(dim=-1, keepdim=True))
    dS = dS.to(torch.bfloat16)

    # ── dV computation via GQA-aware grouped einsum ───────────────────────────
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    Pd = attn_weights_dropped.to(torch.float32).reshape(bs, n_kv, n_g, seq_q, seq_kv)
    # dO: [bs, 80, sq, d] -> [bs, 8, 10, sq, d]
    dO_grouped = dO.reshape(bs, n_kv, n_g, seq_q, d)

    # dV[b,g,s,d] = sum_h sum_q Pd[b,g,h,q,s] * dO_grouped[b,g,h,q,d]
    # einsum: 'bghqs,bghqd->bgsd'
    # Use bmm for efficiency: reshape to [bs*8, 10*skv, sq] @ [bs*8, 10*sq, d]?
    # Actually use einsum which torch.compile will optimize:
    dV = torch.einsum('bghqs,bghqd->bgsd', Pd, dO_grouped)
    dV = dV.to(torch.bfloat16)

    return dS, dV


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    return _attn_bwd_compiled(
        grad_attn_output, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, attention_dropout
    )

```

---

## Experiment #4 — 2026-06-29 21:49:04 UTC ✅ KEEP

**Hypothesis:** ** Replaced the `_attn_bwd_compiled` function entirely. The old code expanded value_states to [bs,80,skv,d] float32 (10× memory bandwidth waste). The new code:

**Result:** 871.55 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel — torch.compile + GQA-aware BMMs.

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
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10


@torch.compile(mode="max-autotune", fullgraph=True)
def _attn_bwd_compiled(dO_in, attn_weights, attn_weights_dropped,
                        value_states, dropout_mask, attention_dropout):
    """
    Optimized attention backward using GQA structure in both BMMs:

    - dP BMM: reshape dO to [bs*8, 10*sq, d] and use value_states as [bs*8, skv, d]
              → single batched GEMM [bs*8, 10*sq, d] @ [bs*8, d, skv] = [bs*8, 10*sq, skv]
              → avoids materializing [bs,80,skv,d] expanded tensor

    - dV BMM: reshape attn_weights_dropped to [bs*8, 10*sq, skv] and dO to [bs*8, 10*sq, d]
              → single batched GEMM [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] = [bs*8, skv, d]
              → directly gives dV without group-sum (mathematically equivalent)

    - BMMs in bfloat16, only softmax-backward in float32
    - torch.compile fuses elementwise ops
    """
    bs     = dO_in.shape[0]
    seq_q  = dO_in.shape[1]
    seq_kv = value_states.shape[2]
    n_kv   = NUM_KEY_VALUE_HEADS   # 8
    n_g    = N_GROUPS              # 10
    d      = HEAD_DIM              # 128

    # [bs, sq, 80, d] -> [bs, 80, sq, d] -> contiguous bfloat16
    dO = dO_in.transpose(1, 2).contiguous()  # [bs, 80, sq, d] bfloat16

    # ── dP computation: dO @ V^T  (GQA-aware, no expansion) ──────────────────
    # Reshape dO: [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    dO_reshaped = dO.reshape(bs, n_kv, n_g, seq_q, d).reshape(bs * n_kv, n_g * seq_q, d)

    # value_states: [bs, 8, skv, d] -> [bs*8, skv, d]
    vs_flat = value_states.reshape(bs * n_kv, seq_kv, d)

    # BMM: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]  (bfloat16)
    dP_dropped_flat = torch.bmm(dO_reshaped, vs_flat.transpose(-2, -1))  # [bs*8, 10*sq, skv]

    # Reshape back: [bs*8, 10*sq, skv] -> [bs, 80, sq, skv]
    dP_dropped = dP_dropped_flat.reshape(bs, n_kv * n_g, seq_q, seq_kv)

    # Dropout backward (scale, in float32 for softmax bwd)
    scale = 1.0 / (1.0 - attention_dropout)
    dP = dP_dropped.to(torch.float32) * dropout_mask.to(torch.float32) * scale

    # ── Softmax backward (float32) ────────────────────────────────────────────
    P = attn_weights.to(torch.float32)
    dPP = dP * P
    dS = P * (dP - dPP.sum(dim=-1, keepdim=True))
    dS = dS.to(torch.bfloat16)

    # ── dV computation: GQA-aware, no expansion, no group-sum ─────────────────
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs*8, 10*sq, skv]
    Pd_flat = attn_weights_dropped.reshape(bs * n_kv, n_g * seq_q, seq_kv)  # [bs*8, 10*sq, skv]

    # dO: [bs*8, 10*sq, d] already computed as dO_reshaped
    # BMM: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]  (bfloat16)
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_reshaped)  # [bs*8, skv, d]

    # Reshape to [bs, 8, skv, d]
    dV = dV_flat.reshape(bs, n_kv, seq_kv, d).to(torch.bfloat16)

    return dS, dV


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    return _attn_bwd_compiled(
        grad_attn_output, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, attention_dropout
    )

```

---

## Experiment #5 — 2026-06-29 21:50:38 UTC ✅ KEEP

**Hypothesis:** ** Added a Triton kernel `_softmax_bwd_dropout_kernel` that processes one `(bs*head*sq)` row per program instance. In a two-pass approach (first pass accumulates `sum(dP*P)` over the seq_kv dimension,

**Result:** 414.85 μs

**Kernel code:**
```python
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

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10


@triton.jit
def _softmax_bwd_dropout_kernel(
    dP_dropped_ptr,   # [N_rows, seq_kv]  bfloat16
    P_ptr,            # [N_rows, seq_kv]  bfloat16
    mask_ptr,         # [N_rows, seq_kv]  bool
    dS_ptr,           # [N_rows, seq_kv]  bfloat16  output
    seq_kv: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: for each row r,
        dP[r,:] = dP_dropped[r,:] * mask[r,:] * scale
        dS[r,:] = P[r,:] * (dP[r,:] - sum(dP[r,:] * P[r,:]))
    All in float32, output in bfloat16.
    Processes one row per program instance.
    """
    row_idx = tl.program_id(0)

    row_start = row_idx * seq_kv

    # Accumulate sum(dP * P) over blocks
    acc = tl.zeros([1], dtype=tl.float32)

    # First pass: compute sum(dP * P)
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

    row_sum = tl.sum(acc, axis=0)  # scalar

    # Second pass: compute dS = P * (dP - row_sum) and store
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

    # Choose block size based on seq_kv
    if seq_kv <= 256:
        BLOCK_SIZE = 256
    elif seq_kv <= 512:
        BLOCK_SIZE = 512
    elif seq_kv <= 1024:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048

    # Clamp to seq_kv (must be power of 2 for Triton constexpr)
    import math
    BLOCK_SIZE = min(BLOCK_SIZE, 2 ** math.ceil(math.log2(seq_kv)))

    grid = (N_rows,)
    _softmax_bwd_dropout_kernel[grid](
        dP_dropped, P, mask, dS,
        seq_kv=seq_kv,
        scale=scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return dS


@torch.compile(mode="max-autotune", fullgraph=True)
def _bmm_dV(attn_weights_dropped, dO_reshaped, bs, n_kv, seq_kv, d):
    """Fused BMM for dV computation."""
    Pd_flat = attn_weights_dropped.reshape(bs * n_kv, -1, seq_kv)
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_reshaped)
    return dV_flat.reshape(bs, n_kv, seq_kv, d).to(torch.bfloat16)


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

    # ── dP computation: dO @ V^T  (GQA-aware, no expansion) ──────────────────
    # Reshape dO: [bs, 80, sq, d] -> [bs*8, 10*sq, d]
    dO_reshaped = dO.reshape(bs, n_kv, n_g, seq_q, d).reshape(bs * n_kv, n_g * seq_q, d)

    # value_states: [bs, 8, skv, d] -> [bs*8, skv, d]
    vs_flat = value_states.reshape(bs * n_kv, seq_kv, d)

    # BMM: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]  (bfloat16)
    dP_dropped_flat = torch.bmm(dO_reshaped, vs_flat.transpose(-2, -1))  # [bs*8, 10*sq, skv]

    # Reshape back: [bs*8, 10*sq, skv] -> [bs, 80, sq, skv]
    dP_dropped = dP_dropped_flat.reshape(bs, n_kv * n_g, seq_q, seq_kv)

    # ── Fused softmax backward + dropout (Triton kernel) ─────────────────────
    scale = 1.0 / (1.0 - attention_dropout)

    # Flatten row dimensions: [bs, 80, sq, skv] -> [bs*80*sq, skv]
    N_rows = bs * NUM_ATTENTION_HEADS * seq_q
    dP_dropped_2d = dP_dropped.contiguous().reshape(N_rows, seq_kv)
    P_2d = attn_weights.contiguous().reshape(N_rows, seq_kv)
    mask_2d = dropout_mask.contiguous().reshape(N_rows, seq_kv)

    dS_2d = fused_softmax_bwd_dropout(dP_dropped_2d, P_2d, mask_2d, scale, seq_kv)
    dS = dS_2d.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ── dV computation: GQA-aware, no expansion, no group-sum ─────────────────
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs*8, 10*sq, skv]
    Pd_flat = attn_weights_dropped.reshape(bs * n_kv, n_g * seq_q, seq_kv)

    # BMM: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]  (bfloat16)
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_reshaped)  # [bs*8, skv, d]

    # Reshape to [bs, 8, skv, d]
    dV = dV_flat.reshape(bs, n_kv, seq_kv, d).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #6 — 2026-06-29 21:52:09 UTC ✅ KEEP

**Hypothesis:** **

**Result:** 411.65 μs

**Kernel code:**
```python
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
    """
    Single-pass fused kernel: for each row r,
        dP[r,:] = dP_dropped[r,:] * mask[r,:] * scale
        dS[r,:] = P[r,:] * (dP[r,:] - sum(dP[r,:] * P[r,:]))

    When BLOCK_SIZE >= seq_kv, the entire row fits in registers so we do:
      1) Load everything into registers
      2) Compute row_sum in-register
      3) Compute dS and store
    This avoids a second pass over global memory entirely.

    For larger seq_kv (multiple blocks), we still need two passes over the
    blocks but keep data in registers within each block to avoid re-loading.
    We use the standard two-loop approach but with the key optimization that
    when seq_kv <= BLOCK_SIZE we only touch memory once per element.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * seq_kv

    # ── Single-pass: load all data into registers, accumulate, then store ────
    # This works when seq_kv fits in one block (BLOCK_SIZE >= seq_kv).
    # For the multi-block case we fall back to two passes, but the inner loop
    # benefit is still that we don't re-issue global loads for the second pass
    # within each block — instead we cache in registers.

    # We unroll by processing the whole row in one shot if it fits.
    # Triton will keep vectors in registers across the two loops.
    # The trick: use a static number of blocks and keep arrays alive.

    # Use tl.static_range for the case where seq_kv == BLOCK_SIZE (one block)
    # For the general case we compute row_sum first then write.

    # Since BLOCK_SIZE is constexpr and we pick BLOCK_SIZE >= seq_kv when
    # seq_kv is small (<=2048), we handle the single-block case specially.
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
    Pass 1: accumulate row_sum = sum(dP * P)
    Pass 2: compute and store dS = P * (dP - row_sum)
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

    # Determine block size
    # Single-pass kernel: BLOCK_SIZE must be >= seq_kv (power of 2)
    SINGLE_PASS_MAX = 4096  # register pressure limit

    pow2 = 2 ** math.ceil(math.log2(seq_kv)) if seq_kv > 1 else 1

    grid = (N_rows,)

    if pow2 <= SINGLE_PASS_MAX:
        # Single-pass: entire row fits in registers
        BLOCK_SIZE = pow2
        _softmax_bwd_dropout_kernel_singlepass[grid](
            dP_dropped, P, mask, dS,
            seq_kv=seq_kv,
            scale=scale,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Multi-block two-pass for very large seq_kv
        BLOCK_SIZE = 2048
        _softmax_bwd_dropout_kernel_multiblock[grid](
            dP_dropped, P, mask, dS,
            seq_kv=seq_kv,
            scale=scale,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return dS


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

    # ── dP computation: dO @ V^T  (GQA-aware, no expansion) ──────────────────
    # Reshape dO: [bs, 80, sq, d] -> [bs*8, 10*sq, d]
    dO_reshaped = dO.reshape(bs, n_kv, n_g, seq_q, d).reshape(bs * n_kv, n_g * seq_q, d)

    # value_states: [bs, 8, skv, d] -> [bs*8, skv, d]
    vs_flat = value_states.reshape(bs * n_kv, seq_kv, d)

    # Launch both BMMs before softmax kernel so L2 can be warm
    # BMM: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]  (bfloat16)
    dP_dropped_flat = torch.bmm(dO_reshaped, vs_flat.transpose(-2, -1))  # [bs*8, 10*sq, skv]

    # dV BMM: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]  (bfloat16)
    # Use attn_weights_dropped reshaped: [bs, 80, sq, skv] -> [bs*8, 10*sq, skv]
    Pd_flat = attn_weights_dropped.reshape(bs * n_kv, n_g * seq_q, seq_kv)
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_reshaped)  # [bs*8, skv, d]

    # Reshape back: [bs*8, 10*sq, skv] -> [bs, 80, sq, skv]
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

```

---

## Experiment #7 — 2026-06-29 21:54:25 UTC ❌ DISCARD

**Hypothesis:** ** Extracted the two BMM operations into a separate function `_compute_bmms` decorated with `@torch.compile(mode="max-autotune", fullgraph=True)`. Inside this function, instead of `transpose(1,2).cont

**Result:** 742.71 μs

---

## Experiment #8 — 2026-06-29 21:55:50 UTC 💥 CRASH

**Hypothesis:** Complete rewrite of submission.py replacing the 3-stage pipeline (BMM → softmax-bwd Triton → BMM) with a single fused Triton kernel `_flash_attn_bwd_kernel`. The kernel grid is (bs*80, ceil(seq_q/BLOC

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #9 — 2026-06-29 21:57:36 UTC ❌ DISCARD

**Hypothesis:** ** Replaced the entire `custom_kernel` with a fully fused `_flash_attn_bwd_kernel` Triton kernel. The kernel:

**Result:** 5353.55 μs

---

## Experiment #10 — 2026-06-29 21:58:58 UTC ❌ DISCARD

**Hypothesis:** Replaced the fused flash-attention-style Triton kernel with the three-stage pipeline from experiments #5/#6, augmented with CUDA stream parallelism. Two persistent streams (_stream1, _stream2) are cre

**Result:** 1990.14 μs

---

## Experiment #11 — 2026-06-29 22:00:54 UTC 💥 CRASH

**Hypothesis:** ** Replaced the two separate `torch.bmm` calls (and the CUDA stream overlap logic) with a single `_fused_bmm_kernel` Triton kernel. The grid is `(bs*n_kv, n_skv_tiles, n_groups*n_sq_tiles)`. Each prog

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #12 — 2026-06-29 22:02:48 UTC ❌ DISCARD

**Hypothesis:** ** Extracted `_bmm_pipeline` function containing: (a) dO transpose+contiguous, (b) GQA reshape for dO and value_states, (c) BMM1 (dP_dropped = dO @ V^T), (d) BMM2 (dV = Pd^T @ dO), (e) reshape outputs

**Result:** 786.45 μs

---

## Experiment #13 — 2026-06-29 22:04:57 UTC ❌ DISCARD

**Hypothesis:** ** Wrote a new Triton kernel `_fused_dS_kernel` where each program handles one `(bs, head, sq_row)` triple. It:

**Result:** 3996.62 μs

---

## Experiment #14 — 2026-06-29 22:06:54 UTC ❌ DISCARD

**Hypothesis:** ** Rewrote the entire pipeline to:

**Result:** 1345.12 μs

---

## Experiment #15 — 2026-06-29 22:08:48 UTC ❌ DISCARD

**Hypothesis:** ** Two targeted changes to the BMM operations:

**Result:** 1703.57 μs

---

## Experiment #16 — 2026-06-29 22:10:35 UTC 💥 CRASH

**Hypothesis:** ** Replaced the separate `torch.bmm` + dropout mask + Triton softmax-bwd pipeline with a single `_fused_bmm_softmax_bwd_kernel` that:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #17 — 2026-06-29 22:12:05 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 429.28 μs

---

## Experiment #18 — 2026-06-29 22:13:49 UTC 💥 CRASH

**Hypothesis:** ** Added `_graph_cache` dict, `_build_graph()` function that does warmup + `torch.cuda.CUDAGraph` capture, and rewrote `custom_kernel` to use shape-keyed graph caching with copy-in/replay/copy-out pat

**Result:** CRASH

**Error:**
```
run_eval exited 1
```

---

## Experiment #19 — 2026-06-29 22:15:49 UTC ❌ DISCARD

**Hypothesis:** ** Added `_fused_bmm_kernel` — a Triton kernel with grid `(B, cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))` where each program handles one tile `(batch*kv_head, m_tile, n_tile)`. For each tile: (1) load dO til

**Result:** 5763.83 μs

---

## Experiment #20 — 2026-06-29 22:17:11 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 1233.56 μs

---

## Experiment #21 — 2026-06-29 22:18:56 UTC ❌ DISCARD

**Hypothesis:** ** Restructured `custom_kernel` to:

**Result:** 1302.20 μs

---

## Experiment #22 — 2026-06-29 22:20:25 UTC ❌ DISCARD

**Hypothesis:** ** Changed BMM shapes from `[bs*8, 10*sq, d]` to `[bs*80, sq, d]` for both BMMs. Value states are expanded via `unsqueeze(2).expand(...)` before reshaping to `[bs*80, skv, d]`. BMM2 result is reshaped

**Result:** 1357.18 μs

---

## Experiment #23 — 2026-06-29 22:22:06 UTC ❌ DISCARD

**Hypothesis:** ** Replaced the `[bs*80, sq, d]` BMM layout with `[bs*8, 10*sq, d]` layout:

**Result:** 1225.33 μs

---

## Experiment #24 — 2026-06-29 22:23:51 UTC ❌ DISCARD

**Hypothesis:** ** Added two new Triton kernels (`_softmax_bwd_partial_sum_kernel` and `_softmax_bwd_apply_kernel`) for the two-pass parallel approach. Modified `fused_softmax_bwd_dropout` to use these when `seq_kv >

**Result:** 1222.15 μs

---

## Experiment #25 — 2026-06-29 22:25:34 UTC ❌ DISCARD

**Hypothesis:** ** Replaced the separate `torch.bmm` for BMM1 + Triton softmax-bwd kernel with a single `_fused_bmm1_softmax_bwd_kernel` Triton kernel. One program instance per `(batch, head, query_row)`. The kernel:

**Result:** 6030.24 μs

---

## Experiment #26 — 2026-06-29 22:27:48 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 6140.55 μs

