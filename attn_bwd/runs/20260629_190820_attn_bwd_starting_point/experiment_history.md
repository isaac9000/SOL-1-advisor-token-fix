# Experiment History

Tracks every kernel attempt, its code, hypothesis, and result.

---

## Experiment #1 — 2026-06-29 19:08:36 UTC ✅ KEEP

**Hypothesis:** Baseline 'starting_point' — initial benchmark

**Result:** 3429.12 μs

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

## Experiment #2 — 2026-06-29 19:10:12 UTC ✅ KEEP

**Hypothesis:** Restructured the attention backward computation to:

**Result:** 2091.35 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using torch.compile with restructured
GQA computation that avoids materializing expanded tensors.

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
DROPOUT_SCALE = 1.0 / (1.0 - 0.1)  # precomputed for attention_dropout=0.1


def _attn_backward_compiled(
    grad_attn_output,   # [bs, sq, 80, 128]  bf16
    attn_weights,       # [bs, 80, sq, skv]  bf16
    attn_weights_dropped,  # [bs, 80, sq, skv]  bf16
    value_states,       # [bs, 8, skv, 128]  bf16
    dropout_mask,       # [bs, 80, sq, skv]  bool
    attention_dropout,  # scalar float
):
    bs = grad_attn_output.shape[0]
    seq_q = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # Transpose grad: [bs, sq, 80, d] -> [bs, 80, sq, d], then reshape to groups
    # [bs, 8, 10, sq, d]
    dO = grad_attn_output.transpose(1, 2).to(torch.float32)
    dO_grouped = dO.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM)

    # Reshape attn weights to grouped layout: [bs, 8, 10, sq, skv]
    attn_weights_grouped = attn_weights.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv)
    attn_weights_dropped_grouped = attn_weights_dropped.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv)
    dropout_mask_grouped = dropout_mask.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv)

    # value_states: [bs, 8, skv, d] -> for matmul we need [bs, 8, 1, d, skv]
    # dO_grouped: [bs, 8, 10, sq, d] @ V^T [bs, 8, 1, d, skv] -> [bs, 8, 10, sq, skv]
    V_t = value_states.to(torch.float32).transpose(-2, -1).unsqueeze(2)  # [bs, 8, 1, d, skv]
    dP_dropped = torch.matmul(dO_grouped, V_t)  # [bs, 8, 10, sq, skv]

    # Dropout backward: scale by mask / (1 - p)
    if attention_dropout > 0.0:
        scale = 1.0 / (1.0 - attention_dropout)
        dP = dP_dropped * dropout_mask_grouped * scale
    else:
        dP = dP_dropped

    # Softmax backward: dS = P * (dP - sum(dP * P, dim=-1, keepdim=True))
    P = attn_weights_grouped.to(torch.float32)
    dPP = dP * P
    dS = P * (dP - dPP.sum(dim=-1, keepdim=True))
    dS = dS.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv).to(torch.bfloat16)

    # dV: attn_weights_dropped_grouped^T @ dO_grouped -> [bs, 8, 10, skv, d]
    # sum over groups (dim=2) -> [bs, 8, skv, d]
    # attn_weights_dropped_grouped: [bs, 8, 10, sq, skv]
    # dO_grouped: [bs, 8, 10, sq, d]
    dV = torch.matmul(
        attn_weights_dropped_grouped.to(torch.float32).transpose(-2, -1),  # [bs, 8, 10, skv, sq]
        dO_grouped  # [bs, 8, 10, sq, d]
    ).sum(dim=2).to(torch.bfloat16)  # [bs, 8, skv, d]

    return dS, dV


# Cache the compiled function to avoid recompilation overhead
_compiled_fn = torch.compile(
    _attn_backward_compiled,
    mode="max-autotune",
    fullgraph=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    return _compiled_fn(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        attention_dropout,
    )

```

---

## Experiment #3 — 2026-06-29 19:12:00 UTC ✅ KEEP

**Hypothesis:** Replaced the 5D grouped tensor approach with flat 3D BMMs:

**Result:** 974.64 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using torch.compile with restructured
GQA computation that uses bfloat16 BMMs (flat 3D batched) and float32
only for the elementwise softmax backward accumulation.

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


def _attn_backward_compiled(
    grad_attn_output,      # [bs, sq, 80, 128]     bf16
    attn_weights,          # [bs, 80, sq, skv]     bf16
    attn_weights_dropped,  # [bs, 80, sq, skv]     bf16
    value_states,          # [bs, 8, skv, 128]     bf16
    dropout_mask,          # [bs, 80, sq, skv]     bool
    attention_dropout,     # scalar float
):
    bs = grad_attn_output.shape[0]
    seq_q = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, sq, 80, d] -> [bs, 80, sq, d]
    dO = grad_attn_output.transpose(1, 2)  # bf16, [bs, 80, sq, d]

    # ------------------------------------------------------------------ #
    #  Compute dP_dropped = dO @ V^T  (both bfloat16, flat 3D BMM)
    #  Need V expanded from [bs,8,skv,d] to [bs,80,skv,d]
    #  We do this without materializing via repeat_interleave then flat BMM
    # ------------------------------------------------------------------ #

    # Expand V: [bs, 8, skv, d] -> [bs, 80, skv, d] using interleave-expand
    # Use reshape trick: [bs,8,1,skv,d] -> broadcast -> [bs,8,10,skv,d] -> [bs,80,skv,d]
    V_exp = value_states.unsqueeze(2).expand(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_kv, HEAD_DIM).reshape(bs, NUM_ATTENTION_HEADS, seq_kv, HEAD_DIM)
    # V_exp: [bs, 80, skv, d] bf16

    # Flatten to 3D: [bs*80, sq, d] and [bs*80, d, skv]
    dO_flat = dO.reshape(bs * NUM_ATTENTION_HEADS, seq_q, HEAD_DIM)           # [B80, sq, d]
    V_flat_t = V_exp.transpose(-2, -1).reshape(bs * NUM_ATTENTION_HEADS, HEAD_DIM, seq_kv)  # [B80, d, skv]

    # dP_dropped in bf16
    dP_dropped_flat = torch.bmm(dO_flat, V_flat_t)  # [B80, sq, skv] bf16

    # ------------------------------------------------------------------ #
    #  Dropout backward: mask and scale
    # ------------------------------------------------------------------ #
    if attention_dropout > 0.0:
        scale = 1.0 / (1.0 - attention_dropout)
        dropout_mask_flat = dropout_mask.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv)
        dP_flat = dP_dropped_flat * dropout_mask_flat * scale  # bf16
    else:
        dP_flat = dP_dropped_flat  # bf16

    # ------------------------------------------------------------------ #
    #  Softmax backward: dS = P * (dP - sum(dP * P, dim=-1, keepdim=True))
    #  Do in float32 for numerical stability
    # ------------------------------------------------------------------ #
    P_flat = attn_weights.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv).float()
    dP_flat_f32 = dP_flat.float()
    dPP = dP_flat_f32 * P_flat                                      # [B80, sq, skv]
    dS_flat = P_flat * (dP_flat_f32 - dPP.sum(dim=-1, keepdim=True))  # [B80, sq, skv]
    dS = dS_flat.to(torch.bfloat16).reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ------------------------------------------------------------------ #
    #  Compute dV = attn_weights_dropped^T @ dO  (grouped, sum over groups)
    #  attn_weights_dropped: [bs, 80, sq, skv] -> grouped [bs, 8, 10, sq, skv]
    #  dO: [bs, 80, sq, d] -> grouped [bs, 8, 10, sq, d]
    #  dV_group[i] = sum_g(aw_dropped[:,i,g,:,:]^T @ dO[:,i,g,:,:])
    #
    #  Flatten groups into batch: merge (bs*8*10) -> flat 3D BMM, then sum groups
    # ------------------------------------------------------------------ #
    aw_dropped_grouped = attn_weights_dropped.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv)  # [B8, 10, sq, skv]
    dO_grouped = dO.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM)                          # [B8, 10, sq, d]

    # Flatten groups into batch dim: [B8*10, sq, skv] and [B8*10, sq, d]
    aw_dropped_flat = aw_dropped_grouped.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv)  # [B80, sq, skv] bf16
    dO_flat_kv = dO_grouped.reshape(bs * NUM_ATTENTION_HEADS, seq_q, HEAD_DIM)             # [B80, sq, d]   bf16

    # dV per-group in bf16: [B80, skv, d]
    dV_flat = torch.bmm(aw_dropped_flat.transpose(-2, -1), dO_flat_kv)  # [B80, skv, d] bf16

    # Sum over 10 groups: [B80, skv, d] -> [B8, 10, skv, d] -> sum -> [B8, skv, d]
    dV = dV_flat.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS, seq_kv, HEAD_DIM).sum(dim=1)  # [B8, skv, d] bf16
    dV = dV.reshape(bs, NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV


# Cache the compiled function to avoid recompilation overhead
_compiled_fn = torch.compile(
    _attn_backward_compiled,
    mode="max-autotune",
    fullgraph=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    return _compiled_fn(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        attention_dropout,
    )

```

---

## Experiment #4 — 2026-06-29 19:13:42 UTC 💥 CRASH

**Hypothesis:** ** Replaced the `torch.compile` + PyTorch ops approach with a single Triton kernel `_attn_bwd_kernel` that:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #5 — 2026-06-29 19:15:17 UTC ✅ KEEP

**Hypothesis:** ** Replaced the `V_exp = value_states.unsqueeze(2).expand(...).reshape(...)` pattern with a grouped reshape of dO. For dP: `dO [bs,80,sq,d]` → `[bs,8,10,sq,d]` → `[bs*8, 10*sq, d]`, then BMM against `

**Result:** 867.81 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using torch.compile with restructured
GQA computation that AVOIDS materializing the expanded V tensor.

For dP: reshape dO [bs, 80, sq, d] -> [bs*8, 10*sq, d], V [bs, 8, skv, d] -> [bs*8, d, skv],
        single BMM -> [bs*8, 10*sq, skv] -> reshape [bs, 80, sq, skv].
For dV: grouped BMM on [bs*8*10, sq, skv] x [bs*8*10, sq, d] -> sum over groups.

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


def _attn_backward_compiled(
    grad_attn_output,      # [bs, sq, 80, 128]     bf16
    attn_weights,          # [bs, 80, sq, skv]     bf16
    attn_weights_dropped,  # [bs, 80, sq, skv]     bf16
    value_states,          # [bs, 8, skv, 128]     bf16
    dropout_mask,          # [bs, 80, sq, skv]     bool
    attention_dropout,     # scalar float
):
    bs = grad_attn_output.shape[0]
    seq_q = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, sq, 80, d] -> [bs, 80, sq, d]
    dO = grad_attn_output.transpose(1, 2)  # bf16, [bs, 80, sq, d]

    # ------------------------------------------------------------------ #
    #  Compute dP_dropped = dO @ V^T  WITHOUT materializing expanded V
    #
    #  dO: [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    #  V:  [bs, 8, skv, d] -> [bs*8, d, skv]
    #  BMM: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
    #  reshape -> [bs, 8, 10, sq, skv] -> [bs, 80, sq, skv]
    # ------------------------------------------------------------------ #

    # Reshape dO for grouped BMM: [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    dO_grouped = dO.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM)
    dO_for_dP = dO_grouped.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS * seq_q, HEAD_DIM)  # [B8, 10*sq, d]

    # V: [bs, 8, skv, d] -> [bs*8, d, skv]
    V_flat_t = value_states.reshape(bs * NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM).transpose(-2, -1)  # [B8, d, skv]

    # dP_dropped grouped BMM: [B8, 10*sq, d] @ [B8, d, skv] -> [B8, 10*sq, skv]
    dP_dropped_grouped = torch.bmm(dO_for_dP, V_flat_t)  # [B8, 10*sq, skv] bf16

    # Reshape to [bs, 80, sq, skv]
    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv) \
                                   .reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ------------------------------------------------------------------ #
    #  Dropout backward: mask and scale
    # ------------------------------------------------------------------ #
    if attention_dropout > 0.0:
        scale = 1.0 / (1.0 - attention_dropout)
        dP = dP_dropped * dropout_mask * scale  # bf16
    else:
        dP = dP_dropped  # bf16

    # ------------------------------------------------------------------ #
    #  Softmax backward: dS = P * (dP - sum(dP * P, dim=-1, keepdim=True))
    #  Do in float32 for numerical stability
    # ------------------------------------------------------------------ #
    P_flat = attn_weights.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv).float()
    dP_flat_f32 = dP.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv).float()
    dPP = dP_flat_f32 * P_flat                                               # [B80, sq, skv]
    dS_flat = P_flat * (dP_flat_f32 - dPP.sum(dim=-1, keepdim=True))        # [B80, sq, skv]
    dS = dS_flat.to(torch.bfloat16).reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ------------------------------------------------------------------ #
    #  Compute dV = attn_weights_dropped^T @ dO  (grouped, no V expansion)
    #
    #  attn_weights_dropped: [bs, 80, sq, skv] -> [bs*8, 10, sq, skv]
    #                                           -> [bs*8*10, sq, skv]
    #  dO: [bs, 8, 10, sq, d] -> [bs*8*10, sq, d]
    #  BMM: [B80, skv, sq] @ [B80, sq, d] -> [B80, skv, d]
    #  Sum over 10 groups: [B8, 10, skv, d] -> sum -> [B8, skv, d] -> [bs, 8, skv, d]
    # ------------------------------------------------------------------ #
    aw_dropped_flat = attn_weights_dropped.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv)  # [B80, sq, skv]
    dO_flat_kv = dO_grouped.reshape(bs * NUM_ATTENTION_HEADS, seq_q, HEAD_DIM)              # [B80, sq, d]

    # dV per-group in bf16: [B80, skv, d]
    dV_flat = torch.bmm(aw_dropped_flat.transpose(-2, -1), dO_flat_kv)  # [B80, skv, d] bf16

    # Sum over 10 groups: [B8, 10, skv, d] -> sum -> [B8, skv, d]
    dV = dV_flat.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS, seq_kv, HEAD_DIM).sum(dim=1)   # [B8, skv, d]
    dV = dV.reshape(bs, NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV


# Cache the compiled function to avoid recompilation overhead
_compiled_fn = torch.compile(
    _attn_backward_compiled,
    mode="max-autotune",
    fullgraph=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    return _compiled_fn(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        attention_dropout,
    )

```

---

## Experiment #6 — 2026-06-29 19:16:44 UTC ✅ KEEP

**Hypothesis:** Added a `softmax_bwd_kernel` Triton JIT kernel and `fused_softmax_bwd` launcher. The kernel tiles over the `seq_kv` dimension using a `BLOCK_SKV` constexpr (chosen based on skv size). Each program han

**Result:** 523.01 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using torch.compile for BMMs +
a custom Triton kernel that fuses the softmax-backward elementwise chain:
  1. dropout mask + scale
  2. dP * P row-sum reduction  
  3. final P * (dP - row_sum)
All in one pass, reading dP_dropped, dropout_mask, attn_weights once
and writing grad_attn_scores once.

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
def softmax_bwd_kernel(
    dP_dropped_ptr,   # [B80, sq, skv]  bfloat16
    dropout_mask_ptr, # [B80, sq, skv]  bool (uint8)
    P_ptr,            # [B80, sq, skv]  bfloat16
    dS_ptr,           # [B80, sq, skv]  bfloat16  (output)
    sq: tl.constexpr,
    skv: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """
    Each program handles one row: (batch*head, row_idx).
    Streams over the seq_kv dimension in tiles to compute the row sum,
    then writes dS in a second pass.
    """
    # pid 0 -> row index in [B80 * sq]
    row_id = tl.program_id(0)
    b80_idx = row_id // sq
    q_idx = row_id % sq

    row_offset = b80_idx * sq * skv + q_idx * skv

    # ---- First pass: compute row_sum = sum(dP * P) over skv ----
    row_sum = tl.zeros([1], dtype=tl.float32)

    for start in tl.range(0, skv, BLOCK_SKV):
        kv_ids = start + tl.arange(0, BLOCK_SKV)
        mask = kv_ids < skv
        offsets = row_offset + kv_ids

        dp_dropped = tl.load(dP_dropped_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        dmask = tl.load(dropout_mask_ptr + offsets, mask=mask, other=0).to(tl.float32)
        p = tl.load(P_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        dp = dp_dropped * dmask * scale
        row_sum += tl.sum(dp * p, axis=0)

    # ---- Second pass: compute dS = P * (dP - row_sum) and write ----
    for start in tl.range(0, skv, BLOCK_SKV):
        kv_ids = start + tl.arange(0, BLOCK_SKV)
        mask = kv_ids < skv
        offsets = row_offset + kv_ids

        dp_dropped = tl.load(dP_dropped_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        dmask = tl.load(dropout_mask_ptr + offsets, mask=mask, other=0).to(tl.float32)
        p = tl.load(P_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        dp = dp_dropped * dmask * scale
        ds = p * (dp - row_sum)

        tl.store(dS_ptr + offsets, ds.to(tl.bfloat16), mask=mask)


def fused_softmax_bwd(dP_dropped, dropout_mask, attn_weights, attention_dropout):
    """
    Fused softmax backward + dropout scaling.
    All inputs/output shaped [B80, sq, skv].
    """
    B80, sq, skv = dP_dropped.shape
    dS = torch.empty_like(dP_dropped)

    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Choose block size for skv dimension
    # Power of 2, at least covers common skv values
    if skv <= 64:
        BLOCK_SKV = 64
    elif skv <= 128:
        BLOCK_SKV = 128
    elif skv <= 256:
        BLOCK_SKV = 256
    elif skv <= 512:
        BLOCK_SKV = 512
    elif skv <= 1024:
        BLOCK_SKV = 1024
    else:
        BLOCK_SKV = 2048

    total_rows = B80 * sq
    grid = (total_rows,)

    softmax_bwd_kernel[grid](
        dP_dropped,
        dropout_mask,
        attn_weights,
        dS,
        sq=sq,
        skv=skv,
        scale=scale,
        BLOCK_SKV=BLOCK_SKV,
    )
    return dS


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs = grad_attn_output.shape[0]
    seq_q = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, sq, 80, d] -> [bs, 80, sq, d]
    dO = grad_attn_output.transpose(1, 2)  # bf16, [bs, 80, sq, d]

    # ------------------------------------------------------------------ #
    #  Compute dP_dropped = dO @ V^T  WITHOUT materializing expanded V
    #
    #  dO: [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    #  V:  [bs, 8, skv, d] -> [bs*8, d, skv]
    #  BMM: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
    #  reshape -> [bs, 80, sq, skv]
    # ------------------------------------------------------------------ #
    dO_grouped = dO.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM)
    dO_for_dP = dO_grouped.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS * seq_q, HEAD_DIM)

    V_flat_t = value_states.reshape(bs * NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM).transpose(-2, -1)

    dP_dropped_grouped = torch.bmm(dO_for_dP, V_flat_t)  # [B8, 10*sq, skv] bf16

    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv) \
                                   .reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ------------------------------------------------------------------ #
    #  Fused softmax backward via Triton kernel
    #  Reads dP_dropped, dropout_mask, attn_weights once; writes dS once
    # ------------------------------------------------------------------ #
    # Flatten to [B80, sq, skv] for the kernel
    dP_dropped_flat = dP_dropped.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv).contiguous()
    dropout_mask_flat = dropout_mask.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv).contiguous()
    attn_weights_flat = attn_weights.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv).contiguous()

    dS_flat = fused_softmax_bwd(dP_dropped_flat, dropout_mask_flat, attn_weights_flat, attention_dropout)
    dS = dS_flat.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ------------------------------------------------------------------ #
    #  Compute dV = attn_weights_dropped^T @ dO  (grouped, no V expansion)
    #
    #  attn_weights_dropped: [bs, 80, sq, skv] -> [bs*80, sq, skv]
    #  dO: [bs, 80, sq, d] -> [bs*80, sq, d]
    #  BMM: [B80, skv, sq] @ [B80, sq, d] -> [B80, skv, d]
    #  Sum over 10 groups: [B8, 10, skv, d] -> [B8, skv, d]
    # ------------------------------------------------------------------ #
    aw_dropped_flat = attn_weights_dropped.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv)
    dO_flat_kv = dO_grouped.reshape(bs * NUM_ATTENTION_HEADS, seq_q, HEAD_DIM)

    dV_flat = torch.bmm(aw_dropped_flat.transpose(-2, -1), dO_flat_kv)  # [B80, skv, d] bf16

    dV = dV_flat.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS, seq_kv, HEAD_DIM).sum(dim=1)
    dV = dV.reshape(bs, NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #7 — 2026-06-29 19:18:45 UTC ✅ KEEP

**Hypothesis:** Worker implementation

**Result:** 520.60 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using torch.compile for BMMs +
a custom Triton kernel that fuses the softmax-backward elementwise chain:
  1. dropout mask + scale
  2. dP * P row-sum reduction  
  3. final P * (dP - row_sum)
All in one pass, reading dP_dropped, dropout_mask, attn_weights once
and writing grad_attn_scores once.

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
def softmax_bwd_kernel(
    dP_dropped_ptr,   # [B80, sq, skv]  bfloat16
    dropout_mask_ptr, # [B80, sq, skv]  bool (uint8)
    P_ptr,            # [B80, sq, skv]  bfloat16
    dS_ptr,           # [B80, sq, skv]  bfloat16  (output)
    sq: tl.constexpr,
    skv: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """
    Each program handles one row: (batch*head, row_idx).
    Streams over the seq_kv dimension in tiles to compute the row sum,
    then writes dS in a second pass.
    """
    # pid 0 -> row index in [B80 * sq]
    row_id = tl.program_id(0)
    b80_idx = row_id // sq
    q_idx = row_id % sq

    row_offset = b80_idx * sq * skv + q_idx * skv

    # ---- First pass: compute row_sum = sum(dP * P) over skv ----
    row_sum = tl.zeros([1], dtype=tl.float32)

    for start in tl.range(0, skv, BLOCK_SKV):
        kv_ids = start + tl.arange(0, BLOCK_SKV)
        mask = kv_ids < skv
        offsets = row_offset + kv_ids

        dp_dropped = tl.load(dP_dropped_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        dmask = tl.load(dropout_mask_ptr + offsets, mask=mask, other=0).to(tl.float32)
        p = tl.load(P_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        dp = dp_dropped * dmask * scale
        row_sum += tl.sum(dp * p, axis=0)

    # ---- Second pass: compute dS = P * (dP - row_sum) and write ----
    for start in tl.range(0, skv, BLOCK_SKV):
        kv_ids = start + tl.arange(0, BLOCK_SKV)
        mask = kv_ids < skv
        offsets = row_offset + kv_ids

        dp_dropped = tl.load(dP_dropped_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        dmask = tl.load(dropout_mask_ptr + offsets, mask=mask, other=0).to(tl.float32)
        p = tl.load(P_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        dp = dp_dropped * dmask * scale
        ds = p * (dp - row_sum)

        tl.store(dS_ptr + offsets, ds.to(tl.bfloat16), mask=mask)


def fused_softmax_bwd(dP_dropped, dropout_mask, attn_weights, attention_dropout):
    """
    Fused softmax backward + dropout scaling.
    All inputs/output shaped [B80, sq, skv].
    """
    B80, sq, skv = dP_dropped.shape
    dS = torch.empty_like(dP_dropped)

    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Choose block size for skv dimension
    # Power of 2, at least covers common skv values
    if skv <= 64:
        BLOCK_SKV = 64
    elif skv <= 128:
        BLOCK_SKV = 128
    elif skv <= 256:
        BLOCK_SKV = 256
    elif skv <= 512:
        BLOCK_SKV = 512
    elif skv <= 1024:
        BLOCK_SKV = 1024
    else:
        BLOCK_SKV = 2048

    total_rows = B80 * sq
    grid = (total_rows,)

    softmax_bwd_kernel[grid](
        dP_dropped,
        dropout_mask,
        attn_weights,
        dS,
        sq=sq,
        skv=skv,
        scale=scale,
        BLOCK_SKV=BLOCK_SKV,
    )
    return dS


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs = grad_attn_output.shape[0]
    seq_q = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, sq, 80, d] -> [bs, 80, sq, d]
    dO = grad_attn_output.transpose(1, 2)  # bf16, [bs, 80, sq, d]

    # ------------------------------------------------------------------ #
    #  Compute dP_dropped = dO @ V^T  WITHOUT materializing expanded V
    #
    #  dO: [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    #  V:  [bs, 8, skv, d] -> [bs*8, d, skv]
    #  BMM: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
    #  reshape -> [bs, 80, sq, skv]
    # ------------------------------------------------------------------ #
    dO_grouped = dO.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM)
    dO_for_dP = dO_grouped.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS * seq_q, HEAD_DIM)

    V_flat_t = value_states.reshape(bs * NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM).transpose(-2, -1)

    dP_dropped_grouped = torch.bmm(dO_for_dP, V_flat_t)  # [B8, 10*sq, skv] bf16

    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv) \
                                   .reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ------------------------------------------------------------------ #
    #  Fused softmax backward via Triton kernel
    #  Reads dP_dropped, dropout_mask, attn_weights once; writes dS once
    # ------------------------------------------------------------------ #
    # Flatten to [B80, sq, skv] for the kernel
    dP_dropped_flat = dP_dropped.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv).contiguous()
    dropout_mask_flat = dropout_mask.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv).contiguous()
    attn_weights_flat = attn_weights.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv).contiguous()

    dS_flat = fused_softmax_bwd(dP_dropped_flat, dropout_mask_flat, attn_weights_flat, attention_dropout)
    dS = dS_flat.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ------------------------------------------------------------------ #
    #  Compute dV = attn_weights_dropped^T @ dO  (grouped, no V expansion)
    #
    #  attn_weights_dropped: [bs, 80, sq, skv] -> [bs*80, sq, skv]
    #  dO: [bs, 80, sq, d] -> [bs*80, sq, d]
    #  BMM: [B80, skv, sq] @ [B80, sq, d] -> [B80, skv, d]
    #  Sum over 10 groups: [B8, 10, skv, d] -> [B8, skv, d]
    # ------------------------------------------------------------------ #
    aw_dropped_flat = attn_weights_dropped.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv)
    dO_flat_kv = dO_grouped.reshape(bs * NUM_ATTENTION_HEADS, seq_q, HEAD_DIM)

    dV_flat = torch.bmm(aw_dropped_flat.transpose(-2, -1), dO_flat_kv)  # [B80, skv, d] bf16

    dV = dV_flat.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS, seq_kv, HEAD_DIM).sum(dim=1)
    dV = dV.reshape(bs, NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #8 — 2026-06-29 19:30:15 UTC 💥 CRASH

**Hypothesis:** ** Replaced the separate BMM + `fused_softmax_bwd` with a new `fused_dP_softmax_bwd_kernel` that:

**Result:** CRASH

**Error:**
```
run_eval exited 2
```

---

## Experiment #9 — 2026-06-29 19:31:55 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 631.17 μs

---

## Experiment #10 — 2026-06-29 19:33:36 UTC ❌ DISCARD

**Hypothesis:** ** Replaced the old dV path (B80 separate BMMs + explicit `sum(dim=1)` reduction) with a single grouped BMM: `attn_weights_dropped` is reshaped `[bs, 80, sq, skv] → [B8, 10*sq, skv]`, transposed to `[

**Result:** 1023.24 μs

---

## Experiment #11 — 2026-06-29 19:36:17 UTC ❌ DISCARD

**Hypothesis:** Cleaned up the implementation to:

**Result:** 1016.45 μs

---

## Experiment #12 — 2026-06-29 19:37:47 UTC 💥 CRASH

**Hypothesis:** Replaced `softmax_bwd_phase1_kernel` + `softmax_bwd_phase2_kernel` with a single `softmax_bwd_single_pass_kernel` that handles one row per program ID. The kernel uses two sequential `tl.range` loops o

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #13 — 2026-06-29 19:39:29 UTC 💥 CRASH

**Hypothesis:** ** Replaced the separate `torch.bmm` for dP + `fused_softmax_bwd` Triton kernel with a new `fused_dP_softmax_bwd_kernel` that:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

