# Experiment History

Tracks every kernel attempt, its code, hypothesis, and result.

---

## Experiment #1 — 2026-06-29 18:30:58 UTC ✅ KEEP

**Hypothesis:** Baseline 'starting_point' — initial benchmark

**Result:** 3430.34 μs

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

## Experiment #2 — 2026-06-29 18:32:33 UTC ✅ KEEP

**Hypothesis:** ** Wrapped `_attn_backward_impl` (which does all the math) in `torch.compile(mode="max-autotune", fullgraph=True)`, cached as `_compiled_attn_backward` at module load time. Changed to bf16 matmuls thr

**Result:** 891.01 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using torch.compile with max-autotune.

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


def _attn_backward_impl(
    grad_attn_output,   # [bs, sq, 80, 128]  bf16
    attn_weights,       # [bs, 80, sq, skv]  bf16
    attn_weights_dropped,  # [bs, 80, sq, skv]  bf16
    value_states,       # [bs, 8, skv, 128]  bf16
    dropout_mask,       # [bs, 80, sq, skv]  bool
    attention_dropout,  # float scalar
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # Transpose grad: [bs, sq, 80, d] -> [bs, 80, sq, d]
    dO = grad_attn_output.transpose(1, 2)  # bf16, [bs, 80, sq, d]

    # --- Compute dP_dropped = dO @ V^T in bf16 (avoid float32 cast) ---
    # Expand value_states for GQA: [bs, 8, skv, d] -> [bs, 80, skv, d]
    # Do it via reshape to avoid large intermediate
    # vs_exp: [bs, 80, skv, 128]
    vs_exp = value_states[:, :, None, :, :].expand(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_kv, HEAD_DIM
    ).reshape(bs, NUM_ATTENTION_HEADS, seq_kv, HEAD_DIM)

    # dP_dropped = dO @ vs_exp^T  -> [bs, 80, sq, skv]  in bf16
    dP_dropped = torch.matmul(dO, vs_exp.transpose(-2, -1))

    # --- Dropout backward ---
    # dropout_mask is bool; attention_dropout=0.1 => scale = 1/0.9
    dP = dP_dropped * dropout_mask * (1.0 / (1.0 - attention_dropout))

    # --- Softmax backward: dS = P * (dP - sum(dP * P, dim=-1, keepdim)) ---
    P = attn_weights  # bf16
    dPP = dP * P
    dS = P * (dP - dPP.sum(dim=-1, keepdim=True))
    dS = dS.to(torch.bfloat16)

    # --- Compute dV using grouped matmul to avoid large intermediate ---
    # Reshape attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    # Reshape dO:                   [bs, 80, sq, d]   -> [bs, 8, 10, sq, d]
    # dV_groups = awd_r^T @ dO_r -> [bs, 8, 10, skv, d]
    # dV = dV_groups.sum(dim=2)  -> [bs, 8, skv, d]
    awd_r = attn_weights_dropped.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv)
    dO_r  = dO.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM)

    # batched matmul: awd_r^T @ dO_r => [bs, 8, 10, skv, d]
    dV_groups = torch.matmul(awd_r.transpose(-2, -1), dO_r)

    # Sum over groups (dim=2)
    dV = dV_groups.sum(dim=2).to(torch.bfloat16)

    return dS, dV


# Compile once with max-autotune for best performance on B200
_compiled_attn_backward = torch.compile(
    _attn_backward_impl,
    mode="max-autotune",
    fullgraph=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    return _compiled_attn_backward(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        attention_dropout,
    )

```

---

## Experiment #3 — 2026-06-29 18:34:55 UTC ❌ DISCARD

**Hypothesis:** ** Replaced the `torch.compile`-based implementation with a custom Triton kernel `_attn_bwd_kernel` that:

**Result:** 3453.51 μs

---

## Experiment #4 — 2026-06-29 18:37:49 UTC ❌ DISCARD

**Hypothesis:** ** Replaced the broken Triton kernel with a `torch.compile`-compiled pure-PyTorch implementation that:

**Result:** 2290.97 μs

---

## Experiment #5 — 2026-06-29 18:40:40 UTC ❌ DISCARD

**Hypothesis:** Replaced the broadcast-based `[bs*8, 10, sq, d]` layout with a stacked `[bs*8, 10*sq, d]` layout for both GEMMs:

**Result:** 2215.34 μs

---

## Experiment #6 — 2026-06-29 18:41:58 UTC 💥 CRASH

**Hypothesis:** Added `_cuda_graph_cache` dict and `_get_or_build_graph()` function that manages per-shape CUDA graph capture and replay. The `custom_kernel` entry point now routes through this function instead of ca

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #7 — 2026-06-29 18:43:29 UTC ❌ DISCARD

**Hypothesis:** All three targeted changes applied exactly as specified. Added `.contiguous()` after `dO = grad_attn_output.transpose(1, 2)`, added `.contiguous()` after `vs_exp = ...reshape(...)`, and replaced the `

**Result:** 952.89 μs

---

## Experiment #8 — 2026-06-29 18:45:16 UTC ❌ DISCARD

**Hypothesis:** Added a Triton kernel `_softmax_dropout_bwd_kernel` that processes one row (b,h,q) per program in two passes: (1) compute sum(dP*P) across seq_kv, (2) compute dS = P*(dP - sum). The two matmuls remain

**Result:** 1267.15 μs

---

## Experiment #9 — 2026-06-29 18:47:49 UTC ❌ DISCARD

**Hypothesis:** ** Added a `_dv_kernel` Triton kernel with grid `(bs*n_kv_heads, ceil(skv/BLOCK_KV), ceil(d/BLOCK_D))`. Each program accumulates contributions from all 10 groups and all seq_q tiles using `tl.dot(tl.t

**Result:** 2436.54 μs

