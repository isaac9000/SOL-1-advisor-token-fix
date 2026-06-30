# Experiment History

Tracks every kernel attempt, its code, hypothesis, and result.

---

## Experiment #1 — 2026-06-29 22:39:22 UTC ✅ KEEP

**Hypothesis:** Baseline 'starting_point' — initial benchmark

**Result:** 3429.35 μs

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

## Experiment #2 — 2026-06-29 22:40:32 UTC 💥 CRASH

**Hypothesis:** **

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #3 — 2026-06-29 22:42:27 UTC ✅ KEEP

**Hypothesis:** ** Replaced the baseline with a `torch.compile`-wrapped implementation that:

**Result:** 2067.92 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using torch.compile + GQA-aware computation.

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


def _attention_backward_impl(
    grad_attn_output, attn_weights, attn_weights_dropped,
    value_states, dropout_mask, attention_dropout
):
    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # 1. Transpose grad: [bs, sq, h, d] -> [bs, h, sq, d]  (cast to f32)
    dO = grad_attn_output.transpose(1, 2).to(torch.float32)
    # contiguous for reshape
    dO = dO.contiguous()

    # Reshape dO to exploit GQA: [bs, 80, sq, d] -> [bs, 8, 10, sq, d]
    dO_grouped = dO.reshape(bs, n_kv_heads, n_groups, seq_q, HEAD_DIM)

    # === Compute dP_dropped avoiding 10x V expansion ===
    # value_states: [bs, 8, skv, d] -> transpose to [bs, 8, d, skv]
    # then unsqueeze for group broadcast: [bs, 8, 1, d, skv]
    vs_f32 = value_states.to(torch.float32)
    vs_T = vs_f32.transpose(-2, -1).unsqueeze(2)  # [bs, 8, 1, d, skv]

    # dP_dropped_grouped: [bs, 8, 10, sq, skv] = [bs,8,10,sq,d] @ [bs,8,1,d,skv]
    dP_dropped_grouped = torch.matmul(dO_grouped, vs_T)
    # reshape to [bs, 80, sq, skv]
    dP_dropped = dP_dropped_grouped.reshape(bs, n_heads, seq_q, seq_kv)

    # 3. Dropout backward
    if attention_dropout > 0.0:
        dP = dP_dropped * dropout_mask / (1.0 - attention_dropout)
    else:
        dP = dP_dropped

    # 4. Softmax backward: dS = P * (dP - sum(dP * P, dim=-1, keepdim=True))
    P = attn_weights.to(torch.float32)
    dPP = dP * P
    dS = P * (dP - dPP.sum(dim=-1, keepdim=True))
    dS = dS.to(torch.bfloat16)

    # === Compute dV in [bs, 8, skv, d] space directly (no 10x expansion) ===
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    P_dropped_grouped = attn_weights_dropped.to(torch.float32).reshape(
        bs, n_kv_heads, n_groups, seq_q, seq_kv
    )
    # dV_grouped = P_dropped_grouped^T @ dO_grouped: [bs, 8, 10, skv, d]
    dV_grouped = torch.matmul(
        P_dropped_grouped.transpose(-2, -1),  # [bs, 8, 10, skv, sq]
        dO_grouped                             # [bs, 8, 10, sq, d]
    )
    # Sum over groups: [bs, 8, 10, skv, d] -> [bs, 8, skv, d]
    dV = dV_grouped.sum(dim=2).to(torch.bfloat16)

    return dS, dV


# Compile the inner function for better performance
_compiled_attention_backward = torch.compile(
    _attention_backward_impl,
    mode="max-autotune",
    fullgraph=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    return _compiled_attention_backward(
        grad_attn_output, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, attention_dropout
    )

```

---

## Experiment #4 — 2026-06-29 22:44:00 UTC 💥 CRASH

**Hypothesis:** ** Two Triton kernels replacing the `torch.compile` baseline:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #5 — 2026-06-29 22:45:33 UTC 💥 CRASH

**Hypothesis:** ** Two Triton kernels replacing the `torch.compile` baseline:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #6 — 2026-06-29 22:47:42 UTC ✅ KEEP

**Hypothesis:** Two Triton kernels replacing elementwise/reduction PyTorch ops:

**Result:** 621.16 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- Keep both BMMs as bfloat16 torch.matmul (cuBLAS-optimized)
- Fused Triton kernel for elementwise dropout-bwd + softmax-bwd pass
- Triton kernel for GQA group-sum reduction (float32 accumulation, bf16 output)

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


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel 1: fused dropout-bwd + softmax-bwd
#
# For each row (bs * n_heads * sq), length seq_kv:
#   dP = dP_dropped * mask / (1 - p)
#   dS = P * (dP - sum(P * dP))
#
# Grid: (total_rows,)
# seq_kv passed as runtime arg; BLOCK_KV is constexpr tile size
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _softmax_bwd_kernel(
    dP_dropped_ptr,    # [total_rows, seq_kv]  bfloat16
    attn_weights_ptr,  # [total_rows, seq_kv]  bfloat16
    dropout_mask_ptr,  # [total_rows, seq_kv]  bool (uint8)
    dS_ptr,            # [total_rows, seq_kv]  bfloat16  (output)
    seq_kv,            # runtime int
    inv_keep_prob,     # runtime float32
    BLOCK_KV: tl.constexpr,
):
    row_idx = tl.program_id(0)
    base = row_idx * seq_kv

    # ── Pass 1: compute dot = sum(P * dP) ────────────────────────────────────
    dot = tl.zeros([1], dtype=tl.float32)
    for blk_start in tl.range(0, seq_kv, BLOCK_KV):
        offs = blk_start + tl.arange(0, BLOCK_KV)
        valid = offs < seq_kv

        dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
        dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
        P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

        dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
        dot += tl.sum(P_vals * dP_vals, axis=0)

    # ── Pass 2: compute dS = P * (dP - dot) and store ────────────────────────
    for blk_start in tl.range(0, seq_kv, BLOCK_KV):
        offs = blk_start + tl.arange(0, BLOCK_KV)
        valid = offs < seq_kv

        dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
        dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
        P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

        dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
        dS_vals = P_vals * (dP_vals - dot)
        tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel 2: GQA group sum reduction
#
# dV_grouped: [bs*8, n_groups, skv, d]  bfloat16
# dV:         [bs*8,           skv, d]  bfloat16
#
# Grid: (bs*8*skv,)  — each program handles one (bs_kv, skv_pos) row
# Accumulates over n_groups in float32, stores bf16.
# HEAD_DIM=128 fits in one BLOCK_D=128 tile.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _gqa_sum_kernel(
    dV_grouped_ptr,      # [bs*8, n_groups, skv, d]  bfloat16
    dV_ptr,              # [bs*8, skv, d]             bfloat16
    n_groups: tl.constexpr,
    skv,                 # runtime int
    HEAD_DIM: tl.constexpr,
):
    row_idx = tl.program_id(0)   # in [0, bs*8*skv)

    bs_kv_idx = row_idx // skv   # index into (bs*8) space
    skv_pos   = row_idx % skv

    offs_d = tl.arange(0, HEAD_DIM)

    # Accumulate over n_groups in float32
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for g in tl.range(0, n_groups):
        # dV_grouped layout: [bs*8, n_groups, skv, d]
        # flat index: (bs_kv_idx * n_groups + g) * skv + skv_pos  — then * HEAD_DIM
        grouped_row = (bs_kv_idx * n_groups + g) * skv + skv_pos
        ptr = dV_grouped_ptr + grouped_row * HEAD_DIM + offs_d
        val = tl.load(ptr).to(tl.float32)
        acc += val

    # Output: [bs*8, skv, d] — row = bs_kv_idx * skv + skv_pos
    out_row = bs_kv_idx * skv + skv_pos
    tl.store(dV_ptr + out_row * HEAD_DIM + offs_d, acc.to(tl.bfloat16))


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # ── Step 1: Transpose grad and reshape for grouped computation ───────────
    # [bs, sq, 80, d] -> [bs, 80, sq, d] (contiguous bfloat16)
    dO = grad_attn_output.transpose(1, 2).contiguous()
    # [bs, 80, sq, d] -> [bs, 8, 10, sq, d]
    dO_grouped = dO.reshape(bs, n_kv_heads, n_groups, seq_q, HEAD_DIM)

    # ── Step 2: BMM1 — compute dP_dropped (bfloat16 matmul) ─────────────────
    # vs_T: [bs, 8, 1, d, skv]  (bfloat16)
    vs_T = value_states.transpose(-2, -1).unsqueeze(2)
    # dP_dropped_grouped: [bs, 8, 10, sq, skv]  bfloat16
    dP_dropped_grouped = torch.matmul(dO_grouped, vs_T)
    # Reshape to [bs, 80, sq, skv]
    dP_dropped = dP_dropped_grouped.reshape(bs, n_heads, seq_q, seq_kv)

    # ── Step 3: Fused Triton kernel — dropout bwd + softmax bwd ─────────────
    total_rows = bs * n_heads * seq_q
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    # Flatten to [total_rows, seq_kv]
    dP_dropped_flat   = dP_dropped.reshape(total_rows, seq_kv).contiguous()
    attn_weights_flat = attn_weights.reshape(total_rows, seq_kv).contiguous()
    dropout_mask_flat = dropout_mask.reshape(total_rows, seq_kv).contiguous()
    dS_flat           = dS.reshape(total_rows, seq_kv)

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # BLOCK_KV: next power of 2 >= seq_kv, max 4096 (loop handles larger)
    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 4096)

    _softmax_bwd_kernel[(total_rows,)](
        dP_dropped_flat,
        attn_weights_flat,
        dropout_mask_flat,
        dS_flat,
        seq_kv,
        inv_keep_prob,
        BLOCK_KV=BLOCK_KV,
    )

    # ── Step 4: BMM2 — compute dV_grouped (bfloat16 matmul) ─────────────────
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    P_dropped_grouped = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    # dV_grouped: [bs, 8, 10, skv, d]
    dV_grouped = torch.matmul(
        P_dropped_grouped.transpose(-2, -1),  # [bs, 8, 10, skv, sq]
        dO_grouped                             # [bs, 8, 10, sq, d]
    )

    # ── Step 5: Triton GQA sum reduction ─────────────────────────────────────
    dV = torch.empty((bs, n_kv_heads, seq_kv, HEAD_DIM), dtype=torch.bfloat16, device=dO.device)

    # Flatten: dV_grouped [bs, 8, 10, skv, d] -> [bs*8, 10, skv, d]
    dV_grouped_flat = dV_grouped.reshape(bs * n_kv_heads, n_groups, seq_kv, HEAD_DIM).contiguous()
    dV_flat = dV.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)

    total_out_rows = bs * n_kv_heads * seq_kv

    _gqa_sum_kernel[(total_out_rows,)](
        dV_grouped_flat,
        dV_flat,
        n_groups=n_groups,
        skv=seq_kv,
        HEAD_DIM=HEAD_DIM,
    )

    return dS, dV

```

---

## Experiment #7 — 2026-06-29 22:49:34 UTC ❌ DISCARD

**Hypothesis:** ** Replaced the two-step process (torch.matmul BMM1 → Triton softmax-bwd) with a single `_fused_bmm1_softmax_bwd_kernel` Triton kernel. Each program handles one row `(bs, head, sq_pos)`: loads `dO_row

**Result:** 5059.54 μs

---

## Experiment #8 — 2026-06-29 22:50:50 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 752.86 μs

---

## Experiment #9 — 2026-06-29 22:52:24 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 724.72 μs

---

## Experiment #10 — 2026-06-29 22:54:08 UTC ✅ KEEP

**Hypothesis:** ** The fallback proposal — made the softmax-bwd Triton kernel explicitly single-pass by:

**Result:** 597.81 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- Keep both BMMs as bfloat16 torch.matmul (cuBLAS-optimized)
- Single-pass Triton kernel for elementwise dropout-bwd + softmax-bwd:
  * When seq_kv fits in BLOCK_KV tiles: load dP_dropped+mask+P ONCE,
    compute partial sums, warp-reduce to get dot, write dS — NO second pass
  * For very large seq_kv: fall back to two-pass (rare)
- GQA group-sum via torch.sum (float32 accumulation, bf16 output)

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


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel 1: single-pass fused dropout-bwd + softmax-bwd
#
# For each row (bs * n_heads * sq), length seq_kv:
#   dP = dP_dropped * mask / (1 - p)
#   dS = P * (dP - sum(P * dP))
#
# Single-pass strategy (SINGLE_PASS=True, seq_kv <= BLOCK_KV):
#   - Load dP_dropped, mask, P once into registers
#   - Compute dot = sum(P * dP) using tl.sum (already a reduction over the tile)
#   - Immediately write dS = P * (dP - dot)
#   - Zero extra memory traffic vs. two-pass (which reads all three arrays twice)
#
# Two-pass fallback (SINGLE_PASS=False, seq_kv > BLOCK_KV):
#   - Used only for unusually large seq_kv
#
# Grid: (total_rows,)  one program per (bs, head, sq) row
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _softmax_bwd_kernel(
    dP_dropped_ptr,    # [total_rows, seq_kv]  bfloat16
    attn_weights_ptr,  # [total_rows, seq_kv]  bfloat16
    dropout_mask_ptr,  # [total_rows, seq_kv]  bool (uint8)
    dS_ptr,            # [total_rows, seq_kv]  bfloat16  (output)
    seq_kv,            # runtime int
    inv_keep_prob,     # runtime float32
    BLOCK_KV: tl.constexpr,
    SINGLE_PASS: tl.constexpr,  # True when seq_kv <= BLOCK_KV
):
    row_idx = tl.program_id(0)
    base = row_idx * seq_kv

    if SINGLE_PASS:
        # ── Single pass: load once, compute dot, write dS ────────────────────
        # All data for this row fits in BLOCK_KV registers — zero re-read
        offs = tl.arange(0, BLOCK_KV)
        valid = offs < seq_kv

        dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
        dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
        P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

        # Apply dropout scaling
        dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)

        # Compute dot product (scalar reduction over the tile)
        dot = tl.sum(P_vals * dP_vals, axis=0)

        # Compute and store dS — data already in registers, no re-load needed
        dS_vals = P_vals * (dP_vals - dot)
        tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)
    else:
        # ── Two-pass fallback for very large seq_kv ──────────────────────────
        # Pass 1: compute dot = sum(P * dP)
        dot = tl.zeros([1], dtype=tl.float32)
        for blk_start in tl.range(0, seq_kv, BLOCK_KV):
            offs = blk_start + tl.arange(0, BLOCK_KV)
            valid = offs < seq_kv

            dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
            dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
            P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

            dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
            dot += tl.sum(P_vals * dP_vals, axis=0)

        # Pass 2: compute dS = P * (dP - dot) and store
        for blk_start in tl.range(0, seq_kv, BLOCK_KV):
            offs = blk_start + tl.arange(0, BLOCK_KV)
            valid = offs < seq_kv

            dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
            dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
            P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

            dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
            dS_vals = P_vals * (dP_vals - dot)
            tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel 2: GQA group sum reduction
#
# dV_grouped: [bs*8, n_groups, skv, d]  bfloat16
# dV:         [bs*8,           skv, d]  bfloat16
#
# Grid: (bs*8*skv,)  — each program handles one (bs_kv, skv_pos) row of d=128
# Accumulates over n_groups in float32, stores bf16.
# HEAD_DIM=128 fits in one BLOCK_D=128 tile.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _gqa_sum_kernel(
    dV_grouped_ptr,      # [bs*8, n_groups, skv, d]  bfloat16
    dV_ptr,              # [bs*8, skv, d]             bfloat16
    n_groups: tl.constexpr,
    skv,                 # runtime int
    HEAD_DIM: tl.constexpr,
):
    row_idx = tl.program_id(0)   # in [0, bs*8*skv)

    bs_kv_idx = row_idx // skv   # index into (bs*8) space
    skv_pos   = row_idx % skv

    offs_d = tl.arange(0, HEAD_DIM)

    # Accumulate over n_groups in float32
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for g in tl.range(0, n_groups):
        # dV_grouped layout: [bs*8, n_groups, skv, d]
        grouped_row = (bs_kv_idx * n_groups + g) * skv + skv_pos
        ptr = dV_grouped_ptr + grouped_row * HEAD_DIM + offs_d
        val = tl.load(ptr).to(tl.float32)
        acc += val

    # Output: [bs*8, skv, d] — row = bs_kv_idx * skv + skv_pos
    out_row = bs_kv_idx * skv + skv_pos
    tl.store(dV_ptr + out_row * HEAD_DIM + offs_d, acc.to(tl.bfloat16))


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # ── Step 1: Transpose grad and reshape for grouped computation ───────────
    # [bs, sq, 80, d] -> [bs, 80, sq, d] (contiguous bfloat16)
    dO = grad_attn_output.transpose(1, 2).contiguous()
    # [bs, 80, sq, d] -> [bs, 8, 10, sq, d]
    dO_grouped = dO.reshape(bs, n_kv_heads, n_groups, seq_q, HEAD_DIM)

    # ── Step 2: BMM1 — compute dP_dropped (bfloat16 matmul) ─────────────────
    # vs_T: [bs, 8, 1, d, skv]  (bfloat16)
    vs_T = value_states.transpose(-2, -1).unsqueeze(2)
    # dP_dropped_grouped: [bs, 8, 10, sq, skv]  bfloat16
    dP_dropped_grouped = torch.matmul(dO_grouped, vs_T)
    # Reshape to [bs, 80, sq, skv] — zero-copy view
    dP_dropped = dP_dropped_grouped.reshape(bs, n_heads, seq_q, seq_kv)

    # ── Step 3: Single-pass Triton kernel — dropout bwd + softmax bwd ────────
    total_rows = bs * n_heads * seq_q
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    # Flatten to [total_rows, seq_kv]
    dP_dropped_flat   = dP_dropped.reshape(total_rows, seq_kv).contiguous()
    attn_weights_flat = attn_weights.reshape(total_rows, seq_kv).contiguous()
    dropout_mask_flat = dropout_mask.reshape(total_rows, seq_kv).contiguous()
    dS_flat           = dS.reshape(total_rows, seq_kv)

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # Choose BLOCK_KV: next power of 2 >= seq_kv, capped at 16384 for registers
    # Larger BLOCK_KV enables single-pass for bigger seq_kv values
    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 16384)
    # SINGLE_PASS: True when all seq_kv elements fit in one tile
    SINGLE_PASS = (seq_kv <= BLOCK_KV)

    _softmax_bwd_kernel[(total_rows,)](
        dP_dropped_flat,
        attn_weights_flat,
        dropout_mask_flat,
        dS_flat,
        seq_kv,
        inv_keep_prob,
        BLOCK_KV=BLOCK_KV,
        SINGLE_PASS=SINGLE_PASS,
    )

    # ── Step 4: BMM2 — compute dV_grouped (bfloat16 matmul) ─────────────────
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    P_dropped_grouped = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    # dV_grouped: [bs, 8, 10, skv, d]
    dV_grouped = torch.matmul(
        P_dropped_grouped.transpose(-2, -1),  # [bs, 8, 10, skv, sq]
        dO_grouped                             # [bs, 8, 10, sq, d]
    )

    # ── Step 5: GQA sum reduction via Triton kernel ───────────────────────────
    dV = torch.empty((bs, n_kv_heads, seq_kv, HEAD_DIM), dtype=torch.bfloat16, device=dO.device)

    # Flatten: dV_grouped [bs, 8, 10, skv, d] -> [bs*8, 10, skv, d]
    dV_grouped_flat = dV_grouped.reshape(bs * n_kv_heads, n_groups, seq_kv, HEAD_DIM).contiguous()
    dV_flat = dV.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)

    total_out_rows = bs * n_kv_heads * seq_kv

    _gqa_sum_kernel[(total_out_rows,)](
        dV_grouped_flat,
        dV_flat,
        n_groups=n_groups,
        skv=seq_kv,
        HEAD_DIM=HEAD_DIM,
    )

    return dS, dV

```

---

## Experiment #11 — 2026-06-29 22:55:42 UTC ✅ KEEP

**Hypothesis:** **

**Result:** 573.06 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- Keep both BMMs as bfloat16 torch.matmul (cuBLAS-optimized)
- Single-pass Triton kernel for elementwise dropout-bwd + softmax-bwd:
  * When seq_kv fits in BLOCK_KV tiles: load dP_dropped+mask+P ONCE,
    compute partial sums, warp-reduce to get dot, write dS — NO second pass
  * For very large seq_kv: fall back to two-pass (rare)
- BMM2 fused with GQA reduction: single large GEMM with K = 10*sq
  instead of 10 separate GEMMs + Triton sum kernel

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


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel 1: single-pass fused dropout-bwd + softmax-bwd
#
# For each row (bs * n_heads * sq), length seq_kv:
#   dP = dP_dropped * mask / (1 - p)
#   dS = P * (dP - sum(P * dP))
#
# Single-pass strategy (SINGLE_PASS=True, seq_kv <= BLOCK_KV):
#   - Load dP_dropped, mask, P once into registers
#   - Compute dot = sum(P * dP) using tl.sum (already a reduction over the tile)
#   - Immediately write dS = P * (dP - dot)
#   - Zero extra memory traffic vs. two-pass (which reads all three arrays twice)
#
# Two-pass fallback (SINGLE_PASS=False, seq_kv > BLOCK_KV):
#   - Used only for unusually large seq_kv
#
# Grid: (total_rows,)  one program per (bs, head, sq) row
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _softmax_bwd_kernel(
    dP_dropped_ptr,    # [total_rows, seq_kv]  bfloat16
    attn_weights_ptr,  # [total_rows, seq_kv]  bfloat16
    dropout_mask_ptr,  # [total_rows, seq_kv]  bool (uint8)
    dS_ptr,            # [total_rows, seq_kv]  bfloat16  (output)
    seq_kv,            # runtime int
    inv_keep_prob,     # runtime float32
    BLOCK_KV: tl.constexpr,
    SINGLE_PASS: tl.constexpr,  # True when seq_kv <= BLOCK_KV
):
    row_idx = tl.program_id(0)
    base = row_idx * seq_kv

    if SINGLE_PASS:
        # ── Single pass: load once, compute dot, write dS ────────────────────
        # All data for this row fits in BLOCK_KV registers — zero re-read
        offs = tl.arange(0, BLOCK_KV)
        valid = offs < seq_kv

        dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
        dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
        P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

        # Apply dropout scaling
        dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)

        # Compute dot product (scalar reduction over the tile)
        dot = tl.sum(P_vals * dP_vals, axis=0)

        # Compute and store dS — data already in registers, no re-load needed
        dS_vals = P_vals * (dP_vals - dot)
        tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)
    else:
        # ── Two-pass fallback for very large seq_kv ──────────────────────────
        # Pass 1: compute dot = sum(P * dP)
        dot = tl.zeros([1], dtype=tl.float32)
        for blk_start in tl.range(0, seq_kv, BLOCK_KV):
            offs = blk_start + tl.arange(0, BLOCK_KV)
            valid = offs < seq_kv

            dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
            dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
            P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

            dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
            dot += tl.sum(P_vals * dP_vals, axis=0)

        # Pass 2: compute dS = P * (dP - dot) and store
        for blk_start in tl.range(0, seq_kv, BLOCK_KV):
            offs = blk_start + tl.arange(0, BLOCK_KV)
            valid = offs < seq_kv

            dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
            dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
            P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

            dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
            dS_vals = P_vals * (dP_vals - dot)
            tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # ── Step 1: Transpose grad and reshape for grouped computation ───────────
    # [bs, sq, 80, d] -> [bs, 80, sq, d] (contiguous bfloat16)
    dO = grad_attn_output.transpose(1, 2).contiguous()
    # [bs, 80, sq, d] -> [bs, 8, 10, sq, d]
    dO_grouped = dO.reshape(bs, n_kv_heads, n_groups, seq_q, HEAD_DIM)

    # ── Step 2: BMM1 — compute dP_dropped (bfloat16 matmul) ─────────────────
    # vs_T: [bs, 8, 1, d, skv]  (bfloat16)
    vs_T = value_states.transpose(-2, -1).unsqueeze(2)
    # dP_dropped_grouped: [bs, 8, 10, sq, skv]  bfloat16
    dP_dropped_grouped = torch.matmul(dO_grouped, vs_T)
    # Reshape to [bs, 80, sq, skv] — zero-copy view
    dP_dropped = dP_dropped_grouped.reshape(bs, n_heads, seq_q, seq_kv)

    # ── Step 3: Single-pass Triton kernel — dropout bwd + softmax bwd ────────
    total_rows = bs * n_heads * seq_q
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    # Flatten to [total_rows, seq_kv]
    # dP_dropped was reshaped from a contiguous grouped tensor — check if already contiguous
    dP_dropped_flat   = dP_dropped.reshape(total_rows, seq_kv)
    if not dP_dropped_flat.is_contiguous():
        dP_dropped_flat = dP_dropped_flat.contiguous()
    attn_weights_flat = attn_weights.reshape(total_rows, seq_kv)
    if not attn_weights_flat.is_contiguous():
        attn_weights_flat = attn_weights_flat.contiguous()
    dropout_mask_flat = dropout_mask.reshape(total_rows, seq_kv)
    if not dropout_mask_flat.is_contiguous():
        dropout_mask_flat = dropout_mask_flat.contiguous()
    dS_flat           = dS.reshape(total_rows, seq_kv)

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # Choose BLOCK_KV: next power of 2 >= seq_kv, capped at 16384 for registers
    # Larger BLOCK_KV enables single-pass for bigger seq_kv values
    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 16384)
    # SINGLE_PASS: True when all seq_kv elements fit in one tile
    SINGLE_PASS = (seq_kv <= BLOCK_KV)

    _softmax_bwd_kernel[(total_rows,)](
        dP_dropped_flat,
        attn_weights_flat,
        dropout_mask_flat,
        dS_flat,
        seq_kv,
        inv_keep_prob,
        BLOCK_KV=BLOCK_KV,
        SINGLE_PASS=SINGLE_PASS,
    )

    # ── Step 4: Fused BMM2 + GQA reduction — single large GEMM ──────────────
    #
    # Instead of:
    #   dV_grouped = bmm(P_dropped_grouped.T, dO_grouped)  [bs,8,10,skv,d]
    #   dV = dV_grouped.sum(dim=2)                          [bs,8,skv,d]
    #
    # We reshape to merge the groups into the K dimension:
    #   P_dropped: [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv] -> transpose -> [bs*8, skv, 10*sq]
    #   dO:        [bs, 8, 10, sq, d]   -> [bs*8, 10*sq, d]
    #   result:    [bs*8, skv, d]  (one GEMM with K = 10*sq, replaces 10 GEMMs + reduction)
    #
    # This is mathematically equivalent:
    #   dV[b,kv,s,d] = sum_{g,q} P_dropped[b,kv,g,q,s] * dO[b,kv,g,q,d]
    #                = P_dropped_2d^T @ dO_2d   where K = n_groups * seq_q

    # attn_weights_dropped: [bs, 80, sq, skv] — reshape to [bs*8, 10*sq, skv]
    # Need contiguous for the reshape+transpose
    P_dropped_2d = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    # Make contiguous so transpose+bmm works efficiently
    if not P_dropped_2d.is_contiguous():
        P_dropped_2d = P_dropped_2d.contiguous()
    # Transpose: [bs*8, skv, 10*sq]
    P_dropped_2d_T = P_dropped_2d.transpose(-2, -1)  # non-contiguous view, bmm handles it

    # dO: [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    dO_2d = dO_grouped.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)
    if not dO_2d.is_contiguous():
        dO_2d = dO_2d.contiguous()

    # Single BMM: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
    dV_flat = torch.bmm(P_dropped_2d_T, dO_2d)

    # Reshape to final output shape [bs, 8, skv, d]
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #12 — 2026-06-29 22:57:38 UTC ❌ DISCARD

**Hypothesis:** ** Replaced the single `_softmax_bwd_kernel` with three kernels:

**Result:** 597.09 μs

---

## Experiment #13 — 2026-06-29 22:59:14 UTC ✅ KEEP

**Hypothesis:** ** Replaced the old BMM1 path (`dO_grouped [bs,8,10,sq,d] @ vs_T [bs,8,1,d,skv]` with 5D broadcast) with a clean 3D batched GEMM: `torch.bmm(dO_2d [bs*8, 10*sq, d], vs_T_2d [bs*8, d, skv])` → `[bs*8, 

**Result:** 436.23 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- Both BMMs as clean 3D batched GEMMs (cuBLAS-optimized, no broadcasting)
- BMM1 restructured: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
  then reshape to [bs, 80, sq, skv] — same K-merging trick as BMM2
- BMM2 fused with GQA reduction: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
- dO_2d [bs*8, 10*sq, d] reused across both BMMs (computed once)
- Single-pass Triton kernel for elementwise dropout-bwd + softmax-bwd

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


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: single-pass fused dropout-bwd + softmax-bwd
#
# For each row (bs * n_heads * sq), length seq_kv:
#   dP = dP_dropped * mask / (1 - p)
#   dS = P * (dP - sum(P * dP))
#
# Single-pass strategy (SINGLE_PASS=True, seq_kv <= BLOCK_KV):
#   - Load dP_dropped, mask, P once into registers
#   - Compute dot = sum(P * dP) using tl.sum (already a reduction over the tile)
#   - Immediately write dS = P * (dP - dot)
#   - Zero extra memory traffic vs. two-pass (which reads all three arrays twice)
#
# Two-pass fallback (SINGLE_PASS=False, seq_kv > BLOCK_KV):
#   - Used only for unusually large seq_kv
#
# Grid: (total_rows,)  one program per (bs, head, sq) row
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _softmax_bwd_kernel(
    dP_dropped_ptr,    # [total_rows, seq_kv]  bfloat16
    attn_weights_ptr,  # [total_rows, seq_kv]  bfloat16
    dropout_mask_ptr,  # [total_rows, seq_kv]  bool (uint8)
    dS_ptr,            # [total_rows, seq_kv]  bfloat16  (output)
    seq_kv,            # runtime int
    inv_keep_prob,     # runtime float32
    BLOCK_KV: tl.constexpr,
    SINGLE_PASS: tl.constexpr,  # True when seq_kv <= BLOCK_KV
):
    row_idx = tl.program_id(0)
    base = row_idx * seq_kv

    if SINGLE_PASS:
        # ── Single pass: load once, compute dot, write dS ────────────────────
        # All data for this row fits in BLOCK_KV registers — zero re-read
        offs = tl.arange(0, BLOCK_KV)
        valid = offs < seq_kv

        dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
        dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
        P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

        # Apply dropout scaling
        dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)

        # Compute dot product (scalar reduction over the tile)
        dot = tl.sum(P_vals * dP_vals, axis=0)

        # Compute and store dS — data already in registers, no re-load needed
        dS_vals = P_vals * (dP_vals - dot)
        tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)
    else:
        # ── Two-pass fallback for very large seq_kv ──────────────────────────
        # Pass 1: compute dot = sum(P * dP)
        dot = tl.zeros([1], dtype=tl.float32)
        for blk_start in tl.range(0, seq_kv, BLOCK_KV):
            offs = blk_start + tl.arange(0, BLOCK_KV)
            valid = offs < seq_kv

            dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
            dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
            P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

            dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
            dot += tl.sum(P_vals * dP_vals, axis=0)

        # Pass 2: compute dS = P * (dP - dot) and store
        for blk_start in tl.range(0, seq_kv, BLOCK_KV):
            offs = blk_start + tl.arange(0, BLOCK_KV)
            valid = offs < seq_kv

            dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
            dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
            P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

            dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
            dS_vals = P_vals * (dP_vals - dot)
            tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # ── Step 1: Transpose grad and build dO_2d for both BMMs ─────────────────
    # [bs, sq, 80, d] -> [bs, 80, sq, d] (contiguous bfloat16)
    dO = grad_attn_output.transpose(1, 2).contiguous()
    # [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    # This single reshape is reused for BOTH BMM1 and BMM2
    dO_2d = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)
    if not dO_2d.is_contiguous():
        dO_2d = dO_2d.contiguous()

    # ── Step 2: BMM1 — clean 3D batched GEMM (no broadcasting) ──────────────
    #
    # Old approach: dO_grouped [bs,8,10,sq,d] @ vs_T [bs,8,1,d,skv] (broadcast)
    # New approach: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
    # then reshape to [bs, 80, sq, skv]
    #
    # vs_T_2d: [bs, 8, skv, d] -> transpose -> [bs, 8, d, skv] -> [bs*8, d, skv]
    vs_T_2d = value_states.transpose(-2, -1).reshape(bs * n_kv_heads, HEAD_DIM, seq_kv)
    if not vs_T_2d.is_contiguous():
        vs_T_2d = vs_T_2d.contiguous()

    # Single clean 3D BMM: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
    dP_dropped_2d = torch.bmm(dO_2d, vs_T_2d)

    # Reshape to [bs, 80, sq, skv] for the Triton softmax-bwd kernel
    dP_dropped = dP_dropped_2d.reshape(bs, n_heads, seq_q, seq_kv)

    # ── Step 3: Single-pass Triton kernel — dropout bwd + softmax bwd ────────
    total_rows = bs * n_heads * seq_q
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    # Flatten to [total_rows, seq_kv]
    dP_dropped_flat = dP_dropped.reshape(total_rows, seq_kv)
    if not dP_dropped_flat.is_contiguous():
        dP_dropped_flat = dP_dropped_flat.contiguous()
    attn_weights_flat = attn_weights.reshape(total_rows, seq_kv)
    if not attn_weights_flat.is_contiguous():
        attn_weights_flat = attn_weights_flat.contiguous()
    dropout_mask_flat = dropout_mask.reshape(total_rows, seq_kv)
    if not dropout_mask_flat.is_contiguous():
        dropout_mask_flat = dropout_mask_flat.contiguous()
    dS_flat = dS.reshape(total_rows, seq_kv)

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # Choose BLOCK_KV: next power of 2 >= seq_kv, capped at 16384 for registers
    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 16384)
    SINGLE_PASS = (seq_kv <= BLOCK_KV)

    _softmax_bwd_kernel[(total_rows,)](
        dP_dropped_flat,
        attn_weights_flat,
        dropout_mask_flat,
        dS_flat,
        seq_kv,
        inv_keep_prob,
        BLOCK_KV=BLOCK_KV,
        SINGLE_PASS=SINGLE_PASS,
    )

    # ── Step 4: Fused BMM2 + GQA reduction — single large GEMM ──────────────
    #
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs*8, 10*sq, skv]
    # dO_2d: [bs*8, 10*sq, d]  (already computed above)
    #
    # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
    # Mathematically: dV[b,kv,s,d] = sum_{g,q} P_dropped[b,kv,g,q,s] * dO[b,kv,g,q,d]

    P_dropped_2d = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    if not P_dropped_2d.is_contiguous():
        P_dropped_2d = P_dropped_2d.contiguous()
    # Transpose: [bs*8, skv, 10*sq]
    P_dropped_2d_T = P_dropped_2d.transpose(-2, -1)  # non-contiguous view, bmm handles it

    # Single BMM: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
    dV_flat = torch.bmm(P_dropped_2d_T, dO_2d)

    # Reshape to final output shape [bs, 8, skv, d]
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #14 — 2026-06-29 23:00:55 UTC ✅ KEEP

**Hypothesis:** ** Replaced the 1D grid `(total_rows,)` softmax-bwd kernel with a row-batched version using `ROWS_PER_CTA=4`. Grid size becomes `ceil(total_rows / 4)`. The kernel uses `tl.static_range(ROWS_PER_CTA)` 

**Result:** 426.75 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- Both BMMs as clean 3D batched GEMMs (cuBLAS-optimized, no broadcasting)
- BMM1 restructured: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
  then reshape to [bs, 80, sq, skv] — same K-merging trick as BMM2
- BMM2 fused with GQA reduction: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
- dO_2d [bs*8, 10*sq, d] reused across both BMMs (computed once)
- Row-batched Triton kernel for elementwise dropout-bwd + softmax-bwd:
  * ROWS_PER_CTA rows processed per program to increase SM occupancy
  * Each row is handled independently within the CTA, amortizing launch overhead

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


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: row-batched fused dropout-bwd + softmax-bwd
#
# For each row (bs * n_heads * sq), length seq_kv:
#   dP = dP_dropped * mask / (1 - p)
#   dS = P * (dP - sum(P * dP))
#
# Row-batching strategy (ROWS_PER_CTA > 1):
#   - Each program handles ROWS_PER_CTA consecutive rows
#   - Grid size = ceil(total_rows / ROWS_PER_CTA)
#   - Increases SM occupancy by reducing kernel launch overhead and
#     giving each CTA more work (better warp utilization)
#
# For single-pass (seq_kv <= BLOCK_KV):
#   - Load dP_dropped, mask, P once per row into registers
#   - Compute dot via tl.sum, write dS — no re-read
#
# Two-pass fallback (seq_kv > BLOCK_KV):
#   - Pass 1: compute partial dot sums, Pass 2: write dS
#
# Grid: (ceil(total_rows / ROWS_PER_CTA),)
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _softmax_bwd_kernel(
    dP_dropped_ptr,    # [total_rows, seq_kv]  bfloat16
    attn_weights_ptr,  # [total_rows, seq_kv]  bfloat16
    dropout_mask_ptr,  # [total_rows, seq_kv]  bool (uint8)
    dS_ptr,            # [total_rows, seq_kv]  bfloat16  (output)
    total_rows,        # runtime int
    seq_kv,            # runtime int
    inv_keep_prob,     # runtime float32
    BLOCK_KV: tl.constexpr,
    SINGLE_PASS: tl.constexpr,  # True when seq_kv <= BLOCK_KV
    ROWS_PER_CTA: tl.constexpr,  # number of rows per CTA
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_CTA

    # Process ROWS_PER_CTA rows per CTA
    for i in tl.static_range(ROWS_PER_CTA):
        row_idx = row_start + i
        # Guard: skip rows beyond total_rows
        if row_idx < total_rows:
            base = row_idx * seq_kv

            if SINGLE_PASS:
                # ── Single pass: load once, compute dot, write dS ────────────
                offs = tl.arange(0, BLOCK_KV)
                valid = offs < seq_kv

                dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                dot = tl.sum(P_vals * dP_vals, axis=0)
                dS_vals = P_vals * (dP_vals - dot)
                tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)
            else:
                # ── Two-pass fallback for very large seq_kv ──────────────────
                dot = tl.zeros([1], dtype=tl.float32)
                for blk_start in tl.range(0, seq_kv, BLOCK_KV):
                    offs = blk_start + tl.arange(0, BLOCK_KV)
                    valid = offs < seq_kv

                    dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                    dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                    P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                    dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                    dot += tl.sum(P_vals * dP_vals, axis=0)

                for blk_start in tl.range(0, seq_kv, BLOCK_KV):
                    offs = blk_start + tl.arange(0, BLOCK_KV)
                    valid = offs < seq_kv

                    dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                    dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                    P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                    dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                    dS_vals = P_vals * (dP_vals - dot)
                    tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # ── Step 1: Transpose grad and build dO_2d for both BMMs ─────────────────
    # [bs, sq, 80, d] -> [bs, 80, sq, d] (contiguous bfloat16)
    dO = grad_attn_output.transpose(1, 2).contiguous()
    # [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    # This single reshape is reused for BOTH BMM1 and BMM2
    dO_2d = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)
    if not dO_2d.is_contiguous():
        dO_2d = dO_2d.contiguous()

    # ── Step 2: BMM1 — clean 3D batched GEMM (no broadcasting) ──────────────
    #
    # vs_T_2d: [bs, 8, skv, d] -> transpose -> [bs, 8, d, skv] -> [bs*8, d, skv]
    vs_T_2d = value_states.transpose(-2, -1).reshape(bs * n_kv_heads, HEAD_DIM, seq_kv)
    if not vs_T_2d.is_contiguous():
        vs_T_2d = vs_T_2d.contiguous()

    # Single clean 3D BMM: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
    dP_dropped_2d = torch.bmm(dO_2d, vs_T_2d)

    # Reshape to [bs, 80, sq, skv] for the Triton softmax-bwd kernel
    dP_dropped = dP_dropped_2d.reshape(bs, n_heads, seq_q, seq_kv)

    # ── Step 3: Row-batched Triton kernel — dropout bwd + softmax bwd ────────
    total_rows = bs * n_heads * seq_q
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    # Flatten to [total_rows, seq_kv]
    dP_dropped_flat = dP_dropped.reshape(total_rows, seq_kv)
    if not dP_dropped_flat.is_contiguous():
        dP_dropped_flat = dP_dropped_flat.contiguous()
    attn_weights_flat = attn_weights.reshape(total_rows, seq_kv)
    if not attn_weights_flat.is_contiguous():
        attn_weights_flat = attn_weights_flat.contiguous()
    dropout_mask_flat = dropout_mask.reshape(total_rows, seq_kv)
    if not dropout_mask_flat.is_contiguous():
        dropout_mask_flat = dropout_mask_flat.contiguous()
    dS_flat = dS.reshape(total_rows, seq_kv)

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # Choose BLOCK_KV: next power of 2 >= seq_kv, capped at 16384 for registers
    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 16384)
    SINGLE_PASS = (seq_kv <= BLOCK_KV)

    # ROWS_PER_CTA: batch multiple rows per CTA to increase SM occupancy
    # Using 4 rows per CTA — amortizes launch overhead, increases warp utilization
    ROWS_PER_CTA = 4
    grid_size = triton.cdiv(total_rows, ROWS_PER_CTA)

    _softmax_bwd_kernel[(grid_size,)](
        dP_dropped_flat,
        attn_weights_flat,
        dropout_mask_flat,
        dS_flat,
        total_rows,
        seq_kv,
        inv_keep_prob,
        BLOCK_KV=BLOCK_KV,
        SINGLE_PASS=SINGLE_PASS,
        ROWS_PER_CTA=ROWS_PER_CTA,
        num_warps=4,
    )

    # ── Step 4: Fused BMM2 + GQA reduction — single large GEMM ──────────────
    #
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs*8, 10*sq, skv]
    # dO_2d: [bs*8, 10*sq, d]  (already computed above)
    #
    # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]

    P_dropped_2d = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    if not P_dropped_2d.is_contiguous():
        P_dropped_2d = P_dropped_2d.contiguous()
    # Transpose: [bs*8, skv, 10*sq]
    P_dropped_2d_T = P_dropped_2d.transpose(-2, -1)  # non-contiguous view, bmm handles it

    # Single BMM: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
    dV_flat = torch.bmm(P_dropped_2d_T, dO_2d)

    # Reshape to final output shape [bs, 8, skv, d]
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #15 — 2026-06-29 23:02:31 UTC ✅ KEEP

**Hypothesis:** ** Added `torch.cuda.Stream` pipelining:

**Result:** 424.41 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- Both BMMs as clean 3D batched GEMMs (cuBLAS-optimized, no broadcasting)
- BMM1 restructured: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
  then reshape to [bs, 80, sq, skv] — same K-merging trick as BMM2
- BMM2 fused with GQA reduction: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
- dO_2d [bs*8, 10*sq, d] reused across both BMMs (computed once)
- Dual-stream pipelining: BMM1 on stream A, BMM2 on stream B (launched concurrently)
  Triton softmax-bwd runs on stream A after BMM1 (overlaps with BMM2 on stream B)
  Final sync waits for both streams to complete
- Row-batched Triton kernel for elementwise dropout-bwd + softmax-bwd:
  * ROWS_PER_CTA rows processed per program to increase SM occupancy
  * Each row is handled independently within the CTA, amortizing launch overhead

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


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: row-batched fused dropout-bwd + softmax-bwd
#
# For each row (bs * n_heads * sq), length seq_kv:
#   dP = dP_dropped * mask / (1 - p)
#   dS = P * (dP - sum(P * dP))
#
# Row-batching strategy (ROWS_PER_CTA > 1):
#   - Each program handles ROWS_PER_CTA consecutive rows
#   - Grid size = ceil(total_rows / ROWS_PER_CTA)
#   - Increases SM occupancy by reducing kernel launch overhead and
#     giving each CTA more work (better warp utilization)
#
# For single-pass (seq_kv <= BLOCK_KV):
#   - Load dP_dropped, mask, P once per row into registers
#   - Compute dot via tl.sum, write dS — no re-read
#
# Two-pass fallback (seq_kv > BLOCK_KV):
#   - Pass 1: compute partial dot sums, Pass 2: write dS
#
# Grid: (ceil(total_rows / ROWS_PER_CTA),)
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _softmax_bwd_kernel(
    dP_dropped_ptr,    # [total_rows, seq_kv]  bfloat16
    attn_weights_ptr,  # [total_rows, seq_kv]  bfloat16
    dropout_mask_ptr,  # [total_rows, seq_kv]  bool (uint8)
    dS_ptr,            # [total_rows, seq_kv]  bfloat16  (output)
    total_rows,        # runtime int
    seq_kv,            # runtime int
    inv_keep_prob,     # runtime float32
    BLOCK_KV: tl.constexpr,
    SINGLE_PASS: tl.constexpr,  # True when seq_kv <= BLOCK_KV
    ROWS_PER_CTA: tl.constexpr,  # number of rows per CTA
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_CTA

    # Process ROWS_PER_CTA rows per CTA
    for i in tl.static_range(ROWS_PER_CTA):
        row_idx = row_start + i
        # Guard: skip rows beyond total_rows
        if row_idx < total_rows:
            base = row_idx * seq_kv

            if SINGLE_PASS:
                # ── Single pass: load once, compute dot, write dS ────────────
                offs = tl.arange(0, BLOCK_KV)
                valid = offs < seq_kv

                dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                dot = tl.sum(P_vals * dP_vals, axis=0)
                dS_vals = P_vals * (dP_vals - dot)
                tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)
            else:
                # ── Two-pass fallback for very large seq_kv ──────────────────
                dot = tl.zeros([1], dtype=tl.float32)
                for blk_start in tl.range(0, seq_kv, BLOCK_KV):
                    offs = blk_start + tl.arange(0, BLOCK_KV)
                    valid = offs < seq_kv

                    dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                    dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                    P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                    dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                    dot += tl.sum(P_vals * dP_vals, axis=0)

                for blk_start in tl.range(0, seq_kv, BLOCK_KV):
                    offs = blk_start + tl.arange(0, BLOCK_KV)
                    valid = offs < seq_kv

                    dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                    dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                    P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                    dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                    dS_vals = P_vals * (dP_vals - dot)
                    tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # ── Step 1: Transpose grad and build dO_2d for both BMMs ─────────────────
    # [bs, sq, 80, d] -> [bs, 80, sq, d] (contiguous bfloat16)
    dO = grad_attn_output.transpose(1, 2).contiguous()
    # [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    # This single reshape is reused for BOTH BMM1 and BMM2
    dO_2d = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)
    if not dO_2d.is_contiguous():
        dO_2d = dO_2d.contiguous()

    # ── Prepare inputs for both BMMs (before forking streams) ────────────────

    # BMM1 input: vs_T_2d [bs*8, d, skv]
    vs_T_2d = value_states.transpose(-2, -1).reshape(bs * n_kv_heads, HEAD_DIM, seq_kv)
    if not vs_T_2d.is_contiguous():
        vs_T_2d = vs_T_2d.contiguous()

    # BMM2 inputs: P_dropped_2d_T [bs*8, skv, 10*sq]
    P_dropped_2d = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    if not P_dropped_2d.is_contiguous():
        P_dropped_2d = P_dropped_2d.contiguous()
    P_dropped_2d_T = P_dropped_2d.transpose(-2, -1)  # non-contiguous view, bmm handles it

    # Allocate output tensors
    dP_dropped_2d = torch.empty((bs * n_kv_heads, n_groups * seq_q, seq_kv),
                                 dtype=torch.bfloat16, device=dO.device)
    dV_flat = torch.empty((bs * n_kv_heads, seq_kv, HEAD_DIM),
                           dtype=torch.bfloat16, device=dO.device)

    # ── Step 2: Launch BMM1 on stream A, BMM2 on stream B concurrently ───────
    #
    # BMM1 (stream A): [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
    # BMM2 (stream B): [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
    # These two BMMs are INDEPENDENT — no data dependency between them.
    # Launch them concurrently and overlap execution.

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    # Record a "start" event on the default stream so both streams wait for
    # dO_2d, vs_T_2d, and P_dropped_2d_T to be ready.
    default_stream = torch.cuda.current_stream()
    start_event = torch.cuda.Event()
    start_event.record(default_stream)

    # Stream A: BMM1
    with torch.cuda.stream(stream_a):
        stream_a.wait_event(start_event)
        torch.bmm(dO_2d, vs_T_2d, out=dP_dropped_2d)

    # Stream B: BMM2
    with torch.cuda.stream(stream_b):
        stream_b.wait_event(start_event)
        torch.bmm(P_dropped_2d_T, dO_2d, out=dV_flat)

    # ── Step 3: After BMM1 completes, run Triton softmax-bwd on stream A ─────
    # BMM2 continues on stream B concurrently with the Triton kernel on stream A.

    # Reshape BMM1 output for Triton kernel
    dP_dropped = dP_dropped_2d.reshape(bs, n_heads, seq_q, seq_kv)

    total_rows = bs * n_heads * seq_q
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    # Flatten to [total_rows, seq_kv]
    dP_dropped_flat = dP_dropped.reshape(total_rows, seq_kv)
    if not dP_dropped_flat.is_contiguous():
        dP_dropped_flat = dP_dropped_flat.contiguous()
    attn_weights_flat = attn_weights.reshape(total_rows, seq_kv)
    if not attn_weights_flat.is_contiguous():
        attn_weights_flat = attn_weights_flat.contiguous()
    dropout_mask_flat = dropout_mask.reshape(total_rows, seq_kv)
    if not dropout_mask_flat.is_contiguous():
        dropout_mask_flat = dropout_mask_flat.contiguous()
    dS_flat = dS.reshape(total_rows, seq_kv)

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 16384)
    SINGLE_PASS = (seq_kv <= BLOCK_KV)

    ROWS_PER_CTA = 4
    grid_size = triton.cdiv(total_rows, ROWS_PER_CTA)

    # Launch Triton kernel on stream A (depends on BMM1, runs concurrently with BMM2)
    with torch.cuda.stream(stream_a):
        _softmax_bwd_kernel[(grid_size,)](
            dP_dropped_flat,
            attn_weights_flat,
            dropout_mask_flat,
            dS_flat,
            total_rows,
            seq_kv,
            inv_keep_prob,
            BLOCK_KV=BLOCK_KV,
            SINGLE_PASS=SINGLE_PASS,
            ROWS_PER_CTA=ROWS_PER_CTA,
            num_warps=4,
        )

    # ── Step 4: Sync both streams back to the default stream ─────────────────
    # Record completion events for both streams
    event_a = torch.cuda.Event()
    event_b = torch.cuda.Event()
    event_a.record(stream_a)
    event_b.record(stream_b)

    # Default stream waits for both streams to finish
    default_stream.wait_event(event_a)
    default_stream.wait_event(event_b)

    # ── Step 5: Reshape outputs ───────────────────────────────────────────────
    # dV: [bs*8, skv, d] -> [bs, 8, skv, d]
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #16 — 2026-06-29 23:04:39 UTC ❌ DISCARD

**Hypothesis:** ** Added `_fused_bmm1_softmax_bwd` Triton kernel that:

**Result:** 5264.45 μs

---

## Experiment #17 — 2026-06-29 23:06:27 UTC ❌ DISCARD

**Hypothesis:** ** Added a `_cuda_graph_cache` dictionary keyed on `(bs, seq_q, seq_kv)`. On first call with a given shape: allocates static buffers, runs 3 warm-up passes (compiling Triton kernel and cuBLAS), captur

**Result:** 1058.75 μs

---

## Experiment #18 — 2026-06-29 23:08:27 UTC ❌ DISCARD

**Hypothesis:** Replaced the `submission.py` (which had the broken CUDA-graph approach from experiment #17) with the best known approach (experiment #15 dual-stream pipeline) but with `.contiguous()` copies removed f

**Result:** 491.03 μs

---

## Experiment #19 — 2026-06-29 23:10:28 UTC ✅ KEEP

**Hypothesis:** ** Added `_transpose_sq_heads_kernel` — a Triton kernel that:

**Result:** 398.66 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- Both BMMs as clean 3D batched GEMMs (cuBLAS-optimized, no broadcasting)
- BMM1 restructured: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
  then reshape to [bs, 80, sq, skv] — same K-merging trick as BMM2
- BMM2 fused with GQA reduction: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
- dO_2d [bs*8, 10*sq, d] reused across both BMMs (computed once)
- Dual-stream pipelining: BMM1 on stream A, BMM2 on stream B (launched concurrently)
  Triton softmax-bwd runs on stream A after BMM1 (overlaps with BMM2 on stream B)
  Final sync waits for both streams to complete
- Row-batched Triton kernel for elementwise dropout-bwd + softmax-bwd
- Custom Triton transpose kernel: reads grad_attn_output [bs, sq, 80, d] natively
  and writes transposed [bs, 80, sq, d] using tiled 2D access pattern for
  coalesced reads and writes (avoids non-coalesced generic PyTorch transpose)

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


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: tiled transpose [bs, sq, n_heads, d] -> [bs, n_heads, sq, d]
#
# The transpose swaps dims 1 and 2: (sq <-> n_heads).
# The head dimension (d=128) is kept as a contiguous inner dimension and
# is NOT transposed — we move entire head vectors of size d.
#
# Input layout:  [bs, sq, n_heads, d]   strides: (sq*n_heads*d, n_heads*d, d, 1)
# Output layout: [bs, n_heads, sq, d]   strides: (n_heads*sq*d, sq*d, d, 1)
#
# Grid: (bs * cdiv(sq, TILE_SQ) * cdiv(n_heads, TILE_H),)
# Each program handles a TILE_SQ x TILE_H tile of (sq, n_heads) for one batch.
# Within the tile, all HEAD_DIM elements are processed in one shot.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _transpose_sq_heads_kernel(
    src_ptr,              # [bs, sq, n_heads, d]  bfloat16
    dst_ptr,              # [bs, n_heads, sq, d]  bfloat16
    bs,                   # batch size
    sq,                   # seq_q
    n_heads,              # 80
    HEAD_DIM: tl.constexpr,  # 128
    TILE_SQ: tl.constexpr,   # tile size over sq dimension
    TILE_H: tl.constexpr,    # tile size over n_heads dimension
):
    # Compute grid dimensions
    num_tiles_sq = tl.cdiv(sq, TILE_SQ)
    num_tiles_h  = tl.cdiv(n_heads, TILE_H)

    pid = tl.program_id(0)
    # Decompose pid into (batch_idx, tile_h_idx, tile_sq_idx)
    tile_per_batch = num_tiles_sq * num_tiles_h
    batch_idx  = pid // tile_per_batch
    tile_idx   = pid % tile_per_batch
    tile_sq_idx = tile_idx % num_tiles_sq
    tile_h_idx  = tile_idx // num_tiles_sq

    sq_start = tile_sq_idx * TILE_SQ
    h_start  = tile_h_idx  * TILE_H

    offs_sq = sq_start + tl.arange(0, TILE_SQ)   # [TILE_SQ]
    offs_h  = h_start  + tl.arange(0, TILE_H)    # [TILE_H]
    offs_d  = tl.arange(0, HEAD_DIM)              # [HEAD_DIM]

    valid_sq = offs_sq < sq      # [TILE_SQ]
    valid_h  = offs_h  < n_heads # [TILE_H]

    # Src strides: [bs, sq, n_heads, d]
    #   flat index = batch_idx * sq * n_heads * d
    #              + offs_sq[:, None] * n_heads * d
    #              + offs_h[None, :] * d
    #              + offs_d (broadcast over both sq and h)
    # Load: shape [TILE_SQ, TILE_H, HEAD_DIM]
    src_base = batch_idx * sq * n_heads * HEAD_DIM
    src_offsets = (src_base
                   + offs_sq[:, None, None] * (n_heads * HEAD_DIM)
                   + offs_h[None, :, None] * HEAD_DIM
                   + offs_d[None, None, :])
    valid_mask = (valid_sq[:, None, None] & valid_h[None, :, None])

    vals = tl.load(src_ptr + src_offsets, mask=valid_mask, other=0.0)

    # Dst strides: [bs, n_heads, sq, d]
    #   flat index = batch_idx * n_heads * sq * d
    #              + offs_h[None, :] * sq * d
    #              + offs_sq[:, None] * d
    #              + offs_d
    dst_base = batch_idx * n_heads * sq * HEAD_DIM
    dst_offsets = (dst_base
                   + offs_h[None, :, None] * (sq * HEAD_DIM)
                   + offs_sq[:, None, None] * HEAD_DIM
                   + offs_d[None, None, :])

    tl.store(dst_ptr + dst_offsets, vals, mask=valid_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: row-batched fused dropout-bwd + softmax-bwd
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _softmax_bwd_kernel(
    dP_dropped_ptr,    # [total_rows, seq_kv]  bfloat16
    attn_weights_ptr,  # [total_rows, seq_kv]  bfloat16
    dropout_mask_ptr,  # [total_rows, seq_kv]  bool (uint8)
    dS_ptr,            # [total_rows, seq_kv]  bfloat16  (output)
    total_rows,        # runtime int
    seq_kv,            # runtime int
    inv_keep_prob,     # runtime float32
    BLOCK_KV: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_CTA

    for i in tl.static_range(ROWS_PER_CTA):
        row_idx = row_start + i
        if row_idx < total_rows:
            base = row_idx * seq_kv

            if SINGLE_PASS:
                offs = tl.arange(0, BLOCK_KV)
                valid = offs < seq_kv

                dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                dot = tl.sum(P_vals * dP_vals, axis=0)
                dS_vals = P_vals * (dP_vals - dot)
                tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)
            else:
                dot = tl.zeros([1], dtype=tl.float32)
                for blk_start in tl.range(0, seq_kv, BLOCK_KV):
                    offs = blk_start + tl.arange(0, BLOCK_KV)
                    valid = offs < seq_kv

                    dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                    dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                    P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                    dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                    dot += tl.sum(P_vals * dP_vals, axis=0)

                for blk_start in tl.range(0, seq_kv, BLOCK_KV):
                    offs = blk_start + tl.arange(0, BLOCK_KV)
                    valid = offs < seq_kv

                    dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                    dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                    P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                    dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                    dS_vals = P_vals * (dP_vals - dot)
                    tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    device = grad_attn_output.device

    # ── Step 1: Custom tiled transpose [bs, sq, 80, d] -> [bs, 80, sq, d] ────
    # Use a Triton kernel instead of PyTorch's generic transpose+contiguous.
    # The kernel uses TILE_SQ x TILE_H tiles over (sq, n_heads) for each batch,
    # processing HEAD_DIM elements in one shot per tile element.
    # This avoids the inefficient strided memory access pattern of the generic copy.

    dO = torch.empty((bs, n_heads, seq_q, HEAD_DIM), dtype=torch.bfloat16, device=device)

    # Tile sizes: TILE_SQ=8, TILE_H=8 -> each CTA handles 8*8=64 head-vectors
    TILE_SQ = 8
    TILE_H  = 8
    num_tiles_sq = triton.cdiv(seq_q, TILE_SQ)
    num_tiles_h  = triton.cdiv(n_heads, TILE_H)
    transpose_grid = bs * num_tiles_sq * num_tiles_h

    _transpose_sq_heads_kernel[(transpose_grid,)](
        grad_attn_output,
        dO,
        bs, seq_q, n_heads,
        HEAD_DIM=HEAD_DIM,
        TILE_SQ=TILE_SQ,
        TILE_H=TILE_H,
    )

    # dO is now [bs, 80, sq, d], contiguous
    # Reshape to [bs*8, 10*sq, d] for both BMMs
    dO_2d = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # ── Step 2: Prepare BMM inputs ────────────────────────────────────────────

    # BMM1 input: vs_T_2d [bs*8, d, skv]
    vs_T_2d = value_states.transpose(-2, -1).reshape(bs * n_kv_heads, HEAD_DIM, seq_kv)
    if not vs_T_2d.is_contiguous():
        vs_T_2d = vs_T_2d.contiguous()

    # BMM2 inputs: P_dropped_2d_T [bs*8, skv, 10*sq]
    P_dropped_2d = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    P_dropped_2d_T = P_dropped_2d.transpose(-2, -1)

    # Flatten attn_weights and dropout_mask for Triton kernel
    total_rows = bs * n_heads * seq_q
    attn_weights_flat = attn_weights.reshape(total_rows, seq_kv)
    dropout_mask_flat = dropout_mask.reshape(total_rows, seq_kv)

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 16384)
    SINGLE_PASS = (seq_kv <= BLOCK_KV)
    ROWS_PER_CTA = 4
    grid_size = triton.cdiv(total_rows, ROWS_PER_CTA)

    # Allocate output tensors
    dP_dropped_2d = torch.empty((bs * n_kv_heads, n_groups * seq_q, seq_kv),
                                 dtype=torch.bfloat16, device=device)
    dV_flat = torch.empty((bs * n_kv_heads, seq_kv, HEAD_DIM),
                           dtype=torch.bfloat16, device=device)
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=device)

    # ── Step 3: Launch BMM1 on stream A, BMM2 on stream B concurrently ───────
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    default_stream = torch.cuda.current_stream()
    start_event = torch.cuda.Event()
    start_event.record(default_stream)

    # Stream A: BMM1
    with torch.cuda.stream(stream_a):
        stream_a.wait_event(start_event)
        torch.bmm(dO_2d, vs_T_2d, out=dP_dropped_2d)

    # Stream B: BMM2
    with torch.cuda.stream(stream_b):
        stream_b.wait_event(start_event)
        torch.bmm(P_dropped_2d_T, dO_2d, out=dV_flat)

    # ── Step 4: After BMM1 completes, run Triton softmax-bwd on stream A ─────
    dS_flat = dS.reshape(total_rows, seq_kv)
    dP_dropped_flat = dP_dropped_2d.reshape(total_rows, seq_kv)

    with torch.cuda.stream(stream_a):
        _softmax_bwd_kernel[(grid_size,)](
            dP_dropped_flat,
            attn_weights_flat,
            dropout_mask_flat,
            dS_flat,
            total_rows,
            seq_kv,
            inv_keep_prob,
            BLOCK_KV=BLOCK_KV,
            SINGLE_PASS=SINGLE_PASS,
            ROWS_PER_CTA=ROWS_PER_CTA,
            num_warps=4,
        )

    # ── Step 5: Sync both streams back to the default stream ─────────────────
    event_a = torch.cuda.Event()
    event_b = torch.cuda.Event()
    event_a.record(stream_a)
    event_b.record(stream_b)

    default_stream.wait_event(event_a)
    default_stream.wait_event(event_b)

    # ── Step 6: Reshape outputs ───────────────────────────────────────────────
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #20 — 2026-06-29 23:12:59 UTC ❌ DISCARD

**Hypothesis:** ** Replaced the `_softmax_bwd_kernel` + separate BMM1 path with a new `_fused_bmm1_softmax_bwd_kernel` that uses a 2D grid `(bs*n_kv_heads, n_groups*seq_q)`. Each program: (1) loads `dO_row[128]` into

**Result:** 5143.75 μs

---

## Experiment #21 — 2026-06-29 23:16:29 UTC ✅ KEEP

**Hypothesis:** 1. Added `_stream_a`, `_stream_b`, `_start_event`, `_event_a`, `_event_b` as module-level globals, initialized lazily via `_ensure_streams()` (called once per process).

**Result:** 388.64 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- Custom Triton transpose kernel for dO: [bs, sq, 80, d] -> [bs, 80, sq, d]
- Custom Triton transpose kernel for vs_T: [bs, 8, skv, d] -> [bs*8, d, skv]
  (replaces .transpose().reshape().contiguous() which materializes a strided copy)
- Both BMMs as clean 3D batched GEMMs (cuBLAS-optimized, no broadcasting)
- BMM1: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
- BMM2 fused with GQA: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
- Dual-stream pipelining with module-level cached streams/events
  (eliminates Python object allocation overhead on every call)
- Row-batched Triton softmax-bwd kernel on stream A after BMM1
  (overlaps with BMM2 on stream B)
"""

import torch
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128

# ─────────────────────────────────────────────────────────────────────────────
# Module-level cached CUDA streams and events
# Created once at import/first-call time, reused on every subsequent call
# ─────────────────────────────────────────────────────────────────────────────
_stream_a = None
_stream_b = None
_start_event = None
_event_a = None
_event_b = None


def _ensure_streams():
    global _stream_a, _stream_b, _start_event, _event_a, _event_b
    if _stream_a is None:
        _stream_a = torch.cuda.Stream()
        _stream_b = torch.cuda.Stream()
        _start_event = torch.cuda.Event()
        _event_a = torch.cuda.Event()
        _event_b = torch.cuda.Event()


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: tiled transpose [bs, sq, n_heads, d] -> [bs, n_heads, sq, d]
#
# Swaps dims 1 (sq) and 2 (n_heads), keeps d as inner dim.
# Each CTA handles a TILE_SQ x TILE_H tile for one batch element.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _transpose_sq_heads_kernel(
    src_ptr,
    dst_ptr,
    bs,
    sq,
    n_heads,
    HEAD_DIM: tl.constexpr,
    TILE_SQ: tl.constexpr,
    TILE_H: tl.constexpr,
):
    num_tiles_sq = tl.cdiv(sq, TILE_SQ)
    num_tiles_h  = tl.cdiv(n_heads, TILE_H)

    pid = tl.program_id(0)
    tile_per_batch = num_tiles_sq * num_tiles_h
    batch_idx   = pid // tile_per_batch
    tile_idx    = pid % tile_per_batch
    tile_sq_idx = tile_idx % num_tiles_sq
    tile_h_idx  = tile_idx // num_tiles_sq

    sq_start = tile_sq_idx * TILE_SQ
    h_start  = tile_h_idx  * TILE_H

    offs_sq = sq_start + tl.arange(0, TILE_SQ)
    offs_h  = h_start  + tl.arange(0, TILE_H)
    offs_d  = tl.arange(0, HEAD_DIM)

    valid_sq = offs_sq < sq
    valid_h  = offs_h  < n_heads

    src_base = batch_idx * sq * n_heads * HEAD_DIM
    src_offsets = (src_base
                   + offs_sq[:, None, None] * (n_heads * HEAD_DIM)
                   + offs_h[None, :, None] * HEAD_DIM
                   + offs_d[None, None, :])
    valid_mask = (valid_sq[:, None, None] & valid_h[None, :, None])

    vals = tl.load(src_ptr + src_offsets, mask=valid_mask, other=0.0)

    dst_base = batch_idx * n_heads * sq * HEAD_DIM
    dst_offsets = (dst_base
                   + offs_h[None, :, None] * (sq * HEAD_DIM)
                   + offs_sq[:, None, None] * HEAD_DIM
                   + offs_d[None, None, :])

    tl.store(dst_ptr + dst_offsets, vals, mask=valid_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: transpose-reshape value_states
#   Input:  [bs * n_kv_heads, skv, d]    (contiguous; view of [bs,8,skv,d])
#   Output: [bs * n_kv_heads, d, skv]    (contiguous)
#
# This replaces: value_states.transpose(-2,-1).reshape(bs*8,d,skv).contiguous()
# Each CTA handles a TILE_SKV x TILE_D tile of (skv, d) for one (b, kv_head).
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _transpose_vs_kernel(
    src_ptr,              # [n_bkv, skv, d]  bfloat16  (contiguous)
    dst_ptr,              # [n_bkv, d, skv]  bfloat16  (output)
    n_bkv,                # bs * n_kv_heads
    skv,                  # seq_kv
    HEAD_DIM: tl.constexpr,
    TILE_SKV: tl.constexpr,
    TILE_D: tl.constexpr,
):
    num_tiles_skv = tl.cdiv(skv, TILE_SKV)
    num_tiles_d   = tl.cdiv(HEAD_DIM, TILE_D)

    pid = tl.program_id(0)
    tiles_per_bkv = num_tiles_skv * num_tiles_d
    bkv_idx  = pid // tiles_per_bkv
    tile_idx = pid % tiles_per_bkv
    tile_skv = tile_idx % num_tiles_skv
    tile_d   = tile_idx // num_tiles_skv

    skv_start = tile_skv * TILE_SKV
    d_start   = tile_d   * TILE_D

    offs_skv = skv_start + tl.arange(0, TILE_SKV)  # [TILE_SKV]
    offs_d   = d_start   + tl.arange(0, TILE_D)    # [TILE_D]

    valid_skv = offs_skv < skv
    valid_d   = offs_d   < HEAD_DIM
    valid_mask = valid_skv[:, None] & valid_d[None, :]

    # Source layout: [n_bkv, skv, d]
    # Element [bkv, s, d_] is at bkv * skv * HEAD_DIM + s * HEAD_DIM + d_
    src_base = bkv_idx * skv * HEAD_DIM
    src_offsets = (src_base
                   + offs_skv[:, None] * HEAD_DIM   # [TILE_SKV, 1]
                   + offs_d[None, :])                # [1, TILE_D]
    # vals shape: [TILE_SKV, TILE_D]
    vals = tl.load(src_ptr + src_offsets, mask=valid_mask, other=0.0)

    # Destination layout: [n_bkv, d, skv]
    # Element [bkv, d_, s] is at bkv * HEAD_DIM * skv + d_ * skv + s
    # We store vals[skv_local, d_local] at dst[bkv, d_start+d_local, skv_start+skv_local]
    # = dst_base + (d_start + d_local) * skv + (skv_start + skv_local)
    dst_base = bkv_idx * HEAD_DIM * skv
    dst_offsets = (dst_base
                   + offs_d[None, :] * skv            # [1, TILE_D] broadcast
                   + offs_skv[:, None])                # [TILE_SKV, 1] broadcast
    # dst_offsets shape: [TILE_SKV, TILE_D] — same shape as vals ✓
    tl.store(dst_ptr + dst_offsets, vals, mask=valid_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: row-batched fused dropout-bwd + softmax-bwd
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _softmax_bwd_kernel(
    dP_dropped_ptr,    # [total_rows, seq_kv]  bfloat16
    attn_weights_ptr,  # [total_rows, seq_kv]  bfloat16
    dropout_mask_ptr,  # [total_rows, seq_kv]  bool (uint8)
    dS_ptr,            # [total_rows, seq_kv]  bfloat16  (output)
    total_rows,        # runtime int
    seq_kv,            # runtime int
    inv_keep_prob,     # runtime float32
    BLOCK_KV: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_CTA

    for i in tl.static_range(ROWS_PER_CTA):
        row_idx = row_start + i
        if row_idx < total_rows:
            base = row_idx * seq_kv

            if SINGLE_PASS:
                offs = tl.arange(0, BLOCK_KV)
                valid = offs < seq_kv

                dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                dot = tl.sum(P_vals * dP_vals, axis=0)
                dS_vals = P_vals * (dP_vals - dot)
                tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)
            else:
                dot = tl.zeros([1], dtype=tl.float32)
                for blk_start in tl.range(0, seq_kv, BLOCK_KV):
                    offs = blk_start + tl.arange(0, BLOCK_KV)
                    valid = offs < seq_kv

                    dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                    dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                    P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                    dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                    dot += tl.sum(P_vals * dP_vals, axis=0)

                for blk_start in tl.range(0, seq_kv, BLOCK_KV):
                    offs = blk_start + tl.arange(0, BLOCK_KV)
                    valid = offs < seq_kv

                    dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                    dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                    P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                    dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                    dS_vals = P_vals * (dP_vals - dot)
                    tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    device = grad_attn_output.device

    # Ensure module-level streams and events are initialized (cached)
    _ensure_streams()

    # ── Step 1: Custom tiled transpose [bs, sq, 80, d] -> [bs, 80, sq, d] ────
    dO = torch.empty((bs, n_heads, seq_q, HEAD_DIM), dtype=torch.bfloat16, device=device)

    TILE_SQ = 8
    TILE_H  = 8
    num_tiles_sq = triton.cdiv(seq_q, TILE_SQ)
    num_tiles_h  = triton.cdiv(n_heads, TILE_H)
    transpose_grid = bs * num_tiles_sq * num_tiles_h

    _transpose_sq_heads_kernel[(transpose_grid,)](
        grad_attn_output,
        dO,
        bs, seq_q, n_heads,
        HEAD_DIM=HEAD_DIM,
        TILE_SQ=TILE_SQ,
        TILE_H=TILE_H,
    )

    # dO is now [bs, 80, sq, d], contiguous
    # Reshape to [bs*8, 10*sq, d] for both BMMs
    dO_2d = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # ── Step 2: Transpose value_states via Triton kernel ─────────────────────
    # Replaces: value_states.transpose(-2,-1).reshape(bs*8, d, skv).contiguous()
    # Input:  [bs, 8, skv, d] viewed as [bs*8, skv, d]
    # Output: [bs*8, d, skv]
    n_bkv = bs * n_kv_heads
    vs_T_2d = torch.empty((n_bkv, HEAD_DIM, seq_kv), dtype=torch.bfloat16, device=device)

    TILE_SKV = 16
    TILE_D   = 16
    num_tiles_skv = triton.cdiv(seq_kv, TILE_SKV)
    num_tiles_d   = triton.cdiv(HEAD_DIM, TILE_D)
    vs_transpose_grid = n_bkv * num_tiles_skv * num_tiles_d

    # value_states is [bs, 8, skv, d] — reshape to [bs*8, skv, d] for the kernel
    vs_2d = value_states.reshape(n_bkv, seq_kv, HEAD_DIM)
    # vs_2d may be non-contiguous if value_states was non-contiguous; check
    if not vs_2d.is_contiguous():
        vs_2d = vs_2d.contiguous()

    _transpose_vs_kernel[(vs_transpose_grid,)](
        vs_2d,
        vs_T_2d,
        n_bkv, seq_kv,
        HEAD_DIM=HEAD_DIM,
        TILE_SKV=TILE_SKV,
        TILE_D=TILE_D,
    )

    # ── Step 3: Prepare BMM2 input ────────────────────────────────────────────
    # P_dropped_2d_T: [bs*8, skv, 10*sq] — non-contiguous transpose, bmm handles it
    P_dropped_2d = attn_weights_dropped.reshape(n_bkv, n_groups * seq_q, seq_kv)
    P_dropped_2d_T = P_dropped_2d.transpose(-2, -1)

    # Flatten for Triton softmax-bwd kernel
    total_rows = bs * n_heads * seq_q
    attn_weights_flat = attn_weights.reshape(total_rows, seq_kv)
    dropout_mask_flat = dropout_mask.reshape(total_rows, seq_kv)

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0
    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 16384)
    SINGLE_PASS = (seq_kv <= BLOCK_KV)
    ROWS_PER_CTA = 4
    grid_size = triton.cdiv(total_rows, ROWS_PER_CTA)

    # ── Step 4: Allocate output tensors ───────────────────────────────────────
    dP_dropped_2d = torch.empty((n_bkv, n_groups * seq_q, seq_kv),
                                 dtype=torch.bfloat16, device=device)
    dV_flat = torch.empty((n_bkv, seq_kv, HEAD_DIM), dtype=torch.bfloat16, device=device)
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=device)

    # ── Step 5: Launch both BMMs concurrently on separate streams ─────────────
    default_stream = torch.cuda.current_stream()
    _start_event.record(default_stream)

    # Stream A: BMM1 → softmax-bwd
    with torch.cuda.stream(_stream_a):
        _stream_a.wait_event(_start_event)
        # BMM1: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
        torch.bmm(dO_2d, vs_T_2d, out=dP_dropped_2d)

    # Stream B: BMM2
    with torch.cuda.stream(_stream_b):
        _stream_b.wait_event(_start_event)
        # BMM2: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
        torch.bmm(P_dropped_2d_T, dO_2d, out=dV_flat)

    # ── Step 6: After BMM1 completes, run softmax-bwd on stream A ─────────────
    dS_flat = dS.reshape(total_rows, seq_kv)
    dP_dropped_flat = dP_dropped_2d.reshape(total_rows, seq_kv)

    with torch.cuda.stream(_stream_a):
        _softmax_bwd_kernel[(grid_size,)](
            dP_dropped_flat,
            attn_weights_flat,
            dropout_mask_flat,
            dS_flat,
            total_rows,
            seq_kv,
            inv_keep_prob,
            BLOCK_KV=BLOCK_KV,
            SINGLE_PASS=SINGLE_PASS,
            ROWS_PER_CTA=ROWS_PER_CTA,
            num_warps=4,
        )

    # ── Step 7: Sync both streams back to the default stream ──────────────────
    _event_a.record(_stream_a)
    _event_b.record(_stream_b)
    default_stream.wait_event(_event_a)
    default_stream.wait_event(_event_b)

    # ── Step 8: Reshape outputs ────────────────────────────────────────────────
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #22 — 2026-06-29 23:18:52 UTC ❌ DISCARD

**Hypothesis:** Worker implementation

**Result:** 388.71 μs

---

## Experiment #23 — 2026-06-29 23:21:07 UTC ✅ KEEP

**Hypothesis:** **

**Result:** 379.40 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- Custom Triton transpose kernel for dO: [bs, sq, 80, d] -> [bs, 80, sq, d]
- Custom Triton transpose kernel for vs_T: [bs, 8, skv, d] -> [bs*8, d, skv]
  (replaces .transpose().reshape().contiguous() which materializes a strided copy)
- Both BMMs as clean 3D batched GEMMs (cuBLAS-optimized, no broadcasting)
- BMM1: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
- BMM2 fused with GQA: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
- Dual-stream pipelining with module-level cached streams/events
  (eliminates Python object allocation overhead on every call)
- Row-batched Triton softmax-bwd kernel on stream A after BMM1
  (overlaps with BMM2 on stream B)
- Persistent buffer cache: pre-allocated tensors keyed by (bs, seq_q, seq_kv)
  to eliminate torch.empty() allocation overhead on the hot path
- Shape-adaptive dispatch: skip multi-stream overhead for small workloads
"""

import torch
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128

# ─────────────────────────────────────────────────────────────────────────────
# Module-level cached CUDA streams and events
# Created once at import/first-call time, reused on every subsequent call
# ─────────────────────────────────────────────────────────────────────────────
_stream_a = None
_stream_b = None
_start_event = None
_event_a = None
_event_b = None


def _ensure_streams():
    global _stream_a, _stream_b, _start_event, _event_a, _event_b
    if _stream_a is None:
        _stream_a = torch.cuda.Stream()
        _stream_b = torch.cuda.Stream()
        _start_event = torch.cuda.Event()
        _event_a = torch.cuda.Event()
        _event_b = torch.cuda.Event()


# ─────────────────────────────────────────────────────────────────────────────
# Persistent buffer cache: keyed by (bs, seq_q, seq_kv)
# Stores pre-allocated tensors: (dO, vs_T_2d, dP_dropped_2d, dV_flat, dS)
# ─────────────────────────────────────────────────────────────────────────────
_buffer_cache = {}


def _get_buffers(bs, seq_q, seq_kv, device):
    key = (bs, seq_q, seq_kv)
    if key not in _buffer_cache:
        n_kv_heads = NUM_KEY_VALUE_HEADS
        n_heads    = NUM_ATTENTION_HEADS
        n_groups   = n_heads // n_kv_heads
        n_bkv      = bs * n_kv_heads

        dO         = torch.empty((bs, n_heads, seq_q, HEAD_DIM),
                                  dtype=torch.bfloat16, device=device)
        vs_T_2d    = torch.empty((n_bkv, HEAD_DIM, seq_kv),
                                  dtype=torch.bfloat16, device=device)
        dP_dropped_2d = torch.empty((n_bkv, n_groups * seq_q, seq_kv),
                                     dtype=torch.bfloat16, device=device)
        dV_flat    = torch.empty((n_bkv, seq_kv, HEAD_DIM),
                                  dtype=torch.bfloat16, device=device)
        dS         = torch.empty((bs, n_heads, seq_q, seq_kv),
                                  dtype=torch.bfloat16, device=device)
        _buffer_cache[key] = (dO, vs_T_2d, dP_dropped_2d, dV_flat, dS)
    return _buffer_cache[key]


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: tiled transpose [bs, sq, n_heads, d] -> [bs, n_heads, sq, d]
#
# Swaps dims 1 (sq) and 2 (n_heads), keeps d as inner dim.
# Each CTA handles a TILE_SQ x TILE_H tile for one batch element.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _transpose_sq_heads_kernel(
    src_ptr,
    dst_ptr,
    bs,
    sq,
    n_heads,
    HEAD_DIM: tl.constexpr,
    TILE_SQ: tl.constexpr,
    TILE_H: tl.constexpr,
):
    num_tiles_sq = tl.cdiv(sq, TILE_SQ)
    num_tiles_h  = tl.cdiv(n_heads, TILE_H)

    pid = tl.program_id(0)
    tile_per_batch = num_tiles_sq * num_tiles_h
    batch_idx   = pid // tile_per_batch
    tile_idx    = pid % tile_per_batch
    tile_sq_idx = tile_idx % num_tiles_sq
    tile_h_idx  = tile_idx // num_tiles_sq

    sq_start = tile_sq_idx * TILE_SQ
    h_start  = tile_h_idx  * TILE_H

    offs_sq = sq_start + tl.arange(0, TILE_SQ)
    offs_h  = h_start  + tl.arange(0, TILE_H)
    offs_d  = tl.arange(0, HEAD_DIM)

    valid_sq = offs_sq < sq
    valid_h  = offs_h  < n_heads

    src_base = batch_idx * sq * n_heads * HEAD_DIM
    src_offsets = (src_base
                   + offs_sq[:, None, None] * (n_heads * HEAD_DIM)
                   + offs_h[None, :, None] * HEAD_DIM
                   + offs_d[None, None, :])
    valid_mask = (valid_sq[:, None, None] & valid_h[None, :, None])

    vals = tl.load(src_ptr + src_offsets, mask=valid_mask, other=0.0)

    dst_base = batch_idx * n_heads * sq * HEAD_DIM
    dst_offsets = (dst_base
                   + offs_h[None, :, None] * (sq * HEAD_DIM)
                   + offs_sq[:, None, None] * HEAD_DIM
                   + offs_d[None, None, :])

    tl.store(dst_ptr + dst_offsets, vals, mask=valid_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: transpose-reshape value_states
#   Input:  [bs * n_kv_heads, skv, d]    (contiguous; view of [bs,8,skv,d])
#   Output: [bs * n_kv_heads, d, skv]    (contiguous)
#
# This replaces: value_states.transpose(-2,-1).reshape(bs*8,d,skv).contiguous()
# Each CTA handles a TILE_SKV x TILE_D tile of (skv, d) for one (b, kv_head).
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _transpose_vs_kernel(
    src_ptr,              # [n_bkv, skv, d]  bfloat16  (contiguous)
    dst_ptr,              # [n_bkv, d, skv]  bfloat16  (output)
    n_bkv,                # bs * n_kv_heads
    skv,                  # seq_kv
    HEAD_DIM: tl.constexpr,
    TILE_SKV: tl.constexpr,
    TILE_D: tl.constexpr,
):
    num_tiles_skv = tl.cdiv(skv, TILE_SKV)
    num_tiles_d   = tl.cdiv(HEAD_DIM, TILE_D)

    pid = tl.program_id(0)
    tiles_per_bkv = num_tiles_skv * num_tiles_d
    bkv_idx  = pid // tiles_per_bkv
    tile_idx = pid % tiles_per_bkv
    tile_skv = tile_idx % num_tiles_skv
    tile_d   = tile_idx // num_tiles_skv

    skv_start = tile_skv * TILE_SKV
    d_start   = tile_d   * TILE_D

    offs_skv = skv_start + tl.arange(0, TILE_SKV)  # [TILE_SKV]
    offs_d   = d_start   + tl.arange(0, TILE_D)    # [TILE_D]

    valid_skv = offs_skv < skv
    valid_d   = offs_d   < HEAD_DIM
    valid_mask = valid_skv[:, None] & valid_d[None, :]

    # Source layout: [n_bkv, skv, d]
    # Element [bkv, s, d_] is at bkv * skv * HEAD_DIM + s * HEAD_DIM + d_
    src_base = bkv_idx * skv * HEAD_DIM
    src_offsets = (src_base
                   + offs_skv[:, None] * HEAD_DIM   # [TILE_SKV, 1]
                   + offs_d[None, :])                # [1, TILE_D]
    # vals shape: [TILE_SKV, TILE_D]
    vals = tl.load(src_ptr + src_offsets, mask=valid_mask, other=0.0)

    # Destination layout: [n_bkv, d, skv]
    # Element [bkv, d_, s] is at bkv * HEAD_DIM * skv + d_ * skv + s
    # We store vals[skv_local, d_local] at dst[bkv, d_start+d_local, skv_start+skv_local]
    # = dst_base + (d_start + d_local) * skv + (skv_start + skv_local)
    dst_base = bkv_idx * HEAD_DIM * skv
    dst_offsets = (dst_base
                   + offs_d[None, :] * skv            # [1, TILE_D] broadcast
                   + offs_skv[:, None])                # [TILE_SKV, 1] broadcast
    # dst_offsets shape: [TILE_SKV, TILE_D] — same shape as vals ✓
    tl.store(dst_ptr + dst_offsets, vals, mask=valid_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: row-batched fused dropout-bwd + softmax-bwd
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _softmax_bwd_kernel(
    dP_dropped_ptr,    # [total_rows, seq_kv]  bfloat16
    attn_weights_ptr,  # [total_rows, seq_kv]  bfloat16
    dropout_mask_ptr,  # [total_rows, seq_kv]  bool (uint8)
    dS_ptr,            # [total_rows, seq_kv]  bfloat16  (output)
    total_rows,        # runtime int
    seq_kv,            # runtime int
    inv_keep_prob,     # runtime float32
    BLOCK_KV: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_CTA

    for i in tl.static_range(ROWS_PER_CTA):
        row_idx = row_start + i
        if row_idx < total_rows:
            base = row_idx * seq_kv

            if SINGLE_PASS:
                offs = tl.arange(0, BLOCK_KV)
                valid = offs < seq_kv

                dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                dot = tl.sum(P_vals * dP_vals, axis=0)
                dS_vals = P_vals * (dP_vals - dot)
                tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)
            else:
                dot = tl.zeros([1], dtype=tl.float32)
                for blk_start in tl.range(0, seq_kv, BLOCK_KV):
                    offs = blk_start + tl.arange(0, BLOCK_KV)
                    valid = offs < seq_kv

                    dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                    dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                    P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                    dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                    dot += tl.sum(P_vals * dP_vals, axis=0)

                for blk_start in tl.range(0, seq_kv, BLOCK_KV):
                    offs = blk_start + tl.arange(0, BLOCK_KV)
                    valid = offs < seq_kv

                    dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                    dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                    P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                    dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                    dS_vals = P_vals * (dP_vals - dot)
                    tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)


# Threshold for shape-adaptive dispatch: total elements
# Below this threshold, multi-stream overhead may dominate; use sequential path
_STREAM_THRESHOLD = 80 * 512 * 512  # ~20M elements


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    device = grad_attn_output.device
    n_bkv  = bs * n_kv_heads

    # Ensure module-level streams and events are initialized (cached)
    _ensure_streams()

    # Get or allocate persistent buffers for this shape
    dO, vs_T_2d, dP_dropped_2d, dV_flat, dS = _get_buffers(bs, seq_q, seq_kv, device)

    # ── Step 1: Custom tiled transpose [bs, sq, 80, d] -> [bs, 80, sq, d] ────
    TILE_SQ = 8
    TILE_H  = 8
    num_tiles_sq = triton.cdiv(seq_q, TILE_SQ)
    num_tiles_h  = triton.cdiv(n_heads, TILE_H)
    transpose_grid = bs * num_tiles_sq * num_tiles_h

    _transpose_sq_heads_kernel[(transpose_grid,)](
        grad_attn_output,
        dO,
        bs, seq_q, n_heads,
        HEAD_DIM=HEAD_DIM,
        TILE_SQ=TILE_SQ,
        TILE_H=TILE_H,
    )

    # dO is now [bs, 80, sq, d], contiguous
    # Reshape to [bs*8, 10*sq, d] for both BMMs
    dO_2d = dO.reshape(n_bkv, n_groups * seq_q, HEAD_DIM)

    # ── Step 2: Transpose value_states via Triton kernel ─────────────────────
    TILE_SKV = 16
    TILE_D   = 16
    num_tiles_skv = triton.cdiv(seq_kv, TILE_SKV)
    num_tiles_d   = triton.cdiv(HEAD_DIM, TILE_D)
    vs_transpose_grid = n_bkv * num_tiles_skv * num_tiles_d

    # value_states is [bs, 8, skv, d] — reshape to [bs*8, skv, d] for the kernel
    vs_2d = value_states.reshape(n_bkv, seq_kv, HEAD_DIM)
    if not vs_2d.is_contiguous():
        vs_2d = vs_2d.contiguous()

    _transpose_vs_kernel[(vs_transpose_grid,)](
        vs_2d,
        vs_T_2d,
        n_bkv, seq_kv,
        HEAD_DIM=HEAD_DIM,
        TILE_SKV=TILE_SKV,
        TILE_D=TILE_D,
    )

    # ── Step 3: Prepare BMM2 input ────────────────────────────────────────────
    P_dropped_2d = attn_weights_dropped.reshape(n_bkv, n_groups * seq_q, seq_kv)
    P_dropped_2d_T = P_dropped_2d.transpose(-2, -1)

    # Flatten for Triton softmax-bwd kernel
    total_rows = bs * n_heads * seq_q
    attn_weights_flat = attn_weights.reshape(total_rows, seq_kv)
    dropout_mask_flat = dropout_mask.reshape(total_rows, seq_kv)

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0
    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 16384)
    SINGLE_PASS = (seq_kv <= BLOCK_KV)
    ROWS_PER_CTA = 4
    grid_size = triton.cdiv(total_rows, ROWS_PER_CTA)

    dS_flat = dS.reshape(total_rows, seq_kv)
    dP_dropped_flat = dP_dropped_2d.reshape(total_rows, seq_kv)

    # ── Shape-adaptive dispatch ───────────────────────────────────────────────
    # For small workloads, skip multi-stream overhead (stream fork/join/sync cost
    # can exceed the benefit of overlapping two independent BMMs).
    workload_size = bs * seq_q * seq_kv * n_heads
    use_streams = (workload_size >= _STREAM_THRESHOLD)

    if use_streams:
        # ── Step 4a: Launch both BMMs concurrently on separate streams ────────
        default_stream = torch.cuda.current_stream()
        _start_event.record(default_stream)

        # Stream A: BMM1 → softmax-bwd
        with torch.cuda.stream(_stream_a):
            _stream_a.wait_event(_start_event)
            # BMM1: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
            torch.bmm(dO_2d, vs_T_2d, out=dP_dropped_2d)

        # Stream B: BMM2
        with torch.cuda.stream(_stream_b):
            _stream_b.wait_event(_start_event)
            # BMM2: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
            torch.bmm(P_dropped_2d_T, dO_2d, out=dV_flat)

        # ── Step 5a: Softmax-bwd on stream A (overlaps with BMM2 on stream B) ─
        with torch.cuda.stream(_stream_a):
            _softmax_bwd_kernel[(grid_size,)](
                dP_dropped_flat,
                attn_weights_flat,
                dropout_mask_flat,
                dS_flat,
                total_rows,
                seq_kv,
                inv_keep_prob,
                BLOCK_KV=BLOCK_KV,
                SINGLE_PASS=SINGLE_PASS,
                ROWS_PER_CTA=ROWS_PER_CTA,
                num_warps=4,
            )

        # ── Step 6a: Sync both streams back to the default stream ─────────────
        _event_a.record(_stream_a)
        _event_b.record(_stream_b)
        default_stream.wait_event(_event_a)
        default_stream.wait_event(_event_b)

    else:
        # ── Step 4b: Sequential path for small workloads ──────────────────────
        # BMM1: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
        torch.bmm(dO_2d, vs_T_2d, out=dP_dropped_2d)

        # Softmax-bwd
        _softmax_bwd_kernel[(grid_size,)](
            dP_dropped_flat,
            attn_weights_flat,
            dropout_mask_flat,
            dS_flat,
            total_rows,
            seq_kv,
            inv_keep_prob,
            BLOCK_KV=BLOCK_KV,
            SINGLE_PASS=SINGLE_PASS,
            ROWS_PER_CTA=ROWS_PER_CTA,
            num_warps=4,
        )

        # BMM2: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
        torch.bmm(P_dropped_2d_T, dO_2d, out=dV_flat)

    # ── Step 7: Reshape outputs ────────────────────────────────────────────────
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #24 — 2026-06-29 23:23:37 UTC ✅ KEEP

**Hypothesis:** Worker implementation

**Result:** 379.11 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- Custom Triton transpose kernel for dO: [bs, sq, 80, d] -> [bs, 80, sq, d]
- Custom Triton transpose kernel for vs_T: [bs, 8, skv, d] -> [bs*8, d, skv]
  (replaces .transpose().reshape().contiguous() which materializes a strided copy)
- Both BMMs as clean 3D batched GEMMs (cuBLAS-optimized, no broadcasting)
- BMM1: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
- BMM2 fused with GQA: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
- Dual-stream pipelining with module-level cached streams/events
  (eliminates Python object allocation overhead on every call)
- Row-batched Triton softmax-bwd kernel on stream A after BMM1
  (overlaps with BMM2 on stream B)
- Persistent buffer cache: pre-allocated tensors keyed by (bs, seq_q, seq_kv)
  to eliminate torch.empty() allocation overhead on the hot path
- Shape-adaptive dispatch: skip multi-stream overhead for small workloads
"""

import torch
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128

# ─────────────────────────────────────────────────────────────────────────────
# Module-level cached CUDA streams and events
# Created once at import/first-call time, reused on every subsequent call
# ─────────────────────────────────────────────────────────────────────────────
_stream_a = None
_stream_b = None
_start_event = None
_event_a = None
_event_b = None


def _ensure_streams():
    global _stream_a, _stream_b, _start_event, _event_a, _event_b
    if _stream_a is None:
        _stream_a = torch.cuda.Stream()
        _stream_b = torch.cuda.Stream()
        _start_event = torch.cuda.Event()
        _event_a = torch.cuda.Event()
        _event_b = torch.cuda.Event()


# ─────────────────────────────────────────────────────────────────────────────
# Persistent buffer cache: keyed by (bs, seq_q, seq_kv)
# Stores pre-allocated tensors: (dO, vs_T_2d, dP_dropped_2d, dV_flat, dS)
# ─────────────────────────────────────────────────────────────────────────────
_buffer_cache = {}


def _get_buffers(bs, seq_q, seq_kv, device):
    key = (bs, seq_q, seq_kv)
    if key not in _buffer_cache:
        n_kv_heads = NUM_KEY_VALUE_HEADS
        n_heads    = NUM_ATTENTION_HEADS
        n_groups   = n_heads // n_kv_heads
        n_bkv      = bs * n_kv_heads

        dO         = torch.empty((bs, n_heads, seq_q, HEAD_DIM),
                                  dtype=torch.bfloat16, device=device)
        vs_T_2d    = torch.empty((n_bkv, HEAD_DIM, seq_kv),
                                  dtype=torch.bfloat16, device=device)
        dP_dropped_2d = torch.empty((n_bkv, n_groups * seq_q, seq_kv),
                                     dtype=torch.bfloat16, device=device)
        dV_flat    = torch.empty((n_bkv, seq_kv, HEAD_DIM),
                                  dtype=torch.bfloat16, device=device)
        dS         = torch.empty((bs, n_heads, seq_q, seq_kv),
                                  dtype=torch.bfloat16, device=device)
        _buffer_cache[key] = (dO, vs_T_2d, dP_dropped_2d, dV_flat, dS)
    return _buffer_cache[key]


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: tiled transpose [bs, sq, n_heads, d] -> [bs, n_heads, sq, d]
#
# Swaps dims 1 (sq) and 2 (n_heads), keeps d as inner dim.
# Each CTA handles a TILE_SQ x TILE_H tile for one batch element.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _transpose_sq_heads_kernel(
    src_ptr,
    dst_ptr,
    bs,
    sq,
    n_heads,
    HEAD_DIM: tl.constexpr,
    TILE_SQ: tl.constexpr,
    TILE_H: tl.constexpr,
):
    num_tiles_sq = tl.cdiv(sq, TILE_SQ)
    num_tiles_h  = tl.cdiv(n_heads, TILE_H)

    pid = tl.program_id(0)
    tile_per_batch = num_tiles_sq * num_tiles_h
    batch_idx   = pid // tile_per_batch
    tile_idx    = pid % tile_per_batch
    tile_sq_idx = tile_idx % num_tiles_sq
    tile_h_idx  = tile_idx // num_tiles_sq

    sq_start = tile_sq_idx * TILE_SQ
    h_start  = tile_h_idx  * TILE_H

    offs_sq = sq_start + tl.arange(0, TILE_SQ)
    offs_h  = h_start  + tl.arange(0, TILE_H)
    offs_d  = tl.arange(0, HEAD_DIM)

    valid_sq = offs_sq < sq
    valid_h  = offs_h  < n_heads

    src_base = batch_idx * sq * n_heads * HEAD_DIM
    src_offsets = (src_base
                   + offs_sq[:, None, None] * (n_heads * HEAD_DIM)
                   + offs_h[None, :, None] * HEAD_DIM
                   + offs_d[None, None, :])
    valid_mask = (valid_sq[:, None, None] & valid_h[None, :, None])

    vals = tl.load(src_ptr + src_offsets, mask=valid_mask, other=0.0)

    dst_base = batch_idx * n_heads * sq * HEAD_DIM
    dst_offsets = (dst_base
                   + offs_h[None, :, None] * (sq * HEAD_DIM)
                   + offs_sq[:, None, None] * HEAD_DIM
                   + offs_d[None, None, :])

    tl.store(dst_ptr + dst_offsets, vals, mask=valid_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: transpose-reshape value_states
#   Input:  [bs * n_kv_heads, skv, d]    (contiguous; view of [bs,8,skv,d])
#   Output: [bs * n_kv_heads, d, skv]    (contiguous)
#
# This replaces: value_states.transpose(-2,-1).reshape(bs*8,d,skv).contiguous()
# Each CTA handles a TILE_SKV x TILE_D tile of (skv, d) for one (b, kv_head).
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _transpose_vs_kernel(
    src_ptr,              # [n_bkv, skv, d]  bfloat16  (contiguous)
    dst_ptr,              # [n_bkv, d, skv]  bfloat16  (output)
    n_bkv,                # bs * n_kv_heads
    skv,                  # seq_kv
    HEAD_DIM: tl.constexpr,
    TILE_SKV: tl.constexpr,
    TILE_D: tl.constexpr,
):
    num_tiles_skv = tl.cdiv(skv, TILE_SKV)
    num_tiles_d   = tl.cdiv(HEAD_DIM, TILE_D)

    pid = tl.program_id(0)
    tiles_per_bkv = num_tiles_skv * num_tiles_d
    bkv_idx  = pid // tiles_per_bkv
    tile_idx = pid % tiles_per_bkv
    tile_skv = tile_idx % num_tiles_skv
    tile_d   = tile_idx // num_tiles_skv

    skv_start = tile_skv * TILE_SKV
    d_start   = tile_d   * TILE_D

    offs_skv = skv_start + tl.arange(0, TILE_SKV)  # [TILE_SKV]
    offs_d   = d_start   + tl.arange(0, TILE_D)    # [TILE_D]

    valid_skv = offs_skv < skv
    valid_d   = offs_d   < HEAD_DIM
    valid_mask = valid_skv[:, None] & valid_d[None, :]

    # Source layout: [n_bkv, skv, d]
    # Element [bkv, s, d_] is at bkv * skv * HEAD_DIM + s * HEAD_DIM + d_
    src_base = bkv_idx * skv * HEAD_DIM
    src_offsets = (src_base
                   + offs_skv[:, None] * HEAD_DIM   # [TILE_SKV, 1]
                   + offs_d[None, :])                # [1, TILE_D]
    # vals shape: [TILE_SKV, TILE_D]
    vals = tl.load(src_ptr + src_offsets, mask=valid_mask, other=0.0)

    # Destination layout: [n_bkv, d, skv]
    # Element [bkv, d_, s] is at bkv * HEAD_DIM * skv + d_ * skv + s
    # We store vals[skv_local, d_local] at dst[bkv, d_start+d_local, skv_start+skv_local]
    # = dst_base + (d_start + d_local) * skv + (skv_start + skv_local)
    dst_base = bkv_idx * HEAD_DIM * skv
    dst_offsets = (dst_base
                   + offs_d[None, :] * skv            # [1, TILE_D] broadcast
                   + offs_skv[:, None])                # [TILE_SKV, 1] broadcast
    # dst_offsets shape: [TILE_SKV, TILE_D] — same shape as vals ✓
    tl.store(dst_ptr + dst_offsets, vals, mask=valid_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: row-batched fused dropout-bwd + softmax-bwd
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _softmax_bwd_kernel(
    dP_dropped_ptr,    # [total_rows, seq_kv]  bfloat16
    attn_weights_ptr,  # [total_rows, seq_kv]  bfloat16
    dropout_mask_ptr,  # [total_rows, seq_kv]  bool (uint8)
    dS_ptr,            # [total_rows, seq_kv]  bfloat16  (output)
    total_rows,        # runtime int
    seq_kv,            # runtime int
    inv_keep_prob,     # runtime float32
    BLOCK_KV: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_CTA

    for i in tl.static_range(ROWS_PER_CTA):
        row_idx = row_start + i
        if row_idx < total_rows:
            base = row_idx * seq_kv

            if SINGLE_PASS:
                offs = tl.arange(0, BLOCK_KV)
                valid = offs < seq_kv

                dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                dot = tl.sum(P_vals * dP_vals, axis=0)
                dS_vals = P_vals * (dP_vals - dot)
                tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)
            else:
                dot = tl.zeros([1], dtype=tl.float32)
                for blk_start in tl.range(0, seq_kv, BLOCK_KV):
                    offs = blk_start + tl.arange(0, BLOCK_KV)
                    valid = offs < seq_kv

                    dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                    dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                    P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                    dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                    dot += tl.sum(P_vals * dP_vals, axis=0)

                for blk_start in tl.range(0, seq_kv, BLOCK_KV):
                    offs = blk_start + tl.arange(0, BLOCK_KV)
                    valid = offs < seq_kv

                    dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                    dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                    P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                    dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                    dS_vals = P_vals * (dP_vals - dot)
                    tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)


# Threshold for shape-adaptive dispatch: total elements
# Below this threshold, multi-stream overhead may dominate; use sequential path
_STREAM_THRESHOLD = 80 * 512 * 512  # ~20M elements


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    device = grad_attn_output.device
    n_bkv  = bs * n_kv_heads

    # Ensure module-level streams and events are initialized (cached)
    _ensure_streams()

    # Get or allocate persistent buffers for this shape
    dO, vs_T_2d, dP_dropped_2d, dV_flat, dS = _get_buffers(bs, seq_q, seq_kv, device)

    # ── Step 1: Custom tiled transpose [bs, sq, 80, d] -> [bs, 80, sq, d] ────
    TILE_SQ = 8
    TILE_H  = 8
    num_tiles_sq = triton.cdiv(seq_q, TILE_SQ)
    num_tiles_h  = triton.cdiv(n_heads, TILE_H)
    transpose_grid = bs * num_tiles_sq * num_tiles_h

    _transpose_sq_heads_kernel[(transpose_grid,)](
        grad_attn_output,
        dO,
        bs, seq_q, n_heads,
        HEAD_DIM=HEAD_DIM,
        TILE_SQ=TILE_SQ,
        TILE_H=TILE_H,
    )

    # dO is now [bs, 80, sq, d], contiguous
    # Reshape to [bs*8, 10*sq, d] for both BMMs
    dO_2d = dO.reshape(n_bkv, n_groups * seq_q, HEAD_DIM)

    # ── Step 2: Transpose value_states via Triton kernel ─────────────────────
    TILE_SKV = 16
    TILE_D   = 16
    num_tiles_skv = triton.cdiv(seq_kv, TILE_SKV)
    num_tiles_d   = triton.cdiv(HEAD_DIM, TILE_D)
    vs_transpose_grid = n_bkv * num_tiles_skv * num_tiles_d

    # value_states is [bs, 8, skv, d] — reshape to [bs*8, skv, d] for the kernel
    vs_2d = value_states.reshape(n_bkv, seq_kv, HEAD_DIM)
    if not vs_2d.is_contiguous():
        vs_2d = vs_2d.contiguous()

    _transpose_vs_kernel[(vs_transpose_grid,)](
        vs_2d,
        vs_T_2d,
        n_bkv, seq_kv,
        HEAD_DIM=HEAD_DIM,
        TILE_SKV=TILE_SKV,
        TILE_D=TILE_D,
    )

    # ── Step 3: Prepare BMM2 input ────────────────────────────────────────────
    P_dropped_2d = attn_weights_dropped.reshape(n_bkv, n_groups * seq_q, seq_kv)
    P_dropped_2d_T = P_dropped_2d.transpose(-2, -1)

    # Flatten for Triton softmax-bwd kernel
    total_rows = bs * n_heads * seq_q
    attn_weights_flat = attn_weights.reshape(total_rows, seq_kv)
    dropout_mask_flat = dropout_mask.reshape(total_rows, seq_kv)

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0
    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 16384)
    SINGLE_PASS = (seq_kv <= BLOCK_KV)
    ROWS_PER_CTA = 4
    grid_size = triton.cdiv(total_rows, ROWS_PER_CTA)

    dS_flat = dS.reshape(total_rows, seq_kv)
    dP_dropped_flat = dP_dropped_2d.reshape(total_rows, seq_kv)

    # ── Shape-adaptive dispatch ───────────────────────────────────────────────
    # For small workloads, skip multi-stream overhead (stream fork/join/sync cost
    # can exceed the benefit of overlapping two independent BMMs).
    workload_size = bs * seq_q * seq_kv * n_heads
    use_streams = (workload_size >= _STREAM_THRESHOLD)

    if use_streams:
        # ── Step 4a: Launch both BMMs concurrently on separate streams ────────
        default_stream = torch.cuda.current_stream()
        _start_event.record(default_stream)

        # Stream A: BMM1 → softmax-bwd
        with torch.cuda.stream(_stream_a):
            _stream_a.wait_event(_start_event)
            # BMM1: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
            torch.bmm(dO_2d, vs_T_2d, out=dP_dropped_2d)

        # Stream B: BMM2
        with torch.cuda.stream(_stream_b):
            _stream_b.wait_event(_start_event)
            # BMM2: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
            torch.bmm(P_dropped_2d_T, dO_2d, out=dV_flat)

        # ── Step 5a: Softmax-bwd on stream A (overlaps with BMM2 on stream B) ─
        with torch.cuda.stream(_stream_a):
            _softmax_bwd_kernel[(grid_size,)](
                dP_dropped_flat,
                attn_weights_flat,
                dropout_mask_flat,
                dS_flat,
                total_rows,
                seq_kv,
                inv_keep_prob,
                BLOCK_KV=BLOCK_KV,
                SINGLE_PASS=SINGLE_PASS,
                ROWS_PER_CTA=ROWS_PER_CTA,
                num_warps=4,
            )

        # ── Step 6a: Sync both streams back to the default stream ─────────────
        _event_a.record(_stream_a)
        _event_b.record(_stream_b)
        default_stream.wait_event(_event_a)
        default_stream.wait_event(_event_b)

    else:
        # ── Step 4b: Sequential path for small workloads ──────────────────────
        # BMM1: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
        torch.bmm(dO_2d, vs_T_2d, out=dP_dropped_2d)

        # Softmax-bwd
        _softmax_bwd_kernel[(grid_size,)](
            dP_dropped_flat,
            attn_weights_flat,
            dropout_mask_flat,
            dS_flat,
            total_rows,
            seq_kv,
            inv_keep_prob,
            BLOCK_KV=BLOCK_KV,
            SINGLE_PASS=SINGLE_PASS,
            ROWS_PER_CTA=ROWS_PER_CTA,
            num_warps=4,
        )

        # BMM2: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
        torch.bmm(P_dropped_2d_T, dO_2d, out=dV_flat)

    # ── Step 7: Reshape outputs ────────────────────────────────────────────────
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #25 — 2026-06-29 23:26:19 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 401.46 μs

---

## Experiment #26 — 2026-06-29 23:28:58 UTC ❌ DISCARD

**Hypothesis:** ** Added `_fused_transpose_bmm1_kernel` — a tiled GEMM Triton kernel that reads `grad_attn_output` at native strides `[bs, sq, n_heads, d]` and `vs_2d` at `[n_bkv, skv, d]`, computing `dP_dropped_2d[n

**Result:** 547.61 μs

