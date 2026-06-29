# Experiment History

Tracks every kernel attempt, its code, hypothesis, and result.

---

## Experiment #1 — 2026-06-29 20:20:57 UTC ✅ KEEP

**Hypothesis:** Baseline '0619_best' — initial benchmark

**Result:** 394.73 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel:
- Both matmuls use group-reshape [bs*8, 10*sq, ...] — no GQA expansion anywhere.
- dP and dV bmms launched concurrently on separate CUDA streams.
- Module-level cached stream to avoid creation overhead in hot path.
- Pre-allocated output tensors before any stream switching.
- dV: direct attn.T @ dO -> [bs*8, skv, 128] (no post-transpose copy).
- Triton softmax-backward with row batching overlaps with dV on side stream.
- All in bfloat16.

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

# Module-level cached CUDA stream and event (created once, reused every call)
_side_stream = None
_dO_ready_event = None

def _get_side_stream(device):
    global _side_stream, _dO_ready_event
    if _side_stream is None:
        _side_stream = torch.cuda.Stream(device)
        _dO_ready_event = torch.cuda.Event()
    return _side_stream, _dO_ready_event


@triton.jit
def fused_softmax_bwd_batched(
    dP_dropped_ptr,
    P_ptr,
    dropout_mask_ptr,
    dS_ptr,
    total_rows,
    scale: tl.constexpr,
    seq_kv: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    """
    Batched softmax-backward kernel: each program handles ROWS_PER_BLOCK rows.
    Grid: ceil(total_rows / ROWS_PER_BLOCK)
    """
    block_id = tl.program_id(0)
    row_start = block_id * ROWS_PER_BLOCK

    for r in tl.static_range(ROWS_PER_BLOCK):
        row_id = row_start + r
        if row_id < total_rows:
            base = row_id * seq_kv

            if BLOCK_SKV >= seq_kv:
                offsets = tl.arange(0, BLOCK_SKV)
                mask_bounds = offsets < seq_kv

                dp_dropped = tl.load(dP_dropped_ptr + base + offsets,
                                     mask=mask_bounds, other=0.0).to(tl.float32)
                dm = tl.load(dropout_mask_ptr + base + offsets,
                             mask=mask_bounds, other=0).to(tl.int1)
                p = tl.load(P_ptr + base + offsets,
                            mask=mask_bounds, other=0.0).to(tl.float32)

                dp = tl.where(dm, dp_dropped * scale, 0.0)
                row_sum = tl.sum(dp * p, axis=0)
                ds = p * (dp - row_sum)

                tl.store(dS_ptr + base + offsets, ds.to(tl.bfloat16), mask=mask_bounds)
            else:
                row_sum = tl.zeros([1], dtype=tl.float32)
                for block_start in range(0, seq_kv, BLOCK_SKV):
                    offsets = block_start + tl.arange(0, BLOCK_SKV)
                    mask_bounds = offsets < seq_kv
                    dp_dropped = tl.load(dP_dropped_ptr + base + offsets,
                                         mask=mask_bounds, other=0.0).to(tl.float32)
                    dm = tl.load(dropout_mask_ptr + base + offsets,
                                 mask=mask_bounds, other=0).to(tl.int1)
                    p = tl.load(P_ptr + base + offsets,
                                mask=mask_bounds, other=0.0).to(tl.float32)
                    dp = tl.where(dm, dp_dropped * scale, 0.0)
                    row_sum += tl.sum(dp * p, axis=0)

                for block_start in range(0, seq_kv, BLOCK_SKV):
                    offsets = block_start + tl.arange(0, BLOCK_SKV)
                    mask_bounds = offsets < seq_kv
                    dp_dropped = tl.load(dP_dropped_ptr + base + offsets,
                                         mask=mask_bounds, other=0.0).to(tl.float32)
                    dm = tl.load(dropout_mask_ptr + base + offsets,
                                 mask=mask_bounds, other=0).to(tl.int1)
                    p = tl.load(P_ptr + base + offsets,
                                mask=mask_bounds, other=0.0).to(tl.float32)
                    dp = tl.where(dm, dp_dropped * scale, 0.0)
                    ds = p * (dp - row_sum)
                    tl.store(dS_ptr + base + offsets, ds.to(tl.bfloat16), mask=mask_bounds)


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

    # =========================================================================
    # Step 1: Make dO contiguous in [bs, 80, sq, 128] layout (bfloat16).
    # =========================================================================
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()
    # dO: [bs, 80, sq, 128], bfloat16, contiguous

    # Shared group-reshape for both matmuls: [bs*8, 10*sq, 128] — free view
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # Prepare matmul operands (all free views, no copies)
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # =========================================================================
    # Pre-allocate output tensors on the CURRENT stream before any switching.
    # =========================================================================
    # dP output: [bs*8, 10*sq, skv]
    dP_groups = torch.empty(
        (bs * n_kv_heads, n_groups * seq_q, seq_kv),
        dtype=torch.bfloat16, device=device
    )
    # dV output: [bs*8, skv, 128] — direct final layout, no post-transpose needed.
    # attn.T @ dO: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
    # attn_groups_flat.transpose(-2,-1) is a non-contiguous [bs*8, skv, 10*sq] view,
    # cuBLAS handles this as TN GEMM (transpose first arg). Output is contiguous.
    dV_flat = torch.empty(
        (bs * n_kv_heads, seq_kv, HEAD_DIM),
        dtype=torch.bfloat16, device=device
    )

    # =========================================================================
    # Step 2: Concurrent stream execution.
    # Both matmuls read from dO_groups_flat (concurrent reads are safe).
    # - Side stream: dV bmm (attn.T @ dO → directly contiguous [bs*8, skv, 128])
    # - Main stream: dP bmm → Triton softmax
    # =========================================================================
    main_stream = torch.cuda.current_stream(device)
    side_stream, dO_ready_event = _get_side_stream(device)

    # Record event: dO is ready on the main stream
    dO_ready_event.record(main_stream)

    # Side stream waits for dO to be ready, then launches dV
    side_stream.wait_event(dO_ready_event)
    with torch.cuda.stream(side_stream):
        # dV: bmm([bs*8, skv, 10*sq], [bs*8, 10*sq, 128]) -> [bs*8, skv, 128]
        # attn_groups_flat.T is non-contiguous: cuBLAS TN (transpose-N) GEMM
        # Output dV_flat is directly contiguous [bs*8, skv, 128] — no post-copy.
        torch.bmm(attn_groups_flat.transpose(-2, -1), dO_groups_flat, out=dV_flat)

    # Launch dP on main stream (concurrent with dV on side stream)
    # dP: bmm([bs*8, 10*sq, 128], [bs*8, 128, skv]) -> [bs*8, 10*sq, skv]
    torch.bmm(dO_groups_flat, vs_flat.transpose(-2, -1), out=dP_groups)

    # =========================================================================
    # Step 3: Fused softmax backward + dropout correction via Triton.
    # Runs on main stream — overlaps with dV on side stream.
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    total_rows = bs * n_heads * seq_q

    dP_dropped_flat = dP_groups.reshape(total_rows, seq_kv)
    P_flat  = attn_weights.reshape(total_rows, seq_kv)
    dm_flat = dropout_mask.reshape(total_rows, seq_kv)

    dS_flat = torch.empty((total_rows, seq_kv), dtype=torch.bfloat16, device=device)

    # Choose BLOCK_SKV and ROWS_PER_BLOCK based on seq_kv
    if seq_kv <= 128:
        BLOCK_SKV = 128
        ROWS_PER_BLOCK = 16
    elif seq_kv <= 256:
        BLOCK_SKV = 256
        ROWS_PER_BLOCK = 8
    elif seq_kv <= 512:
        BLOCK_SKV = 512
        ROWS_PER_BLOCK = 4
    elif seq_kv <= 1024:
        BLOCK_SKV = 1024
        ROWS_PER_BLOCK = 2
    elif seq_kv <= 2048:
        BLOCK_SKV = 2048
        ROWS_PER_BLOCK = 1
    else:
        BLOCK_SKV = 2048
        ROWS_PER_BLOCK = 1

    num_blocks = (total_rows + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK

    fused_softmax_bwd_batched[(num_blocks,)](
        dP_dropped_flat, P_flat, dm_flat, dS_flat,
        total_rows=total_rows,
        scale=scale,
        seq_kv=seq_kv,
        BLOCK_SKV=BLOCK_SKV,
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
    )

    dS = dS_flat.reshape(bs, n_heads, seq_q, seq_kv)

    # Wait for side stream (dV) to complete — dV_flat is already in final layout
    main_stream.wait_stream(side_stream)

    # dV_flat is already [bs*8, skv, 128] contiguous — just reshape
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV

```

---

## Experiment #2 — 2026-06-29 20:22:41 UTC 💥 CRASH

**Hypothesis:** ** Replaced the separate `torch.bmm(dO, V^T) → dP_groups` + `fused_softmax_bwd_batched` two-kernel sequence with a single `fused_dp_softmax_bwd_kernel` Triton kernel that:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #3 — 2026-06-29 20:24:14 UTC ❌ DISCARD

**Hypothesis:** ** Replaced `fused_softmax_bwd_batched` with `fused_softmax_bwd_kernel` — a completely rewritten Triton kernel where:

**Result:** 403.13 μs

---

## Experiment #4 — 2026-06-29 20:26:28 UTC ❌ DISCARD

**Hypothesis:** ** Replaced the two-step approach (separate BMM + Triton softmax kernel) with a single `fused_dp_softmax_bwd_kernel_v2` Triton kernel that:

**Result:** 5300.94 μs

---

## Experiment #5 — 2026-06-29 20:28:05 UTC ❌ DISCARD

**Hypothesis:** Rewrote `submission.py` to define `_attn_bwd_core` as a clean PyTorch function containing both BMMs (in GQA-reshaped form via `torch.bmm`) and the softmax-backward elementwise block. Wrapped it with `

**Result:** 915.39 μs

---

## Experiment #6 — 2026-06-29 20:29:54 UTC ❌ DISCARD

**Hypothesis:** Replaced the previous (already compiled but potentially buggy) version with a clean, correct implementation using `torch.compile(mode="max-autotune", fullgraph=True)`. Key fixes:

**Result:** 1992.74 μs

---

## Experiment #7 — 2026-06-29 20:31:23 UTC ❌ DISCARD

**Hypothesis:** Replaced the `torch.compile`-based submission with a two-stream concurrent BMM architecture + a new Triton softmax-backward kernel that maps one Triton program per row. The kernel uses `BLOCK_SKV = ne

**Result:** 2924.50 μs

---

## Experiment #8 — 2026-06-29 20:32:46 UTC ❌ DISCARD

**Hypothesis:** Replaced `_softmax_bwd_kernel` with `_softmax_bwd_fused_kernel` that accepts the raw BMM output `dP_raw`, the boolean `dropout_mask`, and `P` directly. The dropout masking and scaling (`tl.where(dm, d

**Result:** 2367.27 μs

---

## Experiment #9 — 2026-06-29 20:33:57 UTC 💥 CRASH

**Hypothesis:** ** Two targeted changes to the existing submission:

**Result:** CRASH

**Error:**
```
Correctness check failed. Failed to load submission:
Traceback (most recent call last):
  File "/root/eval_modal_attn_bwd.py", line 246, in evaluate_kernel
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 936, in exec_module
  File "<frozen importlib._bootstrap_external>", line 1074, in get_code
  File "<frozen importlib._bootstrap_external>", line 1004, in source_to_code
  File "<froz
```

---

## Experiment #10 — 2026-06-29 20:35:24 UTC ❌ DISCARD

**Hypothesis:** ** Replaced `fused_softmax_bwd_batched` (ROWS_PER_BLOCK batching, complex inner loop) with `fused_softmax_bwd_1row` (one program per row, grid = `(total_rows,)`). The kernel handles the single-pass ca

**Result:** 399.08 μs

---

## Experiment #11 — 2026-06-29 20:36:29 UTC ❌ DISCARD

**Hypothesis:** ** Removed the entire `fused_softmax_bwd_1row` Triton kernel and its invocation logic. Replaced with 4 lines of PyTorch: `torch.where` for dropout masking + scaling, `dp * P` + `.sum(dim=-1, keepdim=T

**Result:** 1933.24 μs

---

## Experiment #12 — 2026-06-29 20:37:58 UTC 💥 CRASH

**Hypothesis:** ** A single `_fused_attn_bwd_kernel` Triton kernel that:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #13 — 2026-06-29 20:39:47 UTC 💥 CRASH

**Hypothesis:** ** Added `fused_dP_softmax_bwd_kernel` — a Triton kernel where each program handles one `(batch*head, q_row_block)` pair. It does two passes over KV tiles: Pass 1 loads dO tile (kept in registers), it

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #14 — 2026-06-29 20:41:21 UTC ❌ DISCARD

**Hypothesis:** ** Removed the `_side_stream`, `_dO_ready_event`, `_get_side_stream()` globals and all stream-related code (`event.record`, `stream.wait_event`, `with torch.cuda.stream(...)`, `main_stream.wait_stream

**Result:** 404.47 μs

---

## Experiment #15 — 2026-06-29 20:44:56 UTC ❌ DISCARD

**Hypothesis:** - Reshape `dO` as `[bs, sq, 8, 10, 128]` (free view), then do ONE `permute(0,2,3,1,4).contiguous()` → `[bs, 8, 10, sq, 128]`

**Result:** 599.21 μs

---

## Experiment #16 — 2026-06-29 20:46:38 UTC ❌ DISCARD

**Hypothesis:** 1. Removed the Triton softmax backward kernel entirely

**Result:** 1953.94 μs

---

## Experiment #17 — 2026-06-29 20:48:39 UTC 💥 CRASH

**Hypothesis:** **

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #18 — 2026-06-29 20:51:04 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 506.99 μs

---

## Experiment #19 — 2026-06-29 20:52:27 UTC 💥 CRASH

**Hypothesis:** **

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #20 — 2026-06-29 20:53:54 UTC ❌ DISCARD

**Hypothesis:** Replaced the old `fused_softmax_bwd_batched` kernel with a new `softmax_bwd_one_row_per_program` kernel. The new kernel takes `grid = (total_rows,)` — exactly one Triton program per attention row. For

**Result:** 401.32 μs

---

## Experiment #21 — 2026-06-29 20:55:39 UTC 💥 CRASH

**Hypothesis:** ** A complete Triton kernel `flash_attn_bwd_kernel` with:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

---

## Experiment #22 — 2026-06-29 20:59:49 UTC ❌ DISCARD

**Hypothesis:** Instead of `grad_attn_output.permute(0,2,1,3).contiguous()` which creates `[bs, 80, sq, 128]`, the code now does: `reshape(bs, sq, 8, 10, 128)` → `permute(0, 2, 3, 1, 4).contiguous()` → `reshape(bs*8,

**Result:** 447.29 μs

---

## Experiment #23 — 2026-06-29 21:01:19 UTC 💥 CRASH

**Hypothesis:** ** Replaced `fused_softmax_bwd_batched` (with complex `static_range` batching and multi-pass loops) with `softmax_bwd_one_row_per_program` — a single-pass kernel where each Triton program handles exac

**Result:** CRASH

**Error:**
```
Correctness check failed. Failed to load submission:
Traceback (most recent call last):
  File "/root/eval_modal_attn_bwd.py", line 246, in evaluate_kernel
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 936, in exec_module
  File "<frozen importlib._bootstrap_external>", line 1074, in get_code
  File "<frozen importlib._bootstrap_external>", line 1004, in source_to_code
  File "<froz
```

---

## Experiment #24 — 2026-06-29 21:03:13 UTC ❌ DISCARD

**Hypothesis:** ** Replaced the two-step `permute(0,2,1,3).contiguous()` → `reshape(bs*8, 10*sq, 128)` (which allocates `[bs, 80, sq, 128]` then views it) with a fused chain: `reshape(bs, sq, 8, 10, 128).permute(0,2,

**Result:** 396.12 μs

---

## Experiment #25 — 2026-06-29 21:04:52 UTC ❌ DISCARD

**Hypothesis:** Replaced the Triton softmax-backward + concurrent-stream BMM baseline with a fully PyTorch-ops core function compiled via `torch.compile(mode="reduce-overhead", dynamic=True, fullgraph=True)`. Added a

**Result:** 735.63 μs

---

## Experiment #26 — 2026-06-29 21:06:13 UTC ❌ DISCARD

**Hypothesis:** Added `.contiguous()` calls on both transposed tensors before BMM:

**Result:** 795.90 μs

