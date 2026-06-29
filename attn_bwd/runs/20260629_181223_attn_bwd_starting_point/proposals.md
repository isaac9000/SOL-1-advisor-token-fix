# Advisor Proposals

---

## Iteration 1

## STATE
Only the baseline exists — a pure PyTorch reference at 3483 μs geomean. The speed-of-light target is ~130 μs geomean (score ≈9.3), so we're roughly 27× away. No custom CUDA or Triton work has been attempted yet. The baseline suffers from multiple inefficiencies: FP32 upcasts everywhere, no fused kernels, two separate batched-matmul passes with an intermediate expanded tensor, and no exploitation of the GQA structure.

## RATIONALE
The computation consists of two batched matrix multiplications (dP = dO @ V^T and dV_exp = P̃^T @ dO), a masked scaling (dropout backward), a softmax backward (elementwise), and a GQA group reduction. The heaviest bottleneck is the pair of BMMs operating on large [bs, 80, sq, skv] tensors entirely in FP32. Triton is the right tool here: we can write fused kernels that (a) stay in bfloat16/float32 mixed precision, (b) never materialize the fully expanded value tensor, and (c) fuse the softmax-backward elementwise ops directly into the first BMM's output pipeline. A Triton flash-attention-style backward implementation that handles the GQA structure natively (computing the 10-group reduction as an accumulation rather than expanding to 80 heads) should cut memory traffic dramatically.

## PROPOSAL
Implement a **Triton-based fused attention backward kernel** structured as follows:

**Strategy:** Write two Triton kernels — one for `grad_attn_scores` and one for `grad_value_states` — that exploit GQA structure without expanding value_states to 80 heads.

1. **Kernel 1 (grad_attn_scores):** For each (batch, head, sq_tile) block, load a tile of `dO` [sq_tile × head_dim], load the corresponding GQA-mapped value tile `V[kv_head, :, :]` (where `kv_head = head // 10`), compute `dP̃ = dO @ V^T` in bfloat16 accumulating in fp32, apply the dropout mask and scale, then run the softmax backward in-register. Emit `grad_attn_scores` directly in bfloat16.

2. **Kernel 2 (grad_value_states):** For each (batch, kv_head, skv_tile) block, loop over the 10 query-heads in the group, load tiles of `attn_weights_dropped` and `dO`, accumulate `dV += P̃^T @ dO` across all 10 heads, then write out the reduced `grad_value_states` in bfloat16. This avoids the 10× memory expansion and reduces the output tensor write by 10×.

The worker should use Triton's `tl.dot` for the inner matrix multiplications, keep intermediate accumulation in float32, and select tile sizes that maximize SM occupancy on the B200. Use `tl.constexpr` for compile-time tile dimensions and test a few reasonable tile configurations. Fall back to the PyTorch baseline for correctness checking during development.

---

## Iteration 2

## STATE
Two runs so far. The Triton baseline achieved 1274.79 μs vs the PyTorch 3483.45 μs — a 2.7× improvement, but the SOL target is ~130 μs (geomean), so we're still ~10× away from SOL. The Triton kernel is young and untuned: it uses a 3-pass strategy for `dS` (write intermediate dP, reload for row-sum, reload again to compute dS) which triples the memory traffic for that kernel. The `dV` kernel also has an inner loop over 10 GQA groups and 32-wide sq tiles, which likely underutilizes the tensor cores.

## RATIONALE
The biggest single optimization opportunity is the `dS` kernel's 3-pass design. Currently it writes dP to HBM, reads it back for the row-sum, then reads it again to write dS — that's 3× the minimum memory traffic. A proper 2-pass approach (pass 1: accumulate the row-sum across all skv tiles in registers while also computing dP*P, pass 2: reload dP and P to compute final dS) is still two passes over the large [bs, 80, sq, skv] tensor, but a truly 1-pass approach for `dS` is possible if we process the entire skv dimension without tiling (i.e., block size = skv for small sequences) or use a smarter register-blocking scheme. Additionally, the kernel currently does 3 passes through HBM for the large attention weight tensor per head, which is extremely expensive for large sequences. The `dV` kernel tile sizes (BLOCK_SQ=32, BLOCK_D=128) are reasonable but may not be tuned optimally for B200's tensor core dimensions.

## PROPOSAL
**Restructure the `dS` kernel to eliminate the intermediate HBM write of dP.** Instead of 3 passes, restructure to 2 passes:

1. **Pass 1:** For each (batch, head, sq_row), iterate over all skv tiles, compute `dP_tile = dO @ V^T`, apply dropout mask, then accumulate `row_sum += sum(dP_tile * P_tile, dim=-1)` into a register-resident `[BLOCK_SQ]` vector. Do NOT write anything to HBM in this pass.

2. **Pass 2:** Iterate over skv tiles again — reload P, recompute dP (recompute `dO @ V^T`, apply dropout), compute `dS = P * (dP - row_sum[:, None])`, write dS in bfloat16.

This halves the HBM traffic for the `dS` computation (P and dropout_mask are still read twice, V is read twice, but no intermediate dP is written). For large cases this is the dominant bottleneck.

Additionally, **increase the sq tile size in the dV kernel** and experiment with larger BLOCK_SQ values (e.g., matching the tensor core's preferred matrix dimensions) to improve arithmetic intensity. For the dV kernel, consider restructuring the loop order so that the outer loop is over sq tiles and the inner loop is over GQA groups, which may allow better prefetching patterns.

The worker should also try **eliminating the `.contiguous()` transpose of `dO`** at the Python level by passing the original `[bs, sq, 80, 128]` tensor directly with adjusted strides, avoiding one HBM copy entirely.

---

## Iteration 3

## STATE
Experiment #3 (the 2-pass dS refactor) was discarded at 1570 μs — actually slower than the current best of 1274 μs. This is likely because the approach re-reads P and V from HBM twice in loops rather than once, and the extra kernel complexity hurt L2 reuse patterns. The Triton 2-kernel solution at 1274 μs remains the best but is still ~10× from SOL (~130 μs). The key structural issue is the 3-pass dS kernel that re-traverses the large attention weight tensors repeatedly.

## RATIONALE
The fundamental problem with the current dS kernel is the 3-pass loop structure over the massive [bs, 80, sq, skv] attention weights. For large cases (e.g., bs=32, sq=691, skv=773), this tensor is enormous and HBM bandwidth is the bottleneck. However, we can avoid separate passes by using a **row-tile approach**: if we process the entire skv dimension within a single kernel block (when skv fits in SRAM), we can compute the row-sum in registers without any HBM writes. The critical insight is that the softmax backward requires the row-sum over the entire skv dimension, which requires a full-row tile or a 2-pass approach. The 2-pass approach (pass 1: compute row_sum only, pass 2: write final dS) halves the number of dS writes but still needs to reread the data. A more radical alternative is to switch to **cuBLAS/PyTorch batched matmul for the BMM steps** (which are already highly optimized on B200) and only use Triton for the elementwise softmax-backward fusion — this would leverage NVIDIA's hand-tuned tensor core code for the dominant operations.

## PROPOSAL
**Switch strategy: Use cuBLAS-backed PyTorch for BMMs + fused Triton elementwise kernel.**

The two BMMs (`dO @ V^T` and `Pd^T @ dO`) are the dominant FLOPs and PyTorch's cuBLAS path is extremely well-tuned for B200. The elementwise operations (dropout backward + softmax backward) are memory-bandwidth-bound but small. Structure the solution as:

1. **PyTorch BMM for dP:** Expand value_states for GQA using `.expand().reshape()` (no copy, just a view change to stride the 10 groups), then call `torch.bmm` or `torch.matmul` for `dP = dO @ V^T`. Use bfloat16 throughout (no FP32 upcast — B200 handles bf16 tensor cores natively and the accuracy is sufficient per rtol=1e-2).

2. **Fused Triton elementwise kernel:** Take dP [bs, 80, sq, skv], attn_weights P [bs, 80, sq, skv], dropout_mask — compute the dropout scale, the row-sum of dP*P, and the final dS = P*(dP - row_sum) all in one kernel pass. This kernel is purely memory-bandwidth-bound, processes each row sequentially, and only reads P and dP once each while writing dS once.

3. **PyTorch BMM for dV:** Use `torch.matmul(Pd.transpose(-2,-1), dO)` followed by `.reshape(...).sum(dim=2)` for the GQA reduction. The reshape+sum is well-optimized by PyTorch/cuDNN.

The key insight: PyTorch's cuBLAS BMMs are faster than any Triton-written equivalent for large, regular matmul shapes (they exploit all tensor core optimizations, auto-tuning, and pipeline heuristics that Triton cannot match for standard sizes). The only fusion value is in the elementwise softmax-backward step. This mixed approach should be significantly faster by not re-implementing what cuBLAS already does optimally.

---

## Iteration 4

## STATE
Four experiments so far. The hybrid cuBLAS+Triton approach (Experiment #4) is the current best at 1079 μs, a modest improvement over the pure Triton approach at 1274 μs. Both are still ~8× from SOL (~130 μs). The hybrid approach uses cuBLAS for the two BMMs and a Triton kernel for the softmax backward — a sound architecture — but the implementation has inefficiencies: a `.contiguous()` copy on `vs_exp` (expanding 8→80 heads in bf16, materializing a 10× larger tensor), and another `.contiguous()` on `dO`. The softmax backward Triton kernel is a 2-pass (row-per-program) design that reads P and dP twice.

## RATIONALE
The dominant cost driver now is almost certainly the **GQA expansion copy** (`vs_exp.contiguous()`) which materializes the full [bs, 80, skv, 128] value tensor — 10× larger than the input value_states. For large cases (bs=32, skv=773), this is ~32×80×773×128×2 bytes = 512 MB of HBM writes+reads just for this expansion. Eliminating this copy is the single highest-leverage optimization. Instead of expanding V to all 80 heads, we should restructure the dP BMM to process the 10 GQA groups natively: for each kv-head, do `dP[bs, kv_h, 10, sq, skv] = dO[bs, kv_h, 10, sq, 128] @ V[bs, kv_h, skv, 128]^T`, which requires reshaping dO (not V) and using a grouped BMM pattern. Similarly, the `dO.contiguous()` copy materializes another large tensor that could be avoided.

## PROPOSAL
**Eliminate the V expansion copy by restructuring the BMM to use GQA-native batching.**

The key insight: instead of expanding V from [bs, 8, skv, 128] to [bs, 80, skv, 128], reshape dO from [bs, 80, sq, 128] to [bs, 8, 10, sq, 128], then batch the matmul as [bs, 8, 10, sq, 128] @ [bs, 8, 1, 128, skv] = [bs, 8, 10, sq, skv]. The broadcast over the group dimension (10) means V is read once per kv-head rather than 10 times. This can be done with `torch.matmul` using broadcasting — PyTorch will broadcast the `[bs, 8, 1, 128, skv]` dimension across the 10 groups without materializing the expanded tensor.

Concretely:
1. Reshape `dO_c` from `[bs, 80, sq, 128]` → `[bs, 8, 10, sq, 128]`
2. Transpose `value_states` to `[bs, 8, skv, 128]` → `[bs, 8, 128, skv]`, then unsqueeze to `[bs, 8, 1, 128, skv]`
3. `dP_grouped = dO_grouped @ V_t` → `[bs, 8, 10, sq, skv]` — PyTorch broadcasts V across the 10 groups
4. Reshape to `[bs, 80, sq, skv]` for the softmax backward step

For the dV computation, symmetrically reshape `attn_weights_dropped` to `[bs, 8, 10, sq, skv]`, then `dV = (Pd_grouped.transpose(-2,-1) @ dO_grouped).sum(dim=2)` — this avoids the 10× expansion and the final reshape+sum can be fused.

Also, eliminate the `dO.contiguous()` copy by using the original `[bs, sq, 80, 128]` tensor directly with stride-aware matmul, or make `dO` contiguous in `[bs, 80, sq, 128]` format once and reuse it for both BMMs. The softmax-backward Triton kernel should also be examined: if `attn_weights` is not already contiguous, that `.contiguous()` call is another expensive copy that can be removed.

---

## Iteration 5

## STATE
Five experiments, best at 1079 μs (Experiment #4, hybrid cuBLAS+Triton). Experiment #5 (GQA-native batched BMM without V expansion) came in at 1147 μs — a discard, but only marginally worse than the best, suggesting it's on the right track but not yet tuned. The gap to SOL (~130 μs geomean) is still ~8×. The current winner materializes an expensive `vs_exp.contiguous()` copy (10× expansion of value_states) and separately copies `dO` to contiguous. Both BMMs are still operating on unnecessarily large tensors.

## RATIONALE
The two critical inefficiencies remaining in Experiment #4 are: (1) the `vs_exp.contiguous()` materializes a [bs, 80, skv, 128] bf16 tensor — for large cases this is enormous; (2) the `dP = dP * dropout_mask * dropout_scale` elementwise op is a separate kernel pass over the large [bs, 80, sq, skv] tensor, then the softmax-backward Triton kernel is another 2-pass over the same tensor. We can fuse these together. The dropout backward and the softmax backward can be a single Triton kernel that reads `dP` (from the BMM output) once and writes `dS` once, computing row-sums in registers. But more importantly, the entire architecture is constrained by materializing massive intermediate tensors. The right approach is a **single fully-fused Triton kernel for each output** that avoids all intermediate tensor allocations.

## PROPOSAL
**Pursue a fully-fused single-kernel approach for `grad_attn_scores` that eliminates all intermediate tensor allocations.**

The architecture should be:

**Kernel 1 (grad_attn_scores):** A single Triton kernel that for each (batch, head, sq_row) block:
- Loads `dO` tile [sq_block × 128] once from HBM — note: use the *original* `grad_attn_output` [bs, sq, 80, 128] directly with appropriate strides to avoid the `.contiguous()` transpose copy
- Iterates over skv tiles: for each tile, loads V[kv_head, skv_tile, :] (128 elements) and computes `dP = dO @ V^T` via `tl.dot`, loads the dropout mask, applies scale, loads P, accumulates `row_sum += sum(dP*P)` in registers
- Then makes a **second pass** over skv tiles: recompute `dP = dO @ V^T` (V is already in L2 from first pass), apply dropout, load P, compute `dS = P*(dP - row_sum)`, write dS to output
- This is a 2-pass kernel but avoids all intermediate HBM allocations (no dP tensor written to memory)

**Kernel 2 (grad_value_states):** A Triton kernel with GQA-native looping — for each (batch, kv_head, skv_tile), loop over the 10 groups and sq tiles, accumulate `dV += Pd^T @ dO` using the original `grad_attn_output` with stride access (no contiguous copy of dO needed). The key is passing `grad_attn_output` directly with its [bs, sq, 80, 128] strides and computing the appropriate pointer offsets.

**Critical optimization:** Avoid all `.contiguous()` copies in the Python wrapper — pass tensors with their native strides and let Triton handle the non-contiguous access patterns. The only mandatory copy is the one that creates `dO_c` for cuBLAS, but if we're doing fully-fused Triton kernels, we don't need cuBLAS at all.

This strategy eliminates: (1) the V expansion copy, (2) the dO transpose copy, (3) the intermediate dP tensor, (4) the dropout backward pass. The total HBM traffic should be roughly: read dO once, read V once per kv-head (shared across 10 query heads), read P twice (both passes), read dropout_mask twice, write dS once. This is close to the memory-bandwidth optimum for this computation.

