# Advisor Proposals

---

## Iteration 1

## STATE
Only the baseline exists (3428.46 μs geomean). No custom kernels have been attempted. The SOL target is ~137 μs geomean (score ≈9.3), so we're about 25× away. The baseline does all computation in float32 with repeated memory passes and no fusion. The primary bottlenecks are: (1) the large intermediate tensors for the GQA-expanded value states, (2) multiple separate kernel launches for elementwise ops, and (3) no use of hardware-accelerated matrix operations via cuBLAS/CUTLASS through Triton.

## RATIONALE
The fastest path to large wins is fusing the elementwise operations (dropout backward, softmax backward) with the memory-bound passes and using efficient batched matrix multiplications. Flash-attention style Triton kernels are well-suited here. The GQA structure (10 groups) means the bmm for dV should sum over groups without materializing the full expanded [bs,80,skv,128] intermediate — this alone should give significant speedup. The two BMMs (dP = dO @ V^T and dV = P_dropped^T @ dO) dominate for large sequences; for small cases, kernel launch overhead and memory bandwidth dominate.

## PROPOSAL

Implement a **Triton-based fused kernel** for this attention backward pass with two main components:

**Strategy:**

1. **Fuse the elementwise chain into the BMM for grad_attn_scores:** Write a Triton kernel that computes `dP = dO @ V^T` (using tiled matrix multiply over the seq_kv dimension), then immediately applies the dropout mask scaling and the softmax backward formula (`P * (dP - sum(dP*P))`) all in one kernel, outputting `grad_attn_scores` in bfloat16. Work in float32 internally for accuracy. Each program instance handles one (bs, head, sq_tile) slice.

2. **Fuse the dV computation with GQA group reduction:** Write a second Triton kernel (or extend the first) that computes `dV = sum_over_groups(P_dropped^T @ dO)` without materializing the [bs,80,skv,128] expanded intermediate. Each program instance handles one (bs, kv_head, skv_tile) slice and accumulates across the 10 GQA groups in a loop. This avoids the expensive reshape+sum on a large tensor.

3. **Dispatch heuristic:** For small workloads (small bs×sq), use a grid that maximizes parallelism across heads; for large workloads, prioritize tile sizes that achieve good arithmetic intensity.

The worker should implement these two Triton kernels and use PyTorch BMMs as fallback only if correctness issues arise. Use bfloat16 loads with float32 accumulation throughout.

---

## Iteration 4

## STATE
The run has two confirmed keeps: baseline PyTorch at 3428 μs and the two-Triton-kernel approach at 2405 μs (~30% improvement). Two refinement attempts (Exp #3 and #4) both regressed, suggesting the current Triton kernels are sensitive to changes and the two-pass dS approach and a PyTorch-precomputed row_sum hybrid are both dead ends. The SOL target geomean is roughly 270 μs, so we're still ~9× off. The core bottleneck is almost certainly that both the dS kernel (two passes over V and P) and the dV kernel (nested loop over 10 groups × sq blocks) are heavily memory-bandwidth limited and not leveraging cuBLAS/tensor cores efficiently.

## RATIONALE
The Triton kernels do fused work but pay a heavy price: the dS kernel reads V and P twice (two passes for row_sum), and the dV kernel has a doubly-nested loop over groups and sq blocks that prevents efficient GEMM utilization. The reference algorithm is fundamentally two batched matrix multiplications plus pointwise ops. The fastest path is to exploit cuBLAS's highly optimized batched GEMM directly — `torch.matmul` with proper tensor reshaping is already close to cuBLAS, but we can do better by staging the computation to avoid redundant work and overlapping operations. Specifically: (1) transpose dO once contiguously, (2) compute dP̃ = dO @ V_expanded^T via a single batched GEMM (expanding V via `expand` to avoid copying), (3) fuse the dropout + softmax backward into a single fast Triton pointwise kernel, (4) compute dV_exp = P_drop^T @ dO via batched GEMM then reduce with `reshape+sum`. The key insight is that cuBLAS batched GEMM on a B200 will dramatically outperform hand-written Triton for the large matrix multiplications, while a simple fused pointwise kernel handles the softmax backward with minimal overhead.

## PROPOSAL
Replace the two large Triton kernels with **cuBLAS-backed batched GEMMs for the matrix multiplications** plus a **single fused Triton pointwise kernel** for the softmax backward step:

1. **dO preparation**: Transpose grad_attn_output from `[bs, sq, 80, d]` → `[bs, 80, sq, d]` as float32, contiguous.

2. **V expansion**: Use `expand` (no copy) on value_states to get `[bs, 80, skv, d]` — this is zero-copy and cuBLAS can handle the non-contiguous stride pattern, or make it contiguous with a single `.contiguous()` call (benchmark both).

3. **dP̃ = dO @ V^T**: Single `torch.matmul(dO_f32, V_exp_f32.transpose(-2,-1))` — this is a `[bs*80, sq, d] × [bs*80, d, skv]` batched GEMM, fully utilizing tensor cores.

4. **Fused pointwise Triton kernel**: Takes dP̃, dropout_mask, P (attn_weights) as inputs; computes `dP = dP̃ * mask * inv_scale`, then `row_sum = sum(dP*P, dim=-1)`, then `dS = P*(dP - row_sum)` — all in one pass over the `[bs, 80, sq, skv]` tensor. This eliminates the two-pass issue entirely since we never store intermediate dP.

5. **dV via GEMM + reduce**: `torch.matmul(P_drop^T, dO)` as a `[bs*80, skv, sq] × [bs*80, sq, d]` batched GEMM giving `[bs, 80, skv, d]`, then `reshape(bs, 8, 10, skv, d).sum(dim=2)` for GQA reduction.

The worker should choose tile sizes, warp counts, and staging for the pointwise kernel. The strategic bet is that offloading the two large GEMMs to cuBLAS and minimizing the Triton work to a single-pass pointwise fusion will be the fastest achievable approach.

---

## Iteration 7

## STATE

The run has converged on a hybrid approach (cuBLAS BMMs + Triton pointwise) as the best structure, currently at 2021.69 μs vs a SOL of ~113 μs geomean. The two Triton-kernel-for-BMM approaches (#2, #3) were slower than cuBLAS, and all pointwise kernel variations (#6, #7) are within noise of each other (~2022–2047 μs), suggesting the pointwise kernel is not the bottleneck. The dominant cost is almost certainly the two large BMMs (dP = dO @ V^T and dV = P_drop^T @ dO) plus the V_exp contiguous copy.

## RATIONALE

The biggest waste in the current implementation is expanding V from [bs,8,skv,128] to [bs,80,skv,128] with a `.contiguous()` call — this is a 10× memory amplification for the V tensor, purely to feed cuBLAS. Similarly, P_drop is cast to float32 for the dV BMM, doubling its memory footprint. The key insight: the GQA structure means we only have 8 unique KV heads, so BMM #1 (dP = dO @ V^T) should be computed with [bs,8,sq*10,128] × [bs,8,128,skv] instead of the expanded [bs,80,sq,skv] form. This avoids the V expansion entirely and reduces memory traffic. The dV reduction is naturally handled by grouping dO and P_drop by KV head. Additionally, keeping both BMMs in bfloat16 (which the B200 handles efficiently) instead of converting to float32 would halve memory bandwidth for those operations.

## PROPOSAL

Restructure the BMM operations to exploit GQA natively, eliminating the V expansion:

**For dP (BMM #1):** Reshape dO from [bs,80,sq,128] → [bs,8,10,sq,128] → [bs,8,10*sq,128], then compute batched matmul against V [bs,8,128,skv] to get [bs,8,10*sq,skv], then reshape to [bs,80,sq,skv]. This avoids the 10× V copy entirely.

**For dV (BMM #2):** Reshape attn_weights_dropped from [bs,80,sq,skv] → [bs,8,10,sq,skv] → [bs,8,10*sq,skv] (transpose → [bs,8,skv,10*sq]), and reshape dO similarly to [bs,8,10*sq,128], then compute [bs,8,skv,10*sq] × [bs,8,10*sq,128] → [bs,8,skv,128]. This directly produces the GQA-reduced dV without a separate sum.

Keep the tensors in bfloat16 for the BMMs (torch.matmul handles bf16 on B200 with tensor cores) rather than upcasting to float32 before the matrix multiplications. Only the softmax backward accumulation needs float32 precision. This should significantly reduce memory bandwidth and eliminate the large intermediate expansions.

---

## Iteration 10

## STATE

The run has converged to a stable approach: GQA-native BMMs (no V expansion) + Triton softmax backward kernel. The best geomean is 786.71 μs, down from 3428 μs at start. Experiments #9 and #10 are within ~2 μs of each other (788 vs 787), indicating the current architecture has hit a local plateau — the slope has clearly flattened. The SOL target is ~110 μs (geomean), so we're still ~7× above SOL. The bottleneck is almost certainly the two large BMMs and the intermediate dP_raw float32 tensor allocation/movement.

## RATIONALE

The current approach materializes a large float32 intermediate `dP_raw` of shape `[bs, 80, sq, skv]` — for large cases (bs=32, sq=512, skv=512) this is 32×80×512×512×4 = 3.2 GB of data written then re-read by the Triton kernel. The softmax backward kernel also reads 3 inputs (dP_raw float32, P bfloat16, mask bool) and writes 1 output — total ~5 passes over a large tensor. The key opportunity is to **fuse the dP_raw computation (BMM #1) directly into the softmax backward Triton kernel**, eliminating the float32 intermediate entirely. This means writing a Triton kernel that: (1) loads tiles of dO and V, computes dP tiles via dot products, applies dropout mask, and accumulates the row-sum reduction — all without materializing dP to HBM. This is the classic Flash-Attention style fused backward.

The second opportunity is that the current softmax kernel uses one program per row (bs×80×sq programs), which for small cases (bs=4, sq=256) is only 81,920 programs — potentially under-utilizing the GPU. A tiled approach processing multiple rows per program could help small cases.

## PROPOSAL

**Fuse BMM #1 (dO @ V^T → dP) directly into the softmax backward Triton kernel**, eliminating the large float32 `dP_raw` intermediate tensor entirely.

The strategy:
- Write a single Triton kernel that takes dO `[bs,80,sq,128]` and V `[bs,8,skv,128]` as inputs, and outputs dS `[bs,80,sq,skv]` directly
- Each program handles one (bs, head, sq_row): it loads the dO row (128 elements), then sweeps over skv blocks — for each block, loads the V tile, computes the dot product to get dP elements, applies dropout mask scaling, loads P tile, accumulates into row_sum — all in registers
- After the first sweep for row_sum, do a second sweep to write dS (or use a single-pass if skv fits in SRAM)
- This eliminates one full read+write of `[bs,80,sq,skv]` float32 (~4× the bfloat16 size), which for large cases is the dominant memory traffic
- Keep BMM #2 (dV computation) as a PyTorch matmul using the GQA-native grouped reshape — it's already efficient and doesn't have the intermediate materialization problem (it reads bfloat16 directly)

The key correctness constraint: V must be accessed in GQA mode (kv_head = head // 10), so the kernel uses `kv_idx = h_idx // n_groups` to index into V.

---

## Iteration 13

## STATE

The run has converged on a strong approach: GQA-native BMMs (reshaping dO/P_drop to [bs,8,10*sq,128] to avoid 10x V expansion) + a per-row Triton softmax backward kernel. The best result is 786.71 μs (geomean), achieved at iteration #10. Iterations #9 and #10 are within ~1 μs of each other (noise floor), indicating the current implementation is mature. The SOL target is ~156 μs geomean, so we're roughly 5× away. Recent Triton-only experiments (#11, #12, #13) have all failed badly, suggesting the BMM bottleneck is the dominant cost and pointwise kernel rewrites don't help much.

## RATIONALE

The current approach spends most time on two BMMs: `dO_grouped @ V.T` (for dP_raw) and `P_drop_grouped.T @ dO_grouped` (for dV). These are [bs,8,10*sq,skv] shaped operations running through cuBLAS. The intermediate float32 conversion of dP_raw also creates a large tensor that feeds the Triton softmax kernel. The key opportunity is to **fuse the BMM for dP_raw with the softmax backward** — eliminating the float32 intermediate materialization of size [bs,80,sq,skv]. A Triton kernel that computes dO @ V.T in tiles and simultaneously does the two-pass softmax backward (accumulate row_sum, then write dS) would eliminate one large read+write of [bs,80,sq,skv] float32 data. This is essentially the Flash-Attention style "tile over skv, accumulate statistics" pattern, and it should give a significant speedup especially for large seq_kv cases (cases 3, 6, 8, 16 which dominate the geomean).

## PROPOSAL

Implement a new **fused Triton kernel for dS** that replaces both the `torch.matmul(dO_grouped, V.T)` BMM and the subsequent softmax backward Triton kernel with a single kernel. The kernel should:

1. Assign one program per `(bs, head, sq_row)` or small groups of sq-rows
2. Load the dO row (128 elements, head_dim dimension) once into registers/SRAM
3. Tile over the skv dimension: for each skv tile, load V tile and compute `dP_tile = dot(dO_row, V_tile.T)`, load dropout mask and P tile, accumulate `row_sum += sum(dP_tile * dropout(dP_tile) * P_tile)` — all in the first pass
4. Second pass over skv: recompute dP_tile (or reload), apply dropout, compute `dS = P * (dP - row_sum)`, store to output
5. This avoids materializing the full [bs,80,sq,skv] float32 intermediate entirely

Keep the existing GQA-native `P_drop_grouped.T @ dO_grouped` BMM for dV (it's already efficient). The softmax_bwd fused-with-gemm kernel is the high-value change. The key insight is that the dO row (128 floats) fits in registers, so we only need to stream V, P, and mask tiles — not materialize dP_raw at all.

---

## Iteration 16

## STATE

The run has converged on a strong baseline: GQA-native cuBLAS BMMs (no V expansion) + a per-row Triton softmax backward kernel, sitting at ~787 μs geomean. Experiments #8→#9→#10 show a flat slope (~851→788→787 μs), indicating this approach has matured. All attempts to replace the cuBLAS BMMs with custom Triton GEMM kernels have been decisively slower (2000–6000 μs range). The SOL target is ~114 μs geomean, so we're roughly 7× away.

## RATIONALE

The current approach is memory-bandwidth limited on the softmax backward pass (reads dP_raw float32 + P bfloat16 + dropout_mask bool, writes dS bfloat16 — all large tensors). The key insight is that `dP_raw` is an intermediate float32 tensor that doesn't need to exist at all: the BMM `dO_grouped @ V^T` and the softmax backward can be fused if we move to a tiled Triton kernel that computes the GEMM and the softmax reduction in the same kernel. However, all attempts at that fusion have failed. 

A different angle: the current approach does `attn_weights_dropped.reshape(...)` which should be a free view, but `attn_weights_dropped` may not be contiguous in the [bs,80,sq,skv] layout needed for this reshape. More importantly, the `dO_5d.contiguous()` call materializes a large tensor copy. We should investigate whether the `dO_grouped` materialization can be avoided by feeding `grad_attn_output` directly with proper stride manipulation into the BMM, potentially using `as_strided` or a different permutation strategy. Additionally, the float32 intermediate `dP_raw` (4 bytes/element) costs bandwidth vs bfloat16 (2 bytes) — we could keep the BMM in bfloat16 and feed bfloat16 directly to the Triton softmax kernel, saving memory bandwidth on the largest intermediate.

The most promising unexplored axis: eliminate the float32 `dP_raw` intermediate entirely. Keep the BMM output as bfloat16 and modify the softmax backward kernel to accept bfloat16 `dP_raw` instead of float32. This halves the read bandwidth for that tensor. Combined with the already-contiguous `P` tensor also being bfloat16, both reads become half the size.

## PROPOSAL

Modify the current best approach (#10) to eliminate the float32 intermediate `dP_raw`:

1. Keep `torch.matmul(dO_grouped, V.transpose(-2,-1))` **in bfloat16** (do not cast to float32 after the BMM). The result `dP_raw_grouped` stays as bfloat16.
2. Reshape to `[bs, 80, sq, skv]` bfloat16 — this is now the input to the softmax kernel.
3. Update the `softmax_bwd_kernel_single` and `softmax_bwd_kernel` to accept bfloat16 `dP_raw` (load as bfloat16, convert to float32 inside the kernel for accumulation). This halves the memory bandwidth for the dP_raw reads.
4. Keep all other aspects of the current approach identical: GQA-native grouped BMMs, per-row softmax kernels, same dispatch logic based on seq_kv size.

The key win is reducing the working set for the softmax backward pass from (float32 dP + bf16 P + bool mask → bf16 dS) to (bf16 dP + bf16 P + bool mask → bf16 dS), saving ~25% of the total memory traffic for that phase. For the large benchmark cases (case 3, 6, 8, 16) which dominate the geomean, this could yield a meaningful speedup.

---

## Iteration 19

## STATE

The run has found a solid approach: GQA-native grouped BMMs (no V expansion) + bfloat16 dP_raw + Triton per-row softmax backward. Best is 466 μs vs SOL geomean ~120 μs — roughly 4× gap remaining. The bf16 dtype change in #18 was a major win (~40% speedup from #10 to #18). The recent #19 attempt to tune further was within noise (470 vs 466), suggesting the current approach is maturing. The large remaining gap to SOL implies there's still a structural bottleneck, not just tuning.

## RATIONALE

The current pipeline has three serial stages: (1) transpose+contiguous for dO, (2) BMM for dP_raw, (3) Triton softmax kernel. Stage (1) is a pure memory copy — materializing `[bs, 8, 10*sq, 128]` contiguous before the BMM. For large cases (e.g., case 6: bs=32, sq=691, case 8: bs=32, sq=512), this copy is expensive. The key question is: can the Triton softmax kernel be restructured to also absorb the BMM (dP computation), eliminating the intermediate dP_raw tensor entirely? A fused kernel that computes `dP_raw = dO_row @ V^T` and immediately applies the softmax backward pointwise would eliminate one full HBM round-trip for the large `[bs, 80, sq, skv]` tensor. Previous attempts at fused kernels crashed — but those tried to do everything with tl.dot inside Triton on a per-row basis with non-contiguous access. The correct approach is different: keep the BMMs in cuBLAS but eliminate the intermediate materialization by piping directly, OR restructure the Triton softmax kernel to process multiple sq-rows per program using `tl.dot` on properly contiguous tiles.

## PROPOSAL

Pursue a **multi-row Triton softmax-backward kernel** that processes a block of sq-rows together. Instead of one program per row (bs×80×sq programs), use a 2D grid where each program handles a tile of `BLOCK_SQ` rows simultaneously using `tl.dot` for the row_sum accumulation. The key correctness fix vs. the crashed #13: ensure the program correctly computes the per-row sum (not a cross-row sum) — `tl.dot` produces a `[BLOCK_SQ, BLOCK_SKV]` result, and the row sum should be `tl.sum(..., axis=1)` to get a `[BLOCK_SQ]` vector. This increases arithmetic intensity and SM utilization by amortizing indexing overhead across rows. Keep dP_raw as bfloat16 and the GQA-native grouped BMMs. The grid should be `(bs * n_heads, cdiv(seq_q, BLOCK_SQ))` so each program handles a contiguous block of rows for one (batch, head) pair, enabling coalesced memory access patterns.

---

## Iteration 22

## STATE

The run has converged on a solid architecture: GQA-native cuBLAS BMMs (no V expansion) + Triton softmax-backward kernel. Best geomean is 439.52 μs, down from 3428 μs baseline. The SOL target is ~120 μs (geomean), so we're still ~3.7× above SOL. The softmax kernel has matured through several variants (per-row, multi-row, single-pass) with diminishing returns (~7 μs improvement per iteration lately). The two BMMs (dP_raw and dV) plus the transpose/contiguous for dO_grouped are likely now the dominant cost, not the softmax kernel.

## RATIONALE

The current pipeline does: (1) permute+contiguous on grad_attn_output to make dO_grouped, (2) BMM for dP_raw, (3) Triton softmax backward, (4) BMM for dV. The `.contiguous()` call in Step 1 materializes a full copy of dO, which is expensive. More importantly, Steps 2 and 4 are sequential cuBLAS calls that could potentially overlap or be restructured. A key insight: `torch.matmul` with `.transpose(-2,-1)` forces cuBLAS to handle a non-contiguous input, adding overhead. Also, `attn_weights_dropped.reshape()` for Step 4 requires `attn_weights_dropped` to be contiguous in the right way — it currently is, but verifying this is zero-copy. The biggest remaining opportunity is to fuse the softmax kernel with the dP_raw computation entirely, or to overlap the two BMMs via CUDA streams.

## PROPOSAL

Launch the two cuBLAS BMMs (dP_raw and dV) on **separate CUDA streams** to enable concurrent execution on the GPU, then synchronize before the softmax kernel. Specifically: create two non-default streams, issue `dO_grouped @ V^T` on stream 1 and `P_drop_grouped^T @ dO_grouped` on stream 2 simultaneously, then sync both streams before running the Triton softmax kernel (which depends on dP_raw). This exploits the fact that both BMMs are independent and the GPU likely has capacity for concurrent GEMM execution on a B200. Additionally, investigate whether the `.contiguous()` call for dO_grouped can be eliminated by passing the non-contiguous tensor directly to cuBLAS (which handles arbitrary strides), potentially saving a full tensor copy.

---

## Iteration 25

## STATE

The run has converged to 431.32 μs geomean (best #24), representing roughly 5.7× speedup over baseline and about 57% of the way to SOL (which targets ~756/SOL_geomean ≈ ~46 μs geomean). The last 7 experiments have ranged 431–470 μs, with the last 4 runs (#22–#25) all within 8 μs of each other — this is clearly within the noise floor (~5–20 μs for small cases). The approach has matured: GQA-native batched GEMMs + single-pass Triton softmax kernel + dual-stream concurrency. The slope has definitively flattened.

## RATIONALE

At 431 μs vs SOL ~46 μs, there's roughly a 9× gap remaining, which suggests the current approach is hitting a structural ceiling rather than a tuning ceiling. The SOL numbers imply the benchmark cases can run much faster — looking at cases like #7 (bs=8, sq=128, skv=128, SOL=11.9 μs vs current ~few ms share) and #1 (bs=4, sq=256, skv=256, SOL=20.1 μs), the SOL is dominated by small cases where cuBLAS batched GEMM overhead (kernel launch, stream synchronization) is the bottleneck. The two BMMs (one on each stream) each require separate cuBLAS calls that carry significant per-call overhead for small problem sizes. For large cases like #6 (bs=32, sq=691, skv=773) and #16 (bs=1, sq=4096, skv=4096), the compute dominates. The fundamental issue is that the current approach materializes `dP_raw` as an intermediate tensor `[bs, 80, sq, skv]` which is then read again by the softmax kernel — for small cases this is wasted bandwidth. A Flash-attention-style fused kernel that computes `dP_raw` (via a register-level matmul tile) and immediately applies the softmax backward without writing to global memory could eliminate this bottleneck entirely.

## PROPOSAL

Implement a **fully-fused Triton kernel** for the `dP_raw → softmax_bwd → dS` path that eliminates the intermediate global memory write/read of `dP_raw`. Each Triton program should compute a tile of `dO @ V^T` (the attention score gradient) using `tl.dot`, immediately compute the softmax backward (accumulating the row-sum in registers), and write only the final `dS` output. This fusion eliminates one full read-write of the `[bs, 80, sq, skv]` tensor which for large cases is substantial bandwidth, and for small cases eliminates the GEMM kernel launch overhead. The grid should be over `(bs * n_kv_heads, seq_q_tiles)` to leverage the GQA structure, with each program iterating over `seq_kv` tiles. The `dV` GEMM can continue using cuBLAS on a side stream. This is a structurally different approach with genuine headroom since it attacks the memory bandwidth bottleneck at the algorithmic level.

