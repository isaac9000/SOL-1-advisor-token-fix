# Advisor Proposals

---

## Iteration 1

## STATE

Only the pure-PyTorch baseline exists at 3430.34 μs geomean. The SOL target is ~226 μs (score ≈9.3), meaning roughly a 15× improvement is needed. The baseline does all compute in float32, uses `expand+reshape` for GQA, and issues multiple separate kernels (transpose, matmul, elementwise, sum). No custom CUDA has been attempted yet. This is a cold start — every optimization axis is open.

## RATIONALE

The baseline has several obvious inefficiencies: (1) it casts to float32 before both matmuls, creating 2× memory traffic; (2) it uses `attn_weights_dropped.transpose(-2,-1)` as a separate kernel before bmm #2; (3) the GQA expansion creates a full `[bs,80,skv,128]` intermediate; (4) there are ~6–7 separate kernel dispatches with poor fusion. The highest-leverage first move is to replace the two heavyweight matmuls and surrounding pointwise ops with cuBLAS/cuBLASLt batched GEMMs in bfloat16 (avoiding the float32 cast overhead), fuse the elementwise dropout-backward and softmax-backward into a single Triton kernel, and handle the GQA sum-reduction without materializing the full expanded dV tensor. A well-fused Triton approach or torch.compile with the right strategy can collapse 6+ dispatches into 2–3, dramatically cutting latency.

## PROPOSAL

**Use `torch.compile` with `mode="max-autotune"` plus targeted manual fusion as the first serious attempt.** Specifically:

1. Wrap the entire `custom_kernel` body in a `torch.compile`-decorated function (compile once, cache). Keep bfloat16 throughout — avoid the float32 cast entirely by using bf16 matmuls. PyTorch on B200 supports bf16 bmm natively and cuBLAS will pick fast bf16 GEMM paths.

2. For the GQA dV reduction: instead of expanding value_states to 80 heads, pass `attn_weights_dropped` reshaped as `[bs, 8, 10, sq, skv]` and contract with `dO` reshaped as `[bs, 8, 10, sq, d]` using a single batched matmul, producing `[bs, 8, 10, skv, d]`, then sum over dim=2. This avoids the large `[bs,80,skv,128]` intermediate entirely.

3. Keep the softmax-backward and dropout-backward as simple elementwise ops — `torch.compile` will fuse these automatically via inductor.

4. Use `torch.compile` with `fullgraph=True` to maximize fusion opportunity and reduce Python overhead.

The rationale: `torch.compile` on B200 (sm_100) with inductor will auto-tune tiling for both bmms and fuse the cheap elementwise ops. This is the highest-confidence, lowest-risk first step that should get us well under 1000 μs. If it lands near 400–600 μs, we then layer in Triton kernels for the fused elementwise middle section and hand-tuned GEMMs.

---

## Iteration 2

## STATE

Two experiments so far. The `torch.compile` approach (exp #2) delivered a 3.85× speedup over the baseline, landing at 891 μs geomean — still roughly 4× above the SOL target of ~226 μs. The implementation is clean and fully fused via inductor. The main remaining bottlenecks are: (1) the two large batched GEMMs (`dO @ vs_exp^T` and `awd_r^T @ dO_r`) which still operate on `[bs, 80, sq, skv]`-sized data, and (2) `torch.compile` likely isn't generating cuBLASLt-optimized GEMM calls with the best tile shapes for B200. The approach is still maturing — there's likely more to extract from the compile path, and Triton custom kernels haven't been tried yet.

## RATIONALE

The `torch.compile` result is a solid foundation but the two batched matmuls dominate runtime and `max-autotune` may not be selecting the globally optimal GEMM algorithm on B200. The key insight is that bmm #1 (`dO @ V^T`, `[bs,80,sq,skv]`) and bmm #2 (`awd^T @ dO`, same shape) are the bottleneck. On B200, `torch.nn.functional` or explicit cuBLASLt via `torch._C._nn.linear`-style calls may outperform inductor-generated code. More importantly, we haven't yet tried **Flash-Attention-style tiled Triton kernels** that compute both matmuls and fuse the elementwise middle section (dropout-backward + softmax-backward) in a single pass, avoiding the round-trip to HBM for the large `[bs,80,sq,skv]` intermediate tensor. Flash-attention backward kernels are the state-of-the-art precisely because they keep `dP` in SRAM and never write it to HBM.

## PROPOSAL

**Implement a custom Triton kernel that fuses the entire middle computation — the two GEMMs plus the elementwise section — in a single tiled pass.**

The key architectural idea: tile over `(batch, head, query-block)` and for each tile iterate over key-value blocks. In each tile:
1. Load a tile of `dO` (`[BLOCK_Q, HEAD_DIM]`) into SRAM.
2. Iterate over `skv` blocks: load `V` tile, compute `dP_tile = dO_tile @ V_tile^T`, apply dropout mask, apply softmax-backward (requires `P` tile and the row-sum accumulator `Di = sum(dP*P)`), accumulate `dV_tile += P_dropped_tile^T @ dO_tile`.
3. Write out `dS` tile and the accumulated `dV` contribution.

This is essentially the FlashAttention-2 backward pass adapted for GQA:
- The `dV` accumulation handles the 10-group reduction naturally (each of the 80 heads writes to its corresponding kv-head slot, and atomic-adds or we tile the sum explicitly).
- The intermediate `[bs,80,sq,skv]` `dP` tensor is **never materialized** in HBM — it lives entirely in registers/SRAM.
- For the GQA dV reduction: since groups 0–9 map to kv-head 0, groups 10–19 to kv-head 1, etc., we can handle this with a `atomic_add` into `grad_value_states` or by structuring the grid so the 10 group-heads of each kv-head run cooperatively.

The worker should implement this as a Triton kernel with:
- Grid over `(bs * n_kv_heads * n_groups, ceil(sq/BLOCK_Q))` — one program per attention head tile.
- For `dV`: use `tl.atomic_add` into the kv-head slot (8 slots per bs) since 10 programs write to each kv-head, or restructure to process all 10 groups sequentially within one program instance.
- Keep all intermediate accumulators in float32 for numerical stability, cast outputs to bf16 on write.
- The `dropout_mask` is bool so loading it is cheap (1 bit per element with `tl.load`).

This single-kernel approach should get close to SOL by eliminating HBM round-trips for the large attention-weight-sized intermediates.

---

## Iteration 3

## STATE

Three experiments: baseline at 3430 μs, `torch.compile` at 891 μs (best), and a Triton attempt that crashed back to ~3453 μs (discarded). The Triton kernel attempt regressed severely — likely due to correctness or launch issues preventing the kernel from running properly, forcing fallback to a slow path. The `torch.compile` approach at 891 μs is currently the best and still has room to improve — the SOL gap is still ~4×. The Triton approach was a first attempt and should not be abandoned, but clearly the initial implementation had significant issues.

## RATIONALE

The Triton attempt failed badly (3453 μs ≈ baseline), which usually indicates a kernel bug causing a correctness fallback, a launch configuration error, or the kernel simply not covering all cases. The core idea — fusing the two GEMMs with the elementwise middle section to avoid materializing the large `[bs,80,sq,skv]` intermediate — remains the right strategy; flash-attention backward is the gold standard for exactly this computation. Rather than abandoning Triton, we should take a more conservative, step-by-step approach: first get a correct Triton kernel for just the elementwise middle section (dropout-backward + softmax-backward), then layer in the GEMM fusion. Meanwhile, there's a simpler optimization on the `torch.compile` path that we haven't tried: **restructuring the computation to use `einsum` or explicit batch dimensions that allow cuBLASLt to pick its best algorithm**, or using `torch.compile` with `dynamic=False` to allow static shape specialization.

## PROPOSAL

**Take a two-pronged approach: (A) further optimize the `torch.compile` path, and (B) write a correct, minimal Triton kernel for just the elementwise middle section.**

For (A), modify the compile-based solution to:
1. Avoid the `.expand(...).reshape(...)` for the GQA expansion — instead use `torch.repeat_interleave` or restructure to do `dO` reshaped to `[bs*8, 10, sq, d]` and `value_states` as `[bs*8, skv, d]` so we get a clean 3D batched GEMM that cuBLASLt can optimize well (batch dim = bs*8, not bs*80).
2. For bmm #1: compute `dP` as `[bs, 8, 10, sq, skv]` by doing `dO_r @ value_states^T` where `dO_r = [bs, 8, 10, sq, d]` and `V = [bs, 8, 1, skv, d]` broadcast — this is a single batched GEMM with bs×8×10 batch slots but only bs×8 unique V tiles.
3. Add `torch.compile(dynamic=False)` to get static shape specialization per workload.

For (B), write a standalone Triton kernel that **only** handles the elementwise fusion (dropout-backward + softmax-backward), taking `dP_dropped`, `dropout_mask`, `attn_weights` as inputs and producing `dS` — this is trivially correct and avoids GEMM complexity. Wrap the two `torch.matmul` calls outside in plain PyTorch (still compiled), and insert the Triton elementwise kernel in the middle. This removes the large `[bs,80,sq,skv]` intermediate dP tensor from HBM by fusing it with the matmul outputs — but if the matmuls themselves are the bottleneck, focus on (A) first.

**Priority: push the `torch.compile` approach further first.** Specifically, try using `torch.compile` with `mode="max-autotune-no-cudagraphs"` and also try manually invoking `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` to ensure full precision bf16 GEMMs. The key win available is reducing the number of unique GEMMs from 2×80-head to 2×8-head by restructuring the math to exploit GQA structure at the GEMM level.

---

## Iteration 4

## STATE

Four experiments: baseline at 3430 μs, best at 891 μs (exp #2 `torch.compile`), two discards at 3453 μs (Triton attempt) and 2290 μs (a torch.compile variant that actually regressed significantly from exp #2). The Triton direction has failed twice. The `torch.compile` path at 891 μs is solid and still the champion. The SOL is ~226 μs, so we need another ~4× from here. The two discarded attempts suggest the worker's Triton implementation had bugs or launch issues, and exp #4's regression from exp #2 suggests the restructuring broke something the original compile was handling well.

## RATIONALE

The pattern is clear: Triton from scratch is brittle given the failures, and naive restructuring of the torch.compile path can regress. The best path forward is to exploit what's already working in exp #2 and push it further with **two specific improvements**: (1) use CUDA graphs to eliminate kernel launch overhead, which matters especially for the many small workloads in the benchmark (cases like bs=4,sq=256,skv=256 and similar), and (2) exploit `torch.nn.attention.sdpa_kernel` or `scaled_dot_product_attention` infrastructure that B200 already has highly tuned. More importantly, the fundamental bottleneck is that we're doing two large batched GEMMs with batch_size=80 (all 80 heads), but with GQA we only have 8 unique V matrices. The key restructuring is: for bmm #1, instead of `[bs*80, sq, skv]` batched GEMM, do `[bs*8, 10*sq, skv]` — stacking the 10 groups' query rows together so cuBLAS sees larger, more efficient GEMM tiles and fewer launches.

## PROPOSAL

**Build on exp #2's `torch.compile` approach with a smarter GQA-aware matmul restructuring that reduces the effective batch count from 80 to 8.**

The core insight: with 80 attention heads grouped into 8 KV heads (10 groups each), we can reshape the matmuls to work at the 8-KV-head level rather than the 80-head level:

**For bmm #1 (dP = dO @ V^T):**
- Reshape `dO` from `[bs, 80, sq, d]` to `[bs*8, 10*sq, d]` (stack the 10 group queries together)
- `value_states` is already `[bs, 8, skv, d]` → view as `[bs*8, skv, d]`
- Compute `[bs*8, 10*sq, d] @ [bs*8, d, skv]` → `[bs*8, 10*sq, skv]`
- Reshape back to `[bs, 8, 10, sq, skv]` → `[bs, 80, sq, skv]`
- This is one GEMM with 8× fewer batch slots and 10× larger M dimension — much better GPU utilization

**For bmm #2 (dV = awd^T @ dO):**
- Reshape `attn_weights_dropped` from `[bs, 80, sq, skv]` to `[bs*8, 10*sq, skv]`
- Reshape `dO` from `[bs, 80, sq, d]` to `[bs*8, 10*sq, d]`
- Compute `[bs*8, skv, 10*sq] @ [bs*8, 10*sq, d]` → `[bs*8, skv, d]` directly (sum over groups already done!)
- This replaces the `[bs*8*10, skv, d]` matmul + sum-reduction with a single `[bs*8, skv, d]` matmul — mathematically identical since `A^T @ B` over the stacked rows is the same as summing the per-group `A_i^T @ B_i`

Wrap this in `torch.compile(mode="max-autotune", fullgraph=True, dynamic=False)`. The key advantage: both GEMMs now have batch=bs*8 (not bs*80), and M=10*sq (not sq), so they hit cuBLAS's efficiency sweet spot with larger, squarer tiles. This should reduce latency by ~2× or more for compute-bound cases, and reduce kernel launch overhead for all cases.

---

## Iteration 5

## STATE

Five experiments, best still at 891 μs (exp #2). Three consecutive attempts to improve on exp #2 have all regressed badly (2290, 2215, 3453 μs). The pattern is clear: the workers are inadvertently breaking what makes exp #2 work — likely by touching the reshape/contiguity of tensors in ways that prevent `torch.compile` from fusing efficiently, or by introducing non-contiguous tensor operations that stall cuBLAS. Exp #2's exact code is the gold standard right now. We are ~4× above SOL and need a fundamentally different approach.

## RATIONALE

The repeated regressions from ~891 μs suggest that every attempt to "improve" the torch.compile path by restructuring the math has introduced contiguity issues or broken the fusion that inductor already handles well in exp #2. We should **stop modifying the torch.compile path** and instead try a completely orthogonal approach: **using cuBLAS directly through `torch._C._VariableFunctions` or `torch.ops.aten` with explicit contiguous tensors**, or better yet, leverage **Flash Attention's existing backward pass** via `torch.nn.functional.scaled_dot_product_attention` autograd. The most promising unexplored avenue is using the **`flash_attn` library** which has a highly optimized GQA backward pass already written for exactly this architecture, or `xformers`. If those aren't available, we should try **`torch.backends.cuda.enable_flash_sdp`** and related backends. Alternatively, the simplest high-leverage move that hasn't been tried: **CUDA graphs** wrapping exp #2's exact code.

## PROPOSAL

**Try two things, in order of confidence:**

**Option A (highest confidence): Add CUDA graph replay on top of exp #2's exact compiled implementation.**

Wrap the compiled call in a CUDA graph that captures the computation for each unique input shape. The benchmark has 16 fixed workloads — CUDA graphs eliminate all Python dispatch overhead and kernel launch latency, which is especially impactful for the many small cases (bs=4/8, sq=128-256). Use `torch.cuda.CUDAGraph` with a shape-keyed cache: on first call for a given `(bs, sq, skv)` shape, run a warmup and capture the graph; on subsequent calls, replay with `copy_` to update inputs. This can deliver 20-50% improvement on small cases with zero algorithmic change.

**Option B (medium confidence): Try `flash_attn` library's backward pass directly.**

Import `flash_attn` (available on B200 systems) and use `flash_attn_func` or the lower-level `flash_attn_cuda` ops that expose the backward pass directly. Flash Attention's backward is specifically designed to handle the GQA case and avoids materializing the large attention matrix. This is the closest to the SOL computation path.

**Recommended: implement Option A first** — it's a pure wrapping change with no risk of correctness regression since the underlying computation is identical to exp #2. The CUDA graph cache should be keyed on `(bs, sq, skv)` shape tuple. Use `torch.cuda.make_graphed_callables` which handles the warmup/capture automatically and integrates cleanly with `torch.compile`.

---

## Iteration 6

## STATE

Six experiments; best still at 891 μs (exp #2). We've had 3 discards and 1 crash. The CUDA graph approach crashed on correctness — likely because CUDA graph replay requires tensors to be written into the same pre-allocated buffers (in-place via `copy_`), and the graph capture/replay got the input tensor aliasing wrong. The torch.compile path is our only successful improvement so far. The SOL gap remains ~4×, and we've used 6 of 25 iterations with no further progress.

## RATIONALE

The failures share a common pattern: every attempt to modify or wrap exp #2 has broken it. The CUDA graph crash was almost certainly a correctness issue with how input tensors were copied into static buffers — a known gotcha where the copied tensors must be exactly the static buffers used during capture, not new allocations. The restructuring attempts (exps 4, 5) broke contiguity guarantees that `torch.compile` relied on. 

We need to step back and think about what's actually limiting exp #2. The ~891 μs geomean across 16 cases means the large cases (e.g., case 6: bs=32, sq=691, skv=773 at ~9273 μs baseline) dominate the geomean. For these large cases, the bottleneck is purely the two batched matmuls doing `[bs*80, sq, skv]`-sized work. The key insight we keep missing: **the `expand(...).reshape(...)` in exp #2 for `vs_exp` creates a non-contiguous tensor** — `expand` doesn't allocate new memory, so `vs_exp` has stride 0 on the groups dimension. When `torch.matmul` hits this, cuBLAS may be handling it suboptimally or falling through to a slow path. Making `vs_exp` contiguous (`.contiguous()`) before the matmul could unlock the fast cuBLAS path.

## PROPOSAL

**Make a minimal, surgical modification to exp #2: ensure all inputs to both matmuls are contiguous, and use `torch.einsum` for the dV computation to let cuBLAS pick the optimal path.**

Specifically, the worker should take exp #2's exact code and make only these targeted changes:

1. **Before bmm #1**: call `.contiguous()` on `vs_exp` after the expand+reshape, so cuBLAS sees a properly strided tensor. The expand creates stride-0 in the groups dim which may cause cuBLAS to use a slow fallback.

2. **For dO**: after `grad_attn_output.transpose(1, 2)`, call `.contiguous()` — the transpose creates a non-contiguous view which may force cuBLAS to copy internally.

3. **For dV computation**: instead of the reshape + matmul + sum approach, use `torch.einsum('bghqk,bghqd->bgkd', awd_r, dO_r)` which directly expresses the grouped contraction and lets `torch.compile` optimize it as a single fused operation.

4. **Keep `torch.compile(mode="max-autotune", fullgraph=True)` unchanged** — do not change the compile settings.

The `.contiguous()` calls are the highest-confidence fix — non-contiguous inputs to batched matmuls are a well-known performance pitfall on CUDA, and exp #2's `expand().reshape()` path almost certainly produces non-contiguous memory layout. This single change may unlock significant speedup with zero risk of correctness regression since the math is identical.

---

## Iteration 7

## STATE

Seven experiments; best still at 891 μs (exp #2). Exp #7 tried `.contiguous()` calls and `einsum` and got 952 μs — slightly worse than exp #2, confirming that `torch.compile` was already handling the non-contiguous tensors optimally (inductor likely inserts its own contiguity handling). The compile path appears to have largely plateaued at ~891 μs. We've exhausted easy wins on the torch.compile axis. The SOL gap remains ~4×. We need to change strategy significantly.

## RATIONALE

The `torch.compile` ceiling appears to be around 891 μs — seven attempts have not broken through it. The SOL at ~226 μs implies the compute is achievable with a well-tuned custom kernel. Looking at the problem structure: the two matmuls are the dominant cost, and they involve `[bs, 80, sq, skv]`-sized attention matrices. For large cases like bs=32, sq=691, skv=773, the attention matrices alone are 32×80×691×773×2 bytes ≈ 2.7 GB of memory bandwidth just to read/write them once. The SOL suggests the best path operates at near-bandwidth efficiency.

The key realization: the Triton attempts (exp #3) failed badly (back to baseline ~3453 μs), which means the Triton kernel had a fundamental correctness or performance issue — likely wrong grid launch or a kernel that ran but produced wrong results and fell back. Rather than complex fused Triton kernels, we should try something more reliable: **use `flash_attn` library** which is available on modern GPU systems and has a highly tuned GQA backward pass. Alternatively, try `xformers.ops.memory_efficient_attention_backward`. These are production-quality implementations that hit near-SOL performance.

## PROPOSAL

**Try the `flash_attn` library's backward pass primitives, with a fallback to a carefully written simple Triton kernel for just the elementwise section.**

**Primary approach — use `flash_attn` directly:**

The `flash_attn` package provides low-level primitives including `flash_attn_cuda.bwd` or the higher-level `flash_attn_func` with autograd. The worker should:
1. Import `flash_attn` and check if it's available.
2. Use `flash_attn.flash_attn_interface._flash_attn_backward` or equivalent — this handles GQA natively when `num_heads != num_kv_heads`.
3. If `flash_attn` isn't available, fall back to exp #2's implementation.

The flash_attn backward signature typically takes `(dout, q, k, v, out, softmax_lse, dq, dk, dv, ...)`. We'd need to map our inputs: we have `attn_weights` (the softmax output P) and `attn_weights_dropped` (P after dropout), `value_states`, `grad_attn_output`, and `dropout_mask`. The flash_attn backward expects the LSE (log-sum-exp) from the forward pass, which we don't have directly — but we can reconstruct it from `P` via `lse = log(P.sum(-1))` approximately, or use the relationship `lse_i = log(sum_j exp(s_ij))`.

**Fallback approach — simple correct Triton elementwise kernel:**

Write a Triton kernel that does ONLY the pointwise middle section: takes `dP_dropped` (output of matmul #1), `dropout_mask`, `attn_weights` (P), and computes `dS` — the softmax backward + dropout backward. This is trivially correct (no matmul, no reduction across tiles — just elementwise ops with one row-reduction for the `sum(dP*P)` term). Keep the two `torch.matmul` calls in PyTorch compiled code. This splits the work: PyTorch handles the GEMMs, Triton handles the fused elementwise. The benefit is reducing HBM traffic for the large `[bs,80,sq,skv]` intermediate.

**Recommended order:** Try flash_attn first (if available, it will be ~SOL). If it's not available or has wrong interface, implement the simple Triton elementwise kernel to at least improve the middle section while keeping the PyTorch matmuls.

---

## Iteration 8

## STATE

Eight experiments; best still 891 μs (exp #2). The Triton elementwise middle-section kernel (exp #8) came in at 1267 μs — significantly worse than exp #2, meaning the overhead of the Triton kernel plus the unchanged matmuls outweighs any fusion benefit. The `torch.compile` path at 891 μs appears to be the stable ceiling for all incremental improvements attempted so far. We've burned 8 of 25 iterations and are stuck at 4× above SOL.

## RATIONALE

The key diagnostic: every modification to exp #2 either stays near 891 μs or regresses. This tells us `torch.compile` is already producing near-optimal fused code for the elementwise section, and the dominant bottleneck is the two large batched matmuls themselves — not kernel launch overhead or elementwise fusion. The matmuls are the wall.

Looking at the problem more carefully: for the large cases that dominate the geomean (e.g., case 6: bs=32, sq=691, skv=773), both matmuls are `[32*80, 691, 773]`-sized operations. The SOL for case 6 is 1142 μs vs our 891 μs on average... wait, actually our geomean is 891 μs but the SOL geomean is 226 μs, so we're definitely underperforming on all cases. The fundamental issue is that **we're using bfloat16 matmuls** but the intermediate dP tensor (`[bs,80,sq,skv]`) is being written to and read from HBM between the two matmuls, which is the bandwidth bottleneck.

The only way to break through is to avoid materializing that intermediate — i.e., we need a truly fused kernel. The Triton attempts have all failed, but they've been too complex. Let me rethink: what we need is actually two separate improvements handled differently for small vs large cases.

For **small cases** (the many bs=4-8, sq=128-256 cases): the bottleneck is kernel launch overhead and the fixed cost of dispatching 5+ kernels. CUDA graphs would help here.

For **large cases**: the bottleneck is bandwidth for the attention matrix intermediates.

## PROPOSAL

**Return to Triton but with a radically simpler, correct-first design: write a Triton kernel for only `dV`, the GQA-grouped reduction matmul, which is the computation that's most wasteful in exp #2.**

In exp #2, the dV computation does:
- `awd_r.transpose(-2,-1) @ dO_r` → `[bs, 8, 10, skv, d]` (creates a large intermediate)  
- `.sum(dim=2)` → `[bs, 8, skv, d]`

This materializes a `[bs, 8, 10, skv, 128]` tensor that's 10× larger than the final output. A Triton kernel can compute the sum directly without materializing it — each program handles one `(bs, kv_head, skv_tile)` and accumulates contributions from all 10 query groups in a loop, writing only the final summed result.

**Concretely:** Write a Triton kernel `_dv_kernel` that:
- Grid: `(bs * n_kv_heads, ceil(skv/BLOCK_KV), ceil(d/BLOCK_D))` 
- For each program: loop over the 10 groups and over all sq blocks, accumulating `awd[:, group, sq_block, kv_tile]^T @ dO[:, group, sq_block, d_tile]` into a register accumulator
- Write the accumulated sum to `grad_value_states`

Keep everything else (the dS computation) in the existing `torch.compile` path. This Triton kernel replaces only the dV matmul+sum, avoids the `[bs,8,10,skv,128]` intermediate, and should be straightforward to get correct since it's just a matmul reduction.

The worker should start with correctness (verify against the torch reference) before worrying about tile sizes. This is the most contained, highest-leverage Triton kernel to write — it has no softmax complexity, just batched GEMM with GQA sum-reduction.

