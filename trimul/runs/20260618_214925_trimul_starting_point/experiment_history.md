# Experiment History

Tracks every kernel attempt, its code, hypothesis, and result.

---

## Experiment #1 — 2026-06-18 21:49:42 UTC ✅ KEEP

**Hypothesis:** Baseline 'starting_point' — initial benchmark

**Result:** 11371.68 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #2 — 2026-06-18 21:50:53 UTC ✅ KEEP

**Hypothesis:** Complete rewrite of custom_kernel eliminating nn.Module construction. Stacks left_proj, right_proj, left_gate, right_gate, out_gate weights into a single [5*hidden_dim, dim] matrix and calls F.linear 

**Result:** 10295.46 μs

**Kernel code:**
```python
"""
Optimized TriMul submission — stateless functional kernel with fused projections and bmm einsum.
"""

import torch
import torch.nn.functional as F


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    bs, N, _, _ = input_tensor.shape

    # --- Step 1: LayerNorm on input ---
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])

    # --- Step 2: Fuse all 5 projections into one GEMM ---
    # Stack weights: [5*hidden_dim, dim]
    fused_weight = torch.cat([
        weights['left_proj.weight'],   # [hidden_dim, dim]
        weights['right_proj.weight'],  # [hidden_dim, dim]
        weights['left_gate.weight'],   # [hidden_dim, dim]
        weights['right_gate.weight'],  # [hidden_dim, dim]
        weights['out_gate.weight'],    # [hidden_dim, dim]
    ], dim=0)  # [5*hidden_dim, dim]

    # Flatten spatial dims: [bs*N*N, dim]
    x_flat = x.reshape(bs * N * N, dim)

    # Single GEMM: [bs*N*N, 5*hidden_dim]
    all_proj = F.linear(x_flat, fused_weight)

    # Split into individual projections (all raw, pre-activation)
    lp, rp, lg, rg, og = all_proj.split(hidden_dim, dim=-1)
    # Each: [bs*N*N, hidden_dim]

    # --- Step 3: Apply gates (sigmoid) and combine ---
    left = lp * lg.sigmoid()    # [bs*N*N, hidden_dim]
    right = rp * rg.sigmoid()   # [bs*N*N, hidden_dim]
    # out_gate will be applied later: og.sigmoid()

    # --- Step 4: Apply mask ---
    # mask: [bs, N, N] -> [bs*N*N, 1]
    mask_flat = mask.reshape(bs * N * N, 1)
    left = left * mask_flat    # [bs*N*N, hidden_dim]
    right = right * mask_flat  # [bs*N*N, hidden_dim]

    # --- Step 5: Batched matmul for the einsum ---
    # Reference: einsum "... i k d, ... j k d -> ... i j d"
    # left[b, i, k, d], right[b, j, k, d] -> out[b, i, j, d]
    # Reshape to [bs*hidden_dim, N, N] and use bmm
    left_4d = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # out[b*d, i, j] = sum_k left[b*d, i, k] * right[b*d, j, k]
    # = left_4d @ right_4d^T
    out = torch.bmm(left_4d, right_4d.transpose(-1, -2))  # [bs*hidden_dim, N, N]

    # Reshape back: [bs, hidden_dim, N, N] -> [bs, N, N, hidden_dim]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1)  # [bs, N, N, hidden_dim]

    # --- Step 6: to_out_norm ---
    out = F.layer_norm(out, [hidden_dim],
                       weight=weights['to_out_norm.weight'],
                       bias=weights['to_out_norm.bias'])

    # --- Step 7: Apply out_gate ---
    # og: [bs*N*N, hidden_dim] -> [bs, N, N, hidden_dim]
    out_gate = og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = out * out_gate

    # --- Step 8: Final linear projection ---
    out = F.linear(out, weights['to_out.weight'])

    return out

```

---

## Experiment #3 — 2026-06-18 21:51:53 UTC ✅ KEEP

**Hypothesis:** Two targeted changes combined into one edit: (1) Replaced the single fused torch.cat + F.linear with 5 separate F.linear calls to eliminate the per-call large tensor allocation; (2) Cast left_4d and r

**Result:** 9372.65 μs

**Kernel code:**
```python
"""
Optimized TriMul submission — stateless functional kernel with separate projections,
bf16 bmm for the dominant matmul, and optimized memory layout.
"""

import torch
import torch.nn.functional as F


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    bs, N, _, _ = input_tensor.shape

    # --- Step 1: LayerNorm on input ---
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])

    # --- Step 2: Five separate projections (avoid per-call torch.cat allocation) ---
    # Flatten spatial dims: [bs*N*N, dim]
    x_flat = x.reshape(bs * N * N, dim)

    lp = F.linear(x_flat, weights['left_proj.weight'])    # [bs*N*N, hidden_dim]
    rp = F.linear(x_flat, weights['right_proj.weight'])   # [bs*N*N, hidden_dim]
    lg = F.linear(x_flat, weights['left_gate.weight'])    # [bs*N*N, hidden_dim]
    rg = F.linear(x_flat, weights['right_gate.weight'])   # [bs*N*N, hidden_dim]
    og = F.linear(x_flat, weights['out_gate.weight'])     # [bs*N*N, hidden_dim]

    # --- Step 3: Apply gates (sigmoid) and combine ---
    left = lp * lg.sigmoid()    # [bs*N*N, hidden_dim]
    right = rp * rg.sigmoid()   # [bs*N*N, hidden_dim]

    # --- Step 4: Apply mask ---
    # mask: [bs, N, N] -> [bs*N*N, 1]
    mask_flat = mask.reshape(bs * N * N, 1)
    left = left * mask_flat    # [bs*N*N, hidden_dim]
    right = right * mask_flat  # [bs*N*N, hidden_dim]

    # --- Step 5: Batched matmul for the einsum in bfloat16 ---
    # Reference: einsum "... i k d, ... j k d -> ... i j d"
    # left[b, i, k, d], right[b, j, k, d] -> out[b, i, j, d]
    # Reshape to [bs*hidden_dim, N, N] for bmm
    left_4d = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # Cast to bfloat16 for ~2-4x throughput on the dominant matmul
    # out[b*d, i, j] = sum_k left[b*d, i, k] * right[b*d, j, k]
    out = torch.bmm(left_4d.to(torch.bfloat16), right_4d.to(torch.bfloat16).transpose(-1, -2))
    out = out.to(torch.float32)  # [bs*hidden_dim, N, N]

    # Reshape back: [bs, hidden_dim, N, N] -> [bs, N, N, hidden_dim]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()  # [bs, N, N, hidden_dim]

    # --- Step 6: to_out_norm ---
    out = F.layer_norm(out, [hidden_dim],
                       weight=weights['to_out_norm.weight'],
                       bias=weights['to_out_norm.bias'])

    # --- Step 7: Apply out_gate ---
    # og: [bs*N*N, hidden_dim] -> [bs, N, N, hidden_dim]
    out_gate = og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = out * out_gate

    # --- Step 8: Final linear projection ---
    out = F.linear(out, weights['to_out.weight'])

    return out

```

---

## Experiment #4 — 2026-06-18 21:53:06 UTC ✅ KEEP

**Hypothesis:** Extracted the core computation (5 linear projections, sigmoid gates, mask application, bf16 bmm, layer norm, out gate, final linear) into a standalone function `_trimul_core`, then compiled it with `t

**Result:** 6064.53 μs

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile'd functional kernel with bf16 bmm.
"""

import torch
import torch.nn.functional as F


def _trimul_core(x_flat, mask_flat, bs, N, hidden_dim,
                 w_lp, w_rp, w_lg, w_rg, w_og,
                 w_norm2, b_norm2, w_out):
    lp = F.linear(x_flat, w_lp)
    rp = F.linear(x_flat, w_rp)
    lg = F.linear(x_flat, w_lg)
    rg = F.linear(x_flat, w_rg)
    og = F.linear(x_flat, w_og)

    left = lp * lg.sigmoid() * mask_flat
    right = rp * rg.sigmoid() * mask_flat

    # Reshape for batched matmul: [bs*hidden_dim, N, N]
    left_4d = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # bf16 bmm for throughput
    out = torch.bmm(left_4d.to(torch.bfloat16),
                    right_4d.to(torch.bfloat16).transpose(-1, -2)).to(torch.float32)

    # [bs*hidden_dim, N, N] -> [bs, N, N, hidden_dim]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


_trimul_compiled = torch.compile(_trimul_core, mode="reduce-overhead")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    bs, N, _, _ = input_tensor.shape

    # LayerNorm on input
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])

    x_flat = x.reshape(bs * N * N, dim)
    mask_flat = mask.reshape(bs * N * N, 1)

    return _trimul_compiled(
        x_flat, mask_flat, bs, N, hidden_dim,
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #5 — 2026-06-18 21:57:04 UTC ❌ DISCARD

**Hypothesis:** Combined both directions into one edit: (1) Moved the input F.layer_norm inside _trimul_core so the entire computation is in one compiled region; (2) Replaced the permute(0,3,1,2)→reshape→bmm→reshape→

**Result:** 9346.31 μs

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile max-autotune with einsum inside compiled region.
"""

import torch
import torch.nn.functional as F


def _trimul_core(input_tensor, mask, dim, hidden_dim,
                 w_norm, b_norm,
                 w_lp, w_rp, w_lg, w_rg, w_og,
                 w_norm2, b_norm2, w_out):
    bs, N, _, _ = input_tensor.shape

    # LayerNorm inside compiled region for fusion
    x = F.layer_norm(input_tensor, [dim], weight=w_norm, bias=b_norm)

    x_flat = x.reshape(bs * N * N, dim)
    mask_flat = mask.reshape(bs * N * N, 1)

    lp = F.linear(x_flat, w_lp)
    rp = F.linear(x_flat, w_rp)
    lg = F.linear(x_flat, w_lg)
    rg = F.linear(x_flat, w_rg)
    og = F.linear(x_flat, w_og)

    left = lp * lg.sigmoid() * mask_flat    # [bs*N*N, H]
    right = rp * rg.sigmoid() * mask_flat   # [bs*N*N, H]

    # Reshape to [bs, N, N, H] and use einsum for the contraction over k
    # einsum "b i k d, b j k d -> b i j d" — let compiler lower to optimal matmul
    left_4d = left.reshape(bs, N, N, hidden_dim)   # [bs, N, N, H]
    right_4d = right.reshape(bs, N, N, hidden_dim)  # [bs, N, N, H]

    out = torch.einsum('bnkd,bmkd->bnmd', left_4d, right_4d)  # [bs, N, N, H]

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


_trimul_compiled = torch.compile(_trimul_core, mode="max-autotune")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    return _trimul_compiled(
        input_tensor, mask, dim, hidden_dim,
        weights['norm.weight'],
        weights['norm.bias'],
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #6 — 2026-06-18 21:58:13 UTC 💥 CRASH

**Hypothesis:** Restored the experiment #4 structure (reduce-overhead + manual permute→bf16-bmm→permute in bf16) as the compiled core. Added a module-level `_weight_cache` dict keyed by `(dim, hidden_dim)`. On first 

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

**Kernel code:**
```python
"""
Optimized TriMul submission — reduce-overhead compile + cached fused GEMM + bf16 bmm.
"""

import torch
import torch.nn.functional as F

# Module-level cache: keyed by (dim, hidden_dim) -> fused weight [5*H, dim]
_weight_cache = {}


def _trimul_core(x_flat, mask_flat, bs, N, hidden_dim,
                 fused_weight,
                 w_norm2, b_norm2, w_out):
    # Single fused GEMM: [bs*N*N, dim] x [5*H, dim]^T -> [bs*N*N, 5*H]
    all_proj = F.linear(x_flat, fused_weight)
    lp, rp, lg, rg, og = all_proj.split(hidden_dim, dim=-1)

    left = lp * lg.sigmoid() * mask_flat    # [bs*N*N, H]
    right = rp * rg.sigmoid() * mask_flat   # [bs*N*N, H]

    # Reshape for batched matmul: [bs*hidden_dim, N, N]
    left_4d = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # bf16 bmm for throughput on the dominant matmul
    out = torch.bmm(left_4d.to(torch.bfloat16),
                    right_4d.to(torch.bfloat16).transpose(-1, -2)).to(torch.float32)

    # [bs*hidden_dim, N, N] -> [bs, N, N, hidden_dim]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


_trimul_compiled = torch.compile(_trimul_core, mode="reduce-overhead")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    bs, N, _, _ = input_tensor.shape

    # LayerNorm on input (outside compiled region, same as experiment #4)
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])

    x_flat = x.reshape(bs * N * N, dim)
    mask_flat = mask.reshape(bs * N * N, 1)

    # Retrieve or build cached fused weight [5*H, dim]
    cache_key = (dim, hidden_dim)
    if cache_key not in _weight_cache:
        fused_weight = torch.cat([
            weights['left_proj.weight'],
            weights['right_proj.weight'],
            weights['left_gate.weight'],
            weights['right_gate.weight'],
            weights['out_gate.weight'],
        ], dim=0)  # [5*H, dim]
        _weight_cache[cache_key] = fused_weight
    else:
        fused_weight = _weight_cache[cache_key]

    return _trimul_compiled(
        x_flat, mask_flat, bs, N, hidden_dim,
        fused_weight,
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #7 — 2026-06-18 21:59:09 UTC 💥 CRASH

**Hypothesis:** Added module-level `_weight_cache` dict keyed by `id(weights['left_proj.weight'])`. On first call with a given weight tensor, concatenates the 5 projection weights into `[5*H, dim]` (contiguous) and c

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

**Kernel code:**
```python
"""
Optimized TriMul submission — reduce-overhead compile + id()-keyed fused GEMM cache + bf16 bmm.
"""

import torch
import torch.nn.functional as F

# Module-level cache: keyed by id(left_proj.weight) -> fused weight [5*H, dim]
# id() is stable within a benchmark run since weights are fixed tensors.
_weight_cache = {}


def _trimul_core(x_flat, mask_flat, bs, N, hidden_dim,
                 fused_weight,
                 w_norm2, b_norm2, w_out):
    # Single fused GEMM: [bs*N*N, dim] x [5*H, dim]^T -> [bs*N*N, 5*H]
    all_proj = F.linear(x_flat, fused_weight)
    lp, rp, lg, rg, og = all_proj.split(hidden_dim, dim=-1)

    left = lp * lg.sigmoid() * mask_flat    # [bs*N*N, H]
    right = rp * rg.sigmoid() * mask_flat   # [bs*N*N, H]

    # Reshape for batched matmul: [bs*hidden_dim, N, N]
    left_4d = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # bf16 bmm for throughput on the dominant matmul
    out = torch.bmm(left_4d.to(torch.bfloat16),
                    right_4d.to(torch.bfloat16).transpose(-1, -2)).to(torch.float32)

    # [bs*hidden_dim, N, N] -> [bs, N, N, hidden_dim]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


_trimul_compiled = torch.compile(_trimul_core, mode="reduce-overhead")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    bs, N, _, _ = input_tensor.shape

    # LayerNorm on input (outside compiled region, same as experiment #4)
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])

    x_flat = x.reshape(bs * N * N, dim)
    mask_flat = mask.reshape(bs * N * N, 1)

    # Retrieve or build cached fused weight [5*H, dim]
    # Key on id() of the weight tensor — stable within a benchmark run
    cache_key = id(weights['left_proj.weight'])
    if cache_key not in _weight_cache:
        fused_weight = torch.cat([
            weights['left_proj.weight'],
            weights['right_proj.weight'],
            weights['left_gate.weight'],
            weights['right_gate.weight'],
            weights['out_gate.weight'],
        ], dim=0).contiguous()  # [5*H, dim]
        _weight_cache[cache_key] = fused_weight
    else:
        fused_weight = _weight_cache[cache_key]

    return _trimul_compiled(
        x_flat, mask_flat, bs, N, hidden_dim,
        fused_weight,
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #8 — 2026-06-18 22:00:22 UTC ✅ KEEP

**Hypothesis:** Single targeted change — switched `_trimul_compiled = torch.compile(_trimul_core, mode="reduce-overhead")` to `mode="default"`, and added three precision flag settings at module load time (allow_tf32 

**Result:** 3794.52 μs

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile default mode (no CUDA graphs) + bf16 bmm + precision flags.
"""

import torch
import torch.nn.functional as F

# Enable bf16 reduced precision reduction and TF32 for better throughput
# Safe given the 2e-2 tolerance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _trimul_core(x_flat, mask_flat, bs, N, hidden_dim,
                 w_lp, w_rp, w_lg, w_rg, w_og,
                 w_norm2, b_norm2, w_out):
    lp = F.linear(x_flat, w_lp)
    rp = F.linear(x_flat, w_rp)
    lg = F.linear(x_flat, w_lg)
    rg = F.linear(x_flat, w_rg)
    og = F.linear(x_flat, w_og)

    left = lp * lg.sigmoid() * mask_flat
    right = rp * rg.sigmoid() * mask_flat

    # Reshape for batched matmul: [bs*hidden_dim, N, N]
    left_4d = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # bf16 bmm for throughput on the dominant matmul
    out = torch.bmm(left_4d.to(torch.bfloat16),
                    right_4d.to(torch.bfloat16).transpose(-1, -2)).to(torch.float32)

    # [bs*hidden_dim, N, N] -> [bs, N, N, hidden_dim]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


# Use "default" mode: Triton kernel fusion without CUDA graph capture.
# This avoids the tensor-address-freezing issue of "reduce-overhead" while
# still getting elementwise fusion benefits.
_trimul_compiled = torch.compile(_trimul_core, mode="default")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    bs, N, _, _ = input_tensor.shape

    # LayerNorm on input (outside compiled region)
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])

    x_flat = x.reshape(bs * N * N, dim)
    mask_flat = mask.reshape(bs * N * N, 1)

    return _trimul_compiled(
        x_flat, mask_flat, bs, N, hidden_dim,
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #9 — 2026-06-18 22:01:26 UTC 💥 CRASH

**Hypothesis:** Both changes together in one edit: (1) Added `w_norm`/`b_norm` parameters to `_trimul_core` and moved `F.layer_norm + reshape + mask.reshape` inside the compiled region — the compiler can now fuse the

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile default mode + LayerNorm inside compiled region
+ cached fused GEMM (safe with default mode, no CUDA graph address freezing).
"""

import torch
import torch.nn.functional as F

# Enable bf16 reduced precision reduction and TF32 for better throughput
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True

# Module-level weight cache: keyed by id(left_proj.weight) -> fused [5*H, dim] tensor
# Safe with "default" mode since no CUDA graph address freezing occurs.
_weight_cache = {}


def _trimul_core(input_tensor, mask, dim, hidden_dim,
                 w_norm, b_norm,
                 fused_weight,
                 w_norm2, b_norm2, w_out):
    bs = input_tensor.shape[0]
    N = input_tensor.shape[1]

    # LayerNorm inside compiled region — compiler can fuse its output with GEMM input reads
    x = F.layer_norm(input_tensor, [dim], weight=w_norm, bias=b_norm)
    x_flat = x.reshape(bs * N * N, dim)
    mask_flat = mask.reshape(bs * N * N, 1)

    # Single fused GEMM: [bs*N*N, dim] x [5*H, dim]^T -> [bs*N*N, 5*H]
    all_proj = F.linear(x_flat, fused_weight)
    lp, rp, lg, rg, og = all_proj.split(hidden_dim, dim=-1)

    left = lp * lg.sigmoid() * mask_flat    # [bs*N*N, H]
    right = rp * rg.sigmoid() * mask_flat   # [bs*N*N, H]

    # Reshape for batched matmul: [bs*hidden_dim, N, N]
    left_4d = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # bf16 bmm for throughput on the dominant matmul
    out = torch.bmm(left_4d.to(torch.bfloat16),
                    right_4d.to(torch.bfloat16).transpose(-1, -2)).to(torch.float32)

    # [bs*hidden_dim, N, N] -> [bs, N, N, hidden_dim]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


_trimul_compiled = torch.compile(_trimul_core, mode="default")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    # Retrieve or build cached fused weight [5*H, dim]
    # id() is stable within a benchmark run (same weight tensor objects reused)
    cache_key = id(weights['left_proj.weight'])
    fused_weight = _weight_cache.get(cache_key)
    if fused_weight is None:
        fused_weight = torch.cat([
            weights['left_proj.weight'],
            weights['right_proj.weight'],
            weights['left_gate.weight'],
            weights['right_gate.weight'],
            weights['out_gate.weight'],
        ], dim=0).contiguous()  # [5*H, dim]
        _weight_cache[cache_key] = fused_weight

    return _trimul_compiled(
        input_tensor, mask, dim, hidden_dim,
        weights['norm.weight'],
        weights['norm.bias'],
        fused_weight,
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #10 — 2026-06-18 22:02:32 UTC ❌ DISCARD

**Hypothesis:** Moved `F.layer_norm(input_tensor, [dim], ...)`, `x.reshape(bs*N*N, dim)`, and `mask.reshape(bs*N*N, 1)` from `custom_kernel` into `_trimul_core`. Changed the function signature to accept `input_tensor

**Result:** 4488.87 μs

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile default mode + LayerNorm inside compiled region
+ bf16 bmm + precision flags. No caching.
"""

import torch
import torch.nn.functional as F

# Enable bf16 reduced precision reduction and TF32 for better throughput
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _trimul_core(input_tensor, mask,
                 w_norm, b_norm,
                 w_lp, w_rp, w_lg, w_rg, w_og,
                 w_norm2, b_norm2, w_out):
    bs, N, _, dim = input_tensor.shape
    hidden_dim = w_lp.shape[0]

    # LayerNorm inside compiled region — fuses with downstream GEMM input reads
    x = F.layer_norm(input_tensor, [dim], weight=w_norm, bias=b_norm)
    x_flat = x.reshape(bs * N * N, dim)
    mask_flat = mask.reshape(bs * N * N, 1)

    lp = F.linear(x_flat, w_lp)
    rp = F.linear(x_flat, w_rp)
    lg = F.linear(x_flat, w_lg)
    rg = F.linear(x_flat, w_rg)
    og = F.linear(x_flat, w_og)

    left = lp * lg.sigmoid() * mask_flat
    right = rp * rg.sigmoid() * mask_flat

    # Reshape for batched matmul: [bs*hidden_dim, N, N]
    left_4d = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # bf16 bmm for throughput on the dominant matmul
    out = torch.bmm(left_4d.to(torch.bfloat16),
                    right_4d.to(torch.bfloat16).transpose(-1, -2)).to(torch.float32)

    # [bs*hidden_dim, N, N] -> [bs, N, N, hidden_dim]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


_trimul_compiled = torch.compile(_trimul_core, mode="default")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    return _trimul_compiled(
        input_tensor, mask,
        weights['norm.weight'],
        weights['norm.bias'],
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #11 — 2026-06-18 22:03:38 UTC ✅ KEEP

**Hypothesis:** Restored experiment #8's proven structure (LayerNorm outside, x_flat/mask_flat pre-computed and passed as args, bs/N/hidden_dim as Python ints, reduce→default mode, bf16 bmm). Replaced the 5-arg signa

**Result:** 3372.00 μs

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile default mode + per-call fused GEMM (torch.cat outside
compiled region) + bf16 bmm + precision flags. No caching.
"""

import torch
import torch.nn.functional as F

# Enable bf16 reduced precision reduction and TF32 for better throughput
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _trimul_core(x_flat, mask_flat, bs, N, hidden_dim,
                 fused_weight,
                 w_norm2, b_norm2, w_out):
    # Single fused GEMM: [bs*N*N, dim] x [5*H, dim]^T -> [bs*N*N, 5*H]
    all_proj = F.linear(x_flat, fused_weight)
    lp, rp, lg, rg, og = all_proj.split(hidden_dim, dim=-1)

    left = lp * lg.sigmoid() * mask_flat    # [bs*N*N, H]
    right = rp * rg.sigmoid() * mask_flat   # [bs*N*N, H]

    # Reshape for batched matmul: [bs*hidden_dim, N, N]
    left_4d = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # bf16 bmm for throughput on the dominant matmul
    out = torch.bmm(left_4d.to(torch.bfloat16),
                    right_4d.to(torch.bfloat16).transpose(-1, -2)).to(torch.float32)

    # [bs*hidden_dim, N, N] -> [bs, N, N, hidden_dim]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


_trimul_compiled = torch.compile(_trimul_core, mode="default")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    bs, N, _, _ = input_tensor.shape

    # LayerNorm on input (outside compiled region — proven better in exp #8)
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])

    x_flat = x.reshape(bs * N * N, dim)
    mask_flat = mask.reshape(bs * N * N, 1)

    # Build fused weight per-call (no caching — avoids correctness crashes)
    # torch.cat of small [H, dim] matrices is cheap vs. 5x HBM reads of x_flat
    fused_weight = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0)  # [5*H, dim]

    return _trimul_compiled(
        x_flat, mask_flat, bs, N, hidden_dim,
        fused_weight,
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #12 — 2026-06-18 22:04:39 UTC 💥 CRASH

**Hypothesis:** Added module-level `_weight_cache` dict. In `custom_kernel`, compute `cache_key = (w_lp.data_ptr(), w_lp.shape, hidden_dim)` from `weights['left_proj.weight']`. On cache hit, reuse the pre-concatenate

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile default mode + data_ptr-keyed fused GEMM cache
+ bf16 bmm + precision flags.
"""

import torch
import torch.nn.functional as F

# Enable bf16 reduced precision reduction and TF32 for better throughput
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True

# Cache keyed by (data_ptr, shape) of left_proj.weight — provably correct:
# same data pointer + same shape = same tensor data = same fused weight valid.
# Different allocation (correctness checker fresh tensors) = different data_ptr = cache miss = rebuild.
_weight_cache = {}


def _trimul_core(x_flat, mask_flat, bs, N, hidden_dim,
                 fused_weight,
                 w_norm2, b_norm2, w_out):
    # Single fused GEMM: [bs*N*N, dim] x [5*H, dim]^T -> [bs*N*N, 5*H]
    all_proj = F.linear(x_flat, fused_weight)
    lp, rp, lg, rg, og = all_proj.split(hidden_dim, dim=-1)

    left = lp * lg.sigmoid() * mask_flat    # [bs*N*N, H]
    right = rp * rg.sigmoid() * mask_flat   # [bs*N*N, H]

    # Reshape for batched matmul: [bs*hidden_dim, N, N]
    left_4d = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # bf16 bmm for throughput on the dominant matmul
    out = torch.bmm(left_4d.to(torch.bfloat16),
                    right_4d.to(torch.bfloat16).transpose(-1, -2)).to(torch.float32)

    # [bs*hidden_dim, N, N] -> [bs, N, N, hidden_dim]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


_trimul_compiled = torch.compile(_trimul_core, mode="default")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    bs, N, _, _ = input_tensor.shape

    # LayerNorm on input (outside compiled region — proven better in exp #8)
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])

    x_flat = x.reshape(bs * N * N, dim)
    mask_flat = mask.reshape(bs * N * N, 1)

    # Data-pointer-based cache: same GPU memory = same weights = cached fused weight is valid.
    # Fresh tensor allocations (e.g., correctness checker) have different data_ptr = cache miss.
    w_lp = weights['left_proj.weight']
    cache_key = (w_lp.data_ptr(), w_lp.shape, hidden_dim)
    fused_weight = _weight_cache.get(cache_key)
    if fused_weight is None:
        fused_weight = torch.cat([
            weights['left_proj.weight'],
            weights['right_proj.weight'],
            weights['left_gate.weight'],
            weights['right_gate.weight'],
            weights['out_gate.weight'],
        ], dim=0)  # [5*H, dim]
        _weight_cache[cache_key] = fused_weight

    return _trimul_compiled(
        x_flat, mask_flat, bs, N, hidden_dim,
        fused_weight,
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #13 — 2026-06-18 22:05:47 UTC ✅ KEEP

**Hypothesis:** Single targeted change — removed `.contiguous()` from the line `out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()`, making it just `out = out.reshape(bs, hidden_dim, N, N).permu

**Result:** 3362.30 μs

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile default mode + per-call fused GEMM
+ bf16 bmm + precision flags. Remove explicit .contiguous() after permute to
avoid a full tensor copy — let compiler/LayerNorm handle non-contiguous input.
"""

import torch
import torch.nn.functional as F

# Enable bf16 reduced precision reduction and TF32 for better throughput
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _trimul_core(x_flat, mask_flat, bs, N, hidden_dim,
                 fused_weight,
                 w_norm2, b_norm2, w_out):
    # Single fused GEMM: [bs*N*N, dim] x [5*H, dim]^T -> [bs*N*N, 5*H]
    all_proj = F.linear(x_flat, fused_weight)
    lp, rp, lg, rg, og = all_proj.split(hidden_dim, dim=-1)

    left = lp * lg.sigmoid() * mask_flat    # [bs*N*N, H]
    right = rp * rg.sigmoid() * mask_flat   # [bs*N*N, H]

    # Reshape for batched matmul: [bs*hidden_dim, N, N]
    left_4d = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # bf16 bmm for throughput on the dominant matmul
    out = torch.bmm(left_4d.to(torch.bfloat16),
                    right_4d.to(torch.bfloat16).transpose(-1, -2)).to(torch.float32)

    # [bs*hidden_dim, N, N] -> [bs, N, N, hidden_dim]
    # No explicit .contiguous() — let the compiler fuse the permute with the
    # downstream LayerNorm (which handles non-contiguous input natively).
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1)

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


_trimul_compiled = torch.compile(_trimul_core, mode="default")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    bs, N, _, _ = input_tensor.shape

    # LayerNorm on input (outside compiled region — proven better in exp #8)
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])

    x_flat = x.reshape(bs * N * N, dim)
    mask_flat = mask.reshape(bs * N * N, 1)

    # Build fused weight per-call (no caching — avoids correctness crashes)
    fused_weight = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0)  # [5*H, dim]

    return _trimul_compiled(
        x_flat, mask_flat, bs, N, hidden_dim,
        fused_weight,
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #14 — 2026-06-18 22:06:54 UTC ❌ DISCARD

**Hypothesis:** Single one-line change — added `fullgraph=True` to the `torch.compile` call: `_trimul_compiled = torch.compile(_trimul_core, mode="default", fullgraph=True)`. All other code is identical to experiment

**Result:** 3363.97 μs

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile default mode + fullgraph=True + per-call fused GEMM
+ bf16 bmm + precision flags.
"""

import torch
import torch.nn.functional as F

# Enable bf16 reduced precision reduction and TF32 for better throughput
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _trimul_core(x_flat, mask_flat, bs, N, hidden_dim,
                 fused_weight,
                 w_norm2, b_norm2, w_out):
    # Single fused GEMM: [bs*N*N, dim] x [5*H, dim]^T -> [bs*N*N, 5*H]
    all_proj = F.linear(x_flat, fused_weight)
    lp, rp, lg, rg, og = all_proj.split(hidden_dim, dim=-1)

    left = lp * lg.sigmoid() * mask_flat    # [bs*N*N, H]
    right = rp * rg.sigmoid() * mask_flat   # [bs*N*N, H]

    # Reshape for batched matmul: [bs*hidden_dim, N, N]
    left_4d = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # bf16 bmm for throughput on the dominant matmul
    out = torch.bmm(left_4d.to(torch.bfloat16),
                    right_4d.to(torch.bfloat16).transpose(-1, -2)).to(torch.float32)

    # [bs*hidden_dim, N, N] -> [bs, N, N, hidden_dim]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1)

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


# fullgraph=True forces the compiler to compile the entire function as one graph
# with no graph breaks, enabling deeper inter-op fusion across permutes and bmm.
_trimul_compiled = torch.compile(_trimul_core, mode="default", fullgraph=True)


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    bs, N, _, _ = input_tensor.shape

    # LayerNorm on input (outside compiled region — proven better in exp #8)
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])

    x_flat = x.reshape(bs * N * N, dim)
    mask_flat = mask.reshape(bs * N * N, 1)

    # Build fused weight per-call (no caching — avoids correctness crashes)
    fused_weight = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0)  # [5*H, dim]

    return _trimul_compiled(
        x_flat, mask_flat, bs, N, hidden_dim,
        fused_weight,
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #15 — 2026-06-18 22:08:22 UTC 💥 CRASH

**Hypothesis:** A Triton kernel `fused_proj_transpose_kernel` that:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

**Kernel code:**
```python
"""
Optimized TriMul submission — Custom Triton kernel for fused projection + gating + layout reorder,
eliminating the permute-before-bmm memory copy. Post-bmm path uses torch.compile default mode.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


@triton.jit
def fused_proj_transpose_kernel(
    # Inputs
    x_ptr,       # [M, K]  M = bs*N*N, K = dim
    w_lp_ptr,    # [H, K]
    w_rp_ptr,    # [H, K]
    w_lg_ptr,    # [H, K]
    w_rg_ptr,    # [H, K]
    w_og_ptr,    # [H, K]
    mask_ptr,    # [M]
    # Outputs: layout [bs*H, N, N] = [bs*H, N*N] treated as [BH, N, N]
    left_out_ptr,   # [bs*H, N*N]  -> will be viewed as [bs*H, N, N] for bmm
    right_out_ptr,  # [bs*H, N*N]
    og_out_ptr,     # [M, H]  (og stays in [M, H] layout for later use)
    # Dimensions
    M,    # bs * N * N
    K: tl.constexpr,    # dim
    H: tl.constexpr,    # hidden_dim
    N,    # seqlen
    bs,   # batch size
    # Strides
    stride_xm, stride_xk,
    stride_wh, stride_wk,
    stride_left_bh, stride_left_n,   # strides for left_out [bs*H, N*N]
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Each program instance handles a tile of (m_block, h_block).
    m = flat spatial index (0..M-1), maps to (batch, row, col) in [bs, N, N]
    h = hidden dim index (0..H-1)
    
    Output written to left_out[b*H + h, row*N + col] (transposed layout)
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Row range in x: [pid_m*BLOCK_M, (pid_m+1)*BLOCK_M)
    m_start = pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M

    # Hidden dim range
    h_start = pid_h * BLOCK_H
    h_offs = h_start + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # Accumulate dot products over K dimension
    acc_lp = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    acc_rp = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    acc_lg = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    acc_rg = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    acc_og = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Load x tile: [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load weight tiles: [BLOCK_H, BLOCK_K]
        w_ptrs_lp = w_lp_ptr + h_offs[:, None] * stride_wh + k_offs[None, :] * stride_wk
        w_ptrs_rp = w_rp_ptr + h_offs[:, None] * stride_wh + k_offs[None, :] * stride_wk
        w_ptrs_lg = w_lg_ptr + h_offs[:, None] * stride_wh + k_offs[None, :] * stride_wk
        w_ptrs_rg = w_rg_ptr + h_offs[:, None] * stride_wh + k_offs[None, :] * stride_wk
        w_ptrs_og = w_og_ptr + h_offs[:, None] * stride_wh + k_offs[None, :] * stride_wk

        wlp = tl.load(w_ptrs_lp, mask=h_mask[:, None] & k_mask[None, :], other=0.0)
        wrp = tl.load(w_ptrs_rp, mask=h_mask[:, None] & k_mask[None, :], other=0.0)
        wlg = tl.load(w_ptrs_lg, mask=h_mask[:, None] & k_mask[None, :], other=0.0)
        wrg = tl.load(w_ptrs_rg, mask=h_mask[:, None] & k_mask[None, :], other=0.0)
        wog = tl.load(w_ptrs_og, mask=h_mask[:, None] & k_mask[None, :], other=0.0)

        # Accumulate: x_tile [M, K] @ w.T [K, H] -> [M, H]
        acc_lp += tl.dot(x_tile, tl.trans(wlp))
        acc_rp += tl.dot(x_tile, tl.trans(wrp))
        acc_lg += tl.dot(x_tile, tl.trans(wlg))
        acc_rg += tl.dot(x_tile, tl.trans(wrg))
        acc_og += tl.dot(x_tile, tl.trans(wog))

    # Apply gates
    left_val = acc_lp * tl.sigmoid(acc_lg)    # [BLOCK_M, BLOCK_H]
    right_val = acc_rp * tl.sigmoid(acc_rg)   # [BLOCK_M, BLOCK_H]
    og_val = acc_og  # store raw og (sigmoid applied later)

    # Apply mask: mask[m] in [0,1]
    mask_vals = tl.load(mask_ptr + m_offs, mask=m_mask, other=0.0)  # [BLOCK_M]
    left_val = left_val * mask_vals[:, None]
    right_val = right_val * mask_vals[:, None]

    # Write left/right to transposed layout [bs*H, N*N]
    # m_offs[i] = b*N*N + row*N + col, so b = m//N//N, spatial = m % (N*N)
    # Output index: [b*H + h, spatial] = [(m//N//N)*H + h, m%(N*N)]
    # stride_left_bh = N*N, stride_left_n = 1
    for i in range(BLOCK_M):
        m_i = m_start + i
        if m_i < M:
            b_i = m_i // (N * N)
            spatial_i = m_i % (N * N)
            for j in range(BLOCK_H):
                h_j = h_start + j
                if h_j < H:
                    out_idx = (b_i * H + h_j) * stride_left_bh + spatial_i * stride_left_n
                    tl.store(left_out_ptr + out_idx, left_val[i, j])
                    tl.store(right_out_ptr + out_idx, right_val[i, j])
                    # og in standard [M, H] layout
                    tl.store(og_out_ptr + m_i * H + h_j, og_val[i, j])


def fused_proj_gated(x_flat, w_lp, w_rp, w_lg, w_rg, w_og, mask_flat, bs, N, H):
    """
    Fused projection + gating + layout reorder.
    Returns:
      left:  [bs*H, N, N] contiguous (for bmm)
      right: [bs*H, N, N] contiguous (for bmm)
      og:    [bs*N*N, H]  (raw, sigmoid applied later)
    """
    M = bs * N * N
    K = x_flat.shape[1]

    left_out = torch.empty(bs * H, N * N, device=x_flat.device, dtype=torch.float32)
    right_out = torch.empty(bs * H, N * N, device=x_flat.device, dtype=torch.float32)
    og_out = torch.empty(M, H, device=x_flat.device, dtype=torch.float32)

    BLOCK_M = 32
    BLOCK_H = 32
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_H))

    fused_proj_transpose_kernel[grid](
        x_flat, w_lp, w_rp, w_lg, w_rg, w_og,
        mask_flat.squeeze(-1),
        left_out, right_out, og_out,
        M, K, H, N, bs,
        x_flat.stride(0), x_flat.stride(1),
        w_lp.stride(0), w_lp.stride(1),
        left_out.stride(0), left_out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
    )

    return left_out.view(bs * H, N, N), right_out.view(bs * H, N, N), og_out


def _trimul_post(out_bh_nn, og_flat, bs, N, hidden_dim, w_norm2, b_norm2, w_out):
    """Post-bmm: reshape, layernorm, outgate, linear."""
    # out_bh_nn: [bs*H, N, N] -> [bs, N, N, H]
    out = out_bh_nn.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1)
    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og_flat.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


_post_compiled = torch.compile(_trimul_post, mode="default")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    bs, N, _, _ = input_tensor.shape

    # LayerNorm
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])

    x_flat = x.reshape(bs * N * N, dim)
    mask_flat = mask.reshape(bs * N * N, 1)

    # Fused Triton kernel: project + gate + write in [bs*H, N, N] layout directly
    left_4d, right_4d, og_flat = fused_proj_gated(
        x_flat,
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
        mask_flat, bs, N, hidden_dim
    )

    # bmm on contiguous [bs*H, N, N] tensors — no permute needed
    out = torch.bmm(left_4d.to(torch.bfloat16),
                    right_4d.to(torch.bfloat16).transpose(-1, -2)).to(torch.float32)

    return _post_compiled(
        out, og_flat, bs, N, hidden_dim,
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #16 — 2026-06-18 22:09:55 UTC ❌ DISCARD

**Hypothesis:** `reorder_sigmoid_mask_kernel` — a Triton kernel where each program handles one spatial index `m` (grid size = M = bs*N*N). For each m, it:

**Result:** 4794.14 μs

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile default mode + per-call fused GEMM
+ Triton layout-reorder kernel (sigmoid+mask+[M,H]->[bs*H,N,N]) + bf16 bmm.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


@triton.jit
def reorder_sigmoid_mask_kernel(
    # Inputs
    lp_ptr,     # [M, H]
    rp_ptr,     # [M, H]
    lg_ptr,     # [M, H]
    rg_ptr,     # [M, H]
    mask_ptr,   # [M]
    # Outputs
    left_ptr,   # [bs*H, N*N] = [bs*H, M/bs]
    right_ptr,  # [bs*H, N*N]
    # Dimensions
    M,          # bs * N * N
    H,          # hidden_dim
    NN,         # N * N  (spatial size per batch)
    # Block sizes
    BLOCK_H: tl.constexpr,
):
    """
    Each program handles one spatial index m (0..M-1).
    It loads all H values for that m, applies sigmoid+gate+mask,
    and writes to the transposed output layout [bs*H, N*N].
    
    Output index: left_out[b*H + h, spatial] where b = m // NN, spatial = m % NN
    """
    m = tl.program_id(0)

    # Compute batch index and spatial index
    b = m // NN
    spatial = m % NN

    # Load mask value for this spatial position
    mask_val = tl.load(mask_ptr + m)

    # Process H hidden dims in blocks of BLOCK_H
    for h_start in tl.range(0, H, BLOCK_H):
        h_offs = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offs < H

        # Load projection values: [BLOCK_H]
        lp_vals = tl.load(lp_ptr + m * H + h_offs, mask=h_mask, other=0.0)
        rp_vals = tl.load(rp_ptr + m * H + h_offs, mask=h_mask, other=0.0)
        lg_vals = tl.load(lg_ptr + m * H + h_offs, mask=h_mask, other=0.0)
        rg_vals = tl.load(rg_ptr + m * H + h_offs, mask=h_mask, other=0.0)

        # Apply gating and mask
        left_vals = lp_vals * tl.sigmoid(lg_vals) * mask_val
        right_vals = rp_vals * tl.sigmoid(rg_vals) * mask_val

        # Write to transposed layout: out[b*H + h, spatial]
        out_row = b * H + h_offs   # [BLOCK_H]
        out_idx = out_row * NN + spatial  # [BLOCK_H]

        tl.store(left_ptr + out_idx, left_vals, mask=h_mask)
        tl.store(right_ptr + out_idx, right_vals, mask=h_mask)


def reorder_sigmoid_mask(lp, rp, lg, rg, mask_flat, bs, N, H):
    """
    Apply sigmoid gating, mask, and reorder from [M, H] to [bs*H, N, N].
    Returns left [bs*H, N, N], right [bs*H, N, N] contiguous.
    """
    M = bs * N * N
    NN = N * N

    left_out = torch.empty(bs * H, NN, device=lp.device, dtype=torch.float32)
    right_out = torch.empty(bs * H, NN, device=lp.device, dtype=torch.float32)

    # Make inputs contiguous for efficient access
    lp_c = lp.contiguous()
    rp_c = rp.contiguous()
    lg_c = lg.contiguous()
    rg_c = rg.contiguous()
    mask_c = mask_flat.squeeze(-1).contiguous()

    BLOCK_H = min(128, triton.next_power_of_2(H))
    grid = (M,)

    reorder_sigmoid_mask_kernel[grid](
        lp_c, rp_c, lg_c, rg_c, mask_c,
        left_out, right_out,
        M, H, NN,
        BLOCK_H=BLOCK_H,
    )

    return left_out.view(bs * H, N, N), right_out.view(bs * H, N, N)


def _trimul_core(x_flat, mask_flat, bs, N, hidden_dim,
                 fused_weight,
                 w_norm2, b_norm2, w_out):
    # Single fused GEMM: [bs*N*N, dim] x [5*H, dim]^T -> [bs*N*N, 5*H]
    all_proj = F.linear(x_flat, fused_weight)
    lp, rp, lg, rg, og = all_proj.split(hidden_dim, dim=-1)

    # Triton kernel: sigmoid+gate+mask + layout reorder [M,H] -> [bs*H, N, N]
    left_4d, right_4d = reorder_sigmoid_mask(lp, rp, lg, rg, mask_flat, bs, N, hidden_dim)

    # bf16 bmm on already-contiguous [bs*H, N, N] tensors — no permute needed
    out = torch.bmm(left_4d.to(torch.bfloat16),
                    right_4d.to(torch.bfloat16).transpose(-1, -2)).to(torch.float32)

    # [bs*H, N, N] -> [bs, N, N, H]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1)

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


_trimul_compiled = torch.compile(_trimul_core, mode="default")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    bs, N, _, _ = input_tensor.shape

    # LayerNorm on input
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])

    x_flat = x.reshape(bs * N * N, dim)
    mask_flat = mask.reshape(bs * N * N, 1)

    # Build fused weight per-call (no caching)
    fused_weight = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0)  # [5*H, dim]

    return _trimul_compiled(
        x_flat, mask_flat, bs, N, hidden_dim,
        fused_weight,
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #17 — 2026-06-18 22:11:15 UTC ✅ KEEP

**Hypothesis:** Single targeted change — replaced `.to(torch.bfloat16)` with `.to(torch.float16)` for both left_4d and right_4d before torch.bmm, and kept the `.to(torch.float32)` cast on the output. Restored the cle

**Result:** 3343.04 μs

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile default mode + per-call fused GEMM
+ fp16 bmm (instead of bf16) + precision flags.
"""

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _trimul_core(x_flat, mask_flat, bs, N, hidden_dim,
                 fused_weight,
                 w_norm2, b_norm2, w_out):
    # Single fused GEMM: [bs*N*N, dim] x [5*H, dim]^T -> [bs*N*N, 5*H]
    all_proj = F.linear(x_flat, fused_weight)
    lp, rp, lg, rg, og = all_proj.split(hidden_dim, dim=-1)

    left = lp * lg.sigmoid() * mask_flat    # [bs*N*N, H]
    right = rp * rg.sigmoid() * mask_flat   # [bs*N*N, H]

    # Reshape for batched matmul: [bs*hidden_dim, N, N]
    left_4d = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # fp16 bmm — higher Tensor Core throughput on H100 vs bf16 for some shapes
    out = torch.bmm(left_4d.to(torch.float16),
                    right_4d.to(torch.float16).transpose(-1, -2)).to(torch.float32)

    # [bs*hidden_dim, N, N] -> [bs, N, N, hidden_dim]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1)

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


_trimul_compiled = torch.compile(_trimul_core, mode="default")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    bs, N, _, _ = input_tensor.shape

    # LayerNorm on input (outside compiled region — proven best in exp #8/11/13)
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])

    x_flat = x.reshape(bs * N * N, dim)
    mask_flat = mask.reshape(bs * N * N, 1)

    # Build fused weight per-call (no caching — avoids correctness crashes)
    fused_weight = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0)  # [5*H, dim]

    return _trimul_compiled(
        x_flat, mask_flat, bs, N, hidden_dim,
        fused_weight,
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #18 — 2026-06-18 22:12:30 UTC ❌ DISCARD

**Hypothesis:** Cast `x_flat`, `mask_flat`, `fused_weight`, `w_norm2`, `b_norm2`, `w_out` to fp16 in `custom_kernel` before passing to the compiled function. Inside `_trimul_core`, everything operates in fp16 — the f

**Result:** 4439.33 μs

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile default mode + full fp16 forward pass
+ per-call fused GEMM + precision flags.
Running everything in fp16 halves memory bandwidth and enables fp16 Tensor Cores throughout.
"""

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _trimul_core(x_flat_fp16, mask_flat_fp16, bs, N, hidden_dim,
                 fused_weight_fp16,
                 w_norm2_fp16, b_norm2_fp16, w_out_fp16):
    # All ops in fp16: fused GEMM, gating, permutes, bmm, LayerNorm, final linear

    # Single fused GEMM: [bs*N*N, dim] x [5*H, dim]^T -> [bs*N*N, 5*H]  (fp16)
    all_proj = F.linear(x_flat_fp16, fused_weight_fp16)
    lp, rp, lg, rg, og = all_proj.split(hidden_dim, dim=-1)

    left = lp * lg.sigmoid() * mask_flat_fp16    # [bs*N*N, H]  fp16
    right = rp * rg.sigmoid() * mask_flat_fp16   # [bs*N*N, H]  fp16

    # Reshape for batched matmul: [bs*hidden_dim, N, N]  — permute is fp16 (2x less BW)
    left_4d = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # fp16 bmm — already fp16, no cast needed
    out = torch.bmm(left_4d, right_4d.transpose(-1, -2))  # [bs*H, N, N]  fp16

    # [bs*hidden_dim, N, N] -> [bs, N, N, hidden_dim]  fp16
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1)

    # LayerNorm in fp16
    out = F.layer_norm(out, [hidden_dim], weight=w_norm2_fp16, bias=b_norm2_fp16)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)

    # Final linear in fp16, cast output to fp32
    out = F.linear(out, w_out_fp16).to(torch.float32)
    return out


_trimul_compiled = torch.compile(_trimul_core, mode="default")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    bs, N, _, _ = input_tensor.shape

    # LayerNorm on input in fp32 (for numerical stability of norm computation)
    # then cast to fp16 for all downstream ops
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])

    x_flat_fp16 = x.reshape(bs * N * N, dim).to(torch.float16)
    mask_flat_fp16 = mask.reshape(bs * N * N, 1).to(torch.float16)

    # Build fused weight in fp16 per-call (no caching)
    fused_weight_fp16 = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0).to(torch.float16)  # [5*H, dim] fp16

    return _trimul_compiled(
        x_flat_fp16, mask_flat_fp16, bs, N, hidden_dim,
        fused_weight_fp16,
        weights['to_out_norm.weight'].to(torch.float16),
        weights['to_out_norm.bias'].to(torch.float16),
        weights['to_out.weight'].to(torch.float16),
    )

```

---

## Experiment #19 — 2026-06-18 22:14:05 UTC ❌ DISCARD

**Hypothesis:** Single one-line change — added `dynamic=True` to the compile call: `_trimul_compiled = torch.compile(_trimul_core, mode="default", dynamic=True)`. Restored experiment #17's proven structure (fp32 GEMM

**Result:** 4185.10 μs

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile default mode with dynamic=True + per-call fused GEMM
+ fp16 bmm + precision flags.
dynamic=True enables symbolic shape compilation — one compiled graph handles all shapes,
eliminating the 7-trace overhead from Python integer shape arguments.
"""

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _trimul_core(x_flat, mask_flat, bs, N, hidden_dim,
                 fused_weight,
                 w_norm2, b_norm2, w_out):
    # Single fused GEMM: [bs*N*N, dim] x [5*H, dim]^T -> [bs*N*N, 5*H]
    all_proj = F.linear(x_flat, fused_weight)
    lp, rp, lg, rg, og = all_proj.split(hidden_dim, dim=-1)

    left = lp * lg.sigmoid() * mask_flat    # [bs*N*N, H]
    right = rp * rg.sigmoid() * mask_flat   # [bs*N*N, H]

    # Reshape for batched matmul: [bs*hidden_dim, N, N]
    left_4d = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # fp16 bmm for throughput on the dominant matmul
    out = torch.bmm(left_4d.to(torch.float16),
                    right_4d.to(torch.float16).transpose(-1, -2)).to(torch.float32)

    # [bs*hidden_dim, N, N] -> [bs, N, N, hidden_dim]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1)

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


# dynamic=True: compile one graph with symbolic shapes instead of 7 separate traces.
# Eliminates per-shape recompilation overhead from bs, N, hidden_dim integer arguments.
_trimul_compiled = torch.compile(_trimul_core, mode="default", dynamic=True)


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    bs, N, _, _ = input_tensor.shape

    # LayerNorm on input (outside compiled region — proven best)
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])

    x_flat = x.reshape(bs * N * N, dim)
    mask_flat = mask.reshape(bs * N * N, 1)

    # Build fused weight per-call (no caching — avoids correctness crashes)
    fused_weight = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0)  # [5*H, dim]

    return _trimul_compiled(
        x_flat, mask_flat, bs, N, hidden_dim,
        fused_weight,
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #20 — 2026-06-18 22:17:07 UTC ❌ DISCARD

**Hypothesis:** Single one-line change — replaced `mode="default"` with `mode="max-autotune-no-cudagraphs"` in the torch.compile call. All other code is identical to experiment #17 (fp32 GEMM, fp16 cast just for bmm,

**Result:** 8170.43 μs

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile max-autotune-no-cudagraphs + per-call fused GEMM
+ fp16 bmm + precision flags.
max-autotune-no-cudagraphs runs Triton autotuning for optimal GEMM tile sizes
without CUDA graph capture (avoids address-freezing correctness issues).
"""

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _trimul_core(x_flat, mask_flat, bs, N, hidden_dim,
                 fused_weight,
                 w_norm2, b_norm2, w_out):
    # Single fused GEMM: [bs*N*N, dim] x [5*H, dim]^T -> [bs*N*N, 5*H]
    all_proj = F.linear(x_flat, fused_weight)
    lp, rp, lg, rg, og = all_proj.split(hidden_dim, dim=-1)

    left = lp * lg.sigmoid() * mask_flat    # [bs*N*N, H]
    right = rp * rg.sigmoid() * mask_flat   # [bs*N*N, H]

    # Reshape for batched matmul: [bs*hidden_dim, N, N]
    left_4d = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # fp16 bmm for throughput on the dominant matmul
    out = torch.bmm(left_4d.to(torch.float16),
                    right_4d.to(torch.float16).transpose(-1, -2)).to(torch.float32)

    # [bs*hidden_dim, N, N] -> [bs, N, N, hidden_dim]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1)

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


# max-autotune-no-cudagraphs: Triton autotuning for optimal GEMM/bmm tile configs,
# without CUDA graph capture. Different from max-autotune (which uses CUDA graphs
# and caused correctness crashes). Autotuning runs during warmup, not during timing.
_trimul_compiled = torch.compile(_trimul_core, mode="max-autotune-no-cudagraphs")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    bs, N, _, _ = input_tensor.shape

    # LayerNorm on input (outside compiled region — proven best in exp #17)
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])

    x_flat = x.reshape(bs * N * N, dim)
    mask_flat = mask.reshape(bs * N * N, 1)

    # Build fused weight per-call (no caching — avoids correctness crashes)
    fused_weight = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0)  # [5*H, dim]

    return _trimul_compiled(
        x_flat, mask_flat, bs, N, hidden_dim,
        fused_weight,
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #21 — 2026-06-18 22:18:27 UTC ✅ KEEP

**Hypothesis:** Changed `_trimul_core` signature from `(x_flat, mask_flat, bs, N, hidden_dim, fused_weight, ...)` to `(x_4d, mask, fused_weight, ...)` — removing bs, N, hidden_dim as Python integer arguments entirely

**Result:** 3331.53 μs

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile default mode + tensor-shape-derived dims
(no Python integer args bs/N/hidden_dim) + per-call fused GEMM + fp16 bmm.
"""

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _trimul_core(x_4d, mask, fused_weight, w_norm2, b_norm2, w_out):
    """
    x_4d:        [bs, N, N, dim]  — normalized input (LayerNorm applied outside)
    mask:        [bs, N, N]
    fused_weight:[5*H, dim]
    """
    bs   = x_4d.shape[0]
    N    = x_4d.shape[1]
    dim  = x_4d.shape[3]
    H5   = fused_weight.shape[0]
    hidden_dim = H5 // 5
    M    = bs * N * N

    x_flat   = x_4d.reshape(M, dim)
    mask_flat = mask.reshape(M, 1)

    # Single fused GEMM: [M, dim] x [5*H, dim]^T -> [M, 5*H]
    all_proj = F.linear(x_flat, fused_weight)
    lp, rp, lg, rg, og = all_proj.split(hidden_dim, dim=-1)

    left  = lp * lg.sigmoid() * mask_flat    # [M, H]
    right = rp * rg.sigmoid() * mask_flat    # [M, H]

    # Reshape for batched matmul: [bs*H, N, N]
    left_4d  = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # fp16 bmm
    out = torch.bmm(left_4d.to(torch.float16),
                    right_4d.to(torch.float16).transpose(-1, -2)).to(torch.float32)

    # [bs*H, N, N] -> [bs, N, N, H]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1)

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


_trimul_compiled = torch.compile(_trimul_core, mode="default")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]

    # LayerNorm on input (outside compiled region — proven best)
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])
    # x is [bs, N, N, dim] — pass as 4D tensor, no Python int args for shapes

    # Build fused weight per-call (no caching)
    fused_weight = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0)  # [5*H, dim]

    return _trimul_compiled(
        x, mask, fused_weight,
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #22 — 2026-06-18 22:19:53 UTC ❌ DISCARD

**Hypothesis:** Merged the entire computation into `_trimul_full` — a single function accepting individual weight tensors (not pre-fused). Inside: LayerNorm, torch.cat([w_lp, w_rp, w_lg, w_rg, w_og]) to build fused_w

**Result:** 4049.70 μs

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile default mode + entire pipeline in one compiled graph:
LayerNorm + torch.cat(weights) + fused GEMM + sigmoid/gate/mask + fp16 bmm + post-processing.
All tensor-only args (no Python int scalars).
"""

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _trimul_full(input_tensor, mask,
                 w_norm, b_norm,
                 w_lp, w_rp, w_lg, w_rg, w_og,
                 w_norm2, b_norm2, w_out):
    """
    Full pipeline in one compiled graph.
    input_tensor: [bs, N, N, dim]
    mask:         [bs, N, N]
    """
    bs  = input_tensor.shape[0]
    N   = input_tensor.shape[1]
    dim = input_tensor.shape[3]
    hidden_dim = w_lp.shape[0]
    M   = bs * N * N

    # LayerNorm inside compiled region
    x = F.layer_norm(input_tensor, [dim], weight=w_norm, bias=b_norm)
    x_flat    = x.reshape(M, dim)
    mask_flat = mask.reshape(M, 1)

    # Fuse all 5 projections into one GEMM via torch.cat inside compiled region
    # (compiler can see this as one op and may fuse with downstream split)
    fused_weight = torch.cat([w_lp, w_rp, w_lg, w_rg, w_og], dim=0)  # [5*H, dim]
    all_proj = F.linear(x_flat, fused_weight)
    lp, rp, lg, rg, og = all_proj.split(hidden_dim, dim=-1)

    left  = lp * lg.sigmoid() * mask_flat    # [M, H]
    right = rp * rg.sigmoid() * mask_flat    # [M, H]

    # Reshape for batched matmul: [bs*H, N, N]
    left_4d  = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # fp16 bmm
    out = torch.bmm(left_4d.to(torch.float16),
                    right_4d.to(torch.float16).transpose(-1, -2)).to(torch.float32)

    # [bs*H, N, N] -> [bs, N, N, H]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1)

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


_trimul_compiled = torch.compile(_trimul_full, mode="default")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    return _trimul_compiled(
        input_tensor, mask,
        weights['norm.weight'],
        weights['norm.bias'],
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #23 — 2026-06-18 22:20:47 UTC 💥 CRASH

**Hypothesis:** Restored experiment #21's exact code (4D x_4d tensor, no Python int args, LayerNorm outside, per-call fused weight cat, fp16 bmm, mode="default") and added `options={"shape_padding": True}` to the tor

**Result:** CRASH

**Error:**
```
Correctness check failed. Failed to load submission:
Traceback (most recent call last):
  File "/root/eval_modal_trimul.py", line 245, in evaluate_kernel
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 940, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/tmp/submission_se3o2z9t/submission.py", line 57, in <module>
    _trimul_comp
```

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile default mode + shape_padding=True option
+ tensor-shape-derived dims + per-call fused GEMM + fp16 bmm.
shape_padding pads to power-of-2 sizes for irregular seqlens (256,512,768,1024).
"""

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _trimul_core(x_4d, mask, fused_weight, w_norm2, b_norm2, w_out):
    """
    x_4d:        [bs, N, N, dim]  — normalized input (LayerNorm applied outside)
    mask:        [bs, N, N]
    fused_weight:[5*H, dim]
    """
    bs         = x_4d.shape[0]
    N          = x_4d.shape[1]
    dim        = x_4d.shape[3]
    hidden_dim = fused_weight.shape[0] // 5
    M          = bs * N * N

    x_flat    = x_4d.reshape(M, dim)
    mask_flat = mask.reshape(M, 1)

    # Single fused GEMM: [M, dim] x [5*H, dim]^T -> [M, 5*H]
    all_proj = F.linear(x_flat, fused_weight)
    lp, rp, lg, rg, og = all_proj.split(hidden_dim, dim=-1)

    left  = lp * lg.sigmoid() * mask_flat    # [M, H]
    right = rp * rg.sigmoid() * mask_flat    # [M, H]

    # Reshape for batched matmul: [bs*H, N, N]
    left_4d  = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # fp16 bmm
    out = torch.bmm(left_4d.to(torch.float16),
                    right_4d.to(torch.float16).transpose(-1, -2)).to(torch.float32)

    # [bs*H, N, N] -> [bs, N, N, H]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1)

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


# shape_padding=True: pads tensor dimensions to power-of-2 sizes, which can improve
# performance for irregular shapes like seqlen=768 by aligning GEMM/bmm dimensions
# to boundaries that cuBLAS and Triton handle more efficiently.
_trimul_compiled = torch.compile(
    _trimul_core,
    mode="default",
    options={"shape_padding": True},
)


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]

    # LayerNorm on input (outside compiled region — proven best)
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])
    # x is [bs, N, N, dim] — pass as 4D tensor, no Python int args for shapes

    # Build fused weight per-call (no caching)
    fused_weight = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0)  # [5*H, dim]

    return _trimul_compiled(
        x, mask, fused_weight,
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #24 — 2026-06-18 22:21:56 UTC ❌ DISCARD

**Hypothesis:** Changed the bmm contraction from:

**Result:** 4242.01 μs

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile default mode + 4D torch.matmul
(eliminates reshape before bmm, keeps [bs,H,N,N] throughout) + fp16 + tensor-shape dims.
"""

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _trimul_core(x_4d, mask, fused_weight, w_norm2, b_norm2, w_out):
    """
    x_4d:        [bs, N, N, dim]  — normalized input (LayerNorm applied outside)
    mask:        [bs, N, N]
    fused_weight:[5*H, dim]
    """
    bs         = x_4d.shape[0]
    N          = x_4d.shape[1]
    dim        = x_4d.shape[3]
    hidden_dim = fused_weight.shape[0] // 5
    M          = bs * N * N

    x_flat    = x_4d.reshape(M, dim)
    mask_flat = mask.reshape(M, 1)

    # Single fused GEMM: [M, dim] x [5*H, dim]^T -> [M, 5*H]
    all_proj = F.linear(x_flat, fused_weight)
    lp, rp, lg, rg, og = all_proj.split(hidden_dim, dim=-1)

    left  = lp * lg.sigmoid() * mask_flat    # [M, H]
    right = rp * rg.sigmoid() * mask_flat    # [M, H]

    # Reshape to [bs, H, N, N] — one permute, no second reshape needed
    left_bhnn  = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2)   # [bs, H, N, N]
    right_bhnn = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2)  # [bs, H, N, N]

    # 4D torch.matmul: [bs, H, N, N] @ [bs, H, N, N]^T -> [bs, H, N, N]
    # Uses cuBLAS 4D strided batched GEMM — no extra reshape vs bmm path
    out = torch.matmul(
        left_bhnn.to(torch.float16),
        right_bhnn.to(torch.float16).transpose(-2, -1)
    ).to(torch.float32)  # [bs, H, N, N]

    # [bs, H, N, N] -> [bs, N, N, H]  — one permute
    out = out.permute(0, 2, 3, 1)

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


_trimul_compiled = torch.compile(_trimul_core, mode="default")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]

    # LayerNorm on input (outside compiled region — proven best)
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])

    # Build fused weight per-call (no caching)
    fused_weight = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0)  # [5*H, dim]

    return _trimul_compiled(
        x, mask, fused_weight,
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #25 — 2026-06-18 22:23:10 UTC ❌ DISCARD

**Hypothesis:** Restored experiment #21's exact code: torch.compile(default) + LayerNorm outside compiled region + per-call torch.cat fused weight [5*H,dim] + single fused GEMM + fp16 bmm with explicit permute→reshap

**Result:** 3337.42 μs

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile default mode + tensor-shape-derived dims
(no Python integer args bs/N/hidden_dim) + per-call fused GEMM + fp16 bmm.
This is experiment #21 — the proven best structure at 3,331 μs geomean.
"""

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _trimul_core(x_4d, mask, fused_weight, w_norm2, b_norm2, w_out):
    """
    x_4d:        [bs, N, N, dim]  — normalized input (LayerNorm applied outside)
    mask:        [bs, N, N]
    fused_weight:[5*H, dim]
    """
    bs         = x_4d.shape[0]
    N          = x_4d.shape[1]
    dim        = x_4d.shape[3]
    hidden_dim = fused_weight.shape[0] // 5
    M          = bs * N * N

    x_flat    = x_4d.reshape(M, dim)
    mask_flat = mask.reshape(M, 1)

    # Single fused GEMM: [M, dim] x [5*H, dim]^T -> [M, 5*H]
    all_proj = F.linear(x_flat, fused_weight)
    lp, rp, lg, rg, og = all_proj.split(hidden_dim, dim=-1)

    left  = lp * lg.sigmoid() * mask_flat    # [M, H]
    right = rp * rg.sigmoid() * mask_flat    # [M, H]

    # Reshape for batched matmul: [bs*H, N, N]
    left_4d  = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # fp16 bmm for throughput on the dominant matmul
    out = torch.bmm(left_4d.to(torch.float16),
                    right_4d.to(torch.float16).transpose(-1, -2)).to(torch.float32)

    # [bs*H, N, N] -> [bs, N, N, H]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1)

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


_trimul_compiled = torch.compile(_trimul_core, mode="default")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]

    # LayerNorm on input (outside compiled region — proven best)
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])
    # x is [bs, N, N, dim] — pass as 4D tensor, no Python int args for shapes

    # Build fused weight per-call (no caching)
    fused_weight = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0)  # [5*H, dim]

    return _trimul_compiled(
        x, mask, fused_weight,
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

---

## Experiment #26 — 2026-06-18 22:24:24 UTC ❌ DISCARD

**Hypothesis:** Confirmed and re-wrote experiment #21's exact code unchanged. The file already contained the correct code; this write confirms it as the final submission.

**Result:** 3333.25 μs

**Kernel code:**
```python
"""
Optimized TriMul submission — torch.compile default mode + tensor-shape-derived dims
(no Python integer args bs/N/hidden_dim) + per-call fused GEMM + fp16 bmm.
This is experiment #21 — the proven best structure at 3,331 μs geomean.
"""

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _trimul_core(x_4d, mask, fused_weight, w_norm2, b_norm2, w_out):
    """
    x_4d:        [bs, N, N, dim]  — normalized input (LayerNorm applied outside)
    mask:        [bs, N, N]
    fused_weight:[5*H, dim]
    """
    bs         = x_4d.shape[0]
    N          = x_4d.shape[1]
    dim        = x_4d.shape[3]
    hidden_dim = fused_weight.shape[0] // 5
    M          = bs * N * N

    x_flat    = x_4d.reshape(M, dim)
    mask_flat = mask.reshape(M, 1)

    # Single fused GEMM: [M, dim] x [5*H, dim]^T -> [M, 5*H]
    all_proj = F.linear(x_flat, fused_weight)
    lp, rp, lg, rg, og = all_proj.split(hidden_dim, dim=-1)

    left  = lp * lg.sigmoid() * mask_flat    # [M, H]
    right = rp * rg.sigmoid() * mask_flat    # [M, H]

    # Reshape for batched matmul: [bs*H, N, N]
    left_4d  = left.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)
    right_4d = right.reshape(bs, N, N, hidden_dim).permute(0, 3, 1, 2).reshape(bs * hidden_dim, N, N)

    # fp16 bmm for throughput on the dominant matmul
    out = torch.bmm(left_4d.to(torch.float16),
                    right_4d.to(torch.float16).transpose(-1, -2)).to(torch.float32)

    # [bs*H, N, N] -> [bs, N, N, H]
    out = out.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1)

    out = F.layer_norm(out, [hidden_dim], weight=w_norm2, bias=b_norm2)
    out = out * og.sigmoid().reshape(bs, N, N, hidden_dim)
    out = F.linear(out, w_out)
    return out


_trimul_compiled = torch.compile(_trimul_core, mode="default")


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]

    # LayerNorm on input (outside compiled region — proven best)
    x = F.layer_norm(input_tensor, [dim],
                     weight=weights['norm.weight'],
                     bias=weights['norm.bias'])
    # x is [bs, N, N, dim] — pass as 4D tensor, no Python int args for shapes

    # Build fused weight per-call (no caching)
    fused_weight = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0)  # [5*H, dim]

    return _trimul_compiled(
        x, mask, fused_weight,
        weights['to_out_norm.weight'],
        weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )

```

