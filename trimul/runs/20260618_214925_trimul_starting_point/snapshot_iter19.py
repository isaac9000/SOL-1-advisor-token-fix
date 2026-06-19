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
