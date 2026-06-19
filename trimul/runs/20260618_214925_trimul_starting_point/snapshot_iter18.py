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
