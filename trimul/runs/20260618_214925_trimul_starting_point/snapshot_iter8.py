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
