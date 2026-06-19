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
