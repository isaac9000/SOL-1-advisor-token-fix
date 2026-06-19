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
