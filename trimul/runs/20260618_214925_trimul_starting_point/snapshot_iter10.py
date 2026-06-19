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
