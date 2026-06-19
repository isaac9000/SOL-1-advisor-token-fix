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
