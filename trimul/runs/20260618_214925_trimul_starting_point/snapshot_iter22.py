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
