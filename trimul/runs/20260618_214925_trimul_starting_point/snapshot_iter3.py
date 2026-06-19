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
