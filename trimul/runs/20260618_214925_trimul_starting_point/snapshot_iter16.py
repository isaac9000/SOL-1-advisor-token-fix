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
