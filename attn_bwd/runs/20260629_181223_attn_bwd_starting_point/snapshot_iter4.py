"""
Hybrid attention-backward kernel: cuBLAS BMMs + fused Triton elementwise softmax-bwd.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = 10


# ---------------------------------------------------------------------------
# Fused Triton elementwise kernel:
#   Input:  dP [bs, 80, sq, skv] bfloat16 (result of dO @ V^T with dropout applied)
#           P  [bs, 80, sq, skv] bfloat16
#   Output: dS [bs, 80, sq, skv] bfloat16
#   
#   For each row (bs, h, sq): dS = P * (dP - sum(dP * P))
#   Single pass: read P and dP once (via row tiles), compute row sum, write dS.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,   # [bs, 80, sq, skv]  bfloat16
    P_ptr,    # [bs, 80, sq, skv]  bfloat16
    dS_ptr,   # [bs, 80, sq, skv]  bfloat16  output
    # strides [bs, h, sq, skv]
    stride_b, stride_h, stride_sq, stride_skv,
    # dims
    n_heads: tl.constexpr,
    skv,
    BLOCK_SKV: tl.constexpr,
):
    # Grid: (bs * n_heads * sq,)  — one program per row
    pid = tl.program_id(0)

    # Decode (b, h, sq_idx) from flat pid
    # total rows = bs * n_heads * sq
    row_base = pid * stride_sq  # offset to the start of this row
    # (we'll compute base as pid * stride_sq assuming stride_sq = skv for contiguous)

    skv_tiles = tl.cdiv(skv, BLOCK_SKV)

    # Pass 1: compute row_sum = sum_k(dP[row, k] * P[row, k])
    row_sum = tl.zeros([1], dtype=tl.float32)

    for tile_idx in range(skv_tiles):
        k_offs = tile_idx * BLOCK_SKV + tl.arange(0, BLOCK_SKV)
        k_mask = k_offs < skv

        dP_tile = tl.load(dP_ptr + row_base + k_offs * stride_skv, mask=k_mask, other=0.0).to(tl.float32)
        P_tile  = tl.load(P_ptr  + row_base + k_offs * stride_skv, mask=k_mask, other=0.0).to(tl.float32)
        row_sum += tl.sum(dP_tile * P_tile, axis=0)

    # Pass 2: write dS = P * (dP - row_sum)
    rs = row_sum  # scalar

    for tile_idx in range(skv_tiles):
        k_offs = tile_idx * BLOCK_SKV + tl.arange(0, BLOCK_SKV)
        k_mask = k_offs < skv

        dP_tile = tl.load(dP_ptr + row_base + k_offs * stride_skv, mask=k_mask, other=0.0).to(tl.float32)
        P_tile  = tl.load(P_ptr  + row_base + k_offs * stride_skv, mask=k_mask, other=0.0).to(tl.float32)

        dS_tile = P_tile * (dP_tile - rs)
        tl.store(dS_ptr + row_base + k_offs * stride_skv, dS_tile.to(tl.bfloat16), mask=k_mask)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_heads    = NUM_ATTENTION_HEADS    # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS    # 8
    n_groups   = N_GROUPS               # 10
    head_dim   = HEAD_DIM               # 128

    # --- Step 1: PyTorch transpose (view, no copy) ---
    # grad_attn_output: [bs, sq, 80, 128] -> [bs, 80, sq, 128]
    dO = grad_attn_output.transpose(1, 2)  # non-contiguous view, no copy

    # --- Step 2: GQA expand value_states (view only, no copy) ---
    # [bs, 8, skv, 128] -> [bs, 80, skv, 128]
    vs_exp = value_states[:, :, None, :, :].expand(
        bs, n_kv_heads, n_groups, seq_kv, head_dim
    ).reshape(bs, n_heads, seq_kv, head_dim)
    # reshape of non-contiguous may need contiguous — use as_strided instead
    # Actually expand+reshape on non-contiguous can fail; let's make contiguous for BMM
    # but do it in bf16 to avoid the fp32 upcast cost
    vs_exp_c = vs_exp.contiguous()  # [bs, 80, skv, 128] bf16

    # Make dO contiguous for BMM
    dO_c = dO.contiguous()  # [bs, 80, sq, 128] bf16

    # --- Step 3: cuBLAS BMM: dP = dO @ V^T -> [bs, 80, sq, skv] ---
    # Using bf16 throughout — B200 tensor cores handle bf16 natively
    dP = torch.matmul(dO_c, vs_exp_c.transpose(-2, -1))  # [bs, 80, sq, skv] bf16

    # --- Step 4: Dropout backward ---
    dropout_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0
    # Apply dropout mask: dP = dP * mask * scale (where mask=True means kept)
    dP = dP * dropout_mask * dropout_scale  # bool * bf16 auto-promotes

    # --- Step 5: Fused Triton softmax backward ---
    # dS = P * (dP - sum(dP * P, dim=-1, keepdim=True))
    # dP and attn_weights must be contiguous for the Triton kernel
    dP_c = dP.contiguous()      # should already be contiguous
    P_c  = attn_weights.contiguous()
    grad_attn_scores = torch.empty_like(dP_c)

    # Each program handles one row (bs, h, sq)
    n_rows = bs * n_heads * seq_q
    BLOCK_SKV = 512  # process row in chunks; tune as needed

    softmax_bwd_kernel[(n_rows,)](
        dP_c, P_c, grad_attn_scores,
        # strides
        dP_c.stride(0), dP_c.stride(1), dP_c.stride(2), dP_c.stride(3),
        # dims
        n_heads, seq_kv,
        BLOCK_SKV,
    )

    # --- Step 6: cuBLAS BMM for dV: Pd^T @ dO -> [bs, 80, skv, 128] ---
    attn_weights_dropped_c = attn_weights_dropped.contiguous()
    dV_exp = torch.matmul(
        attn_weights_dropped_c.transpose(-2, -1),  # [bs, 80, skv, sq]
        dO_c                                        # [bs, 80, sq, 128]
    )  # -> [bs, 80, skv, 128]

    # --- Step 7: GQA reduction: sum over groups -> [bs, 8, skv, 128] ---
    grad_value_states = dV_exp.reshape(bs, n_kv_heads, n_groups, seq_kv, head_dim).sum(dim=2).to(torch.bfloat16)

    return grad_attn_scores, grad_value_states
