"""
Hybrid attention-backward kernel: GQA-native cuBLAS BMMs + fused Triton softmax-bwd.

Eliminates the 10× V expansion copy by using grouped BMM with broadcasting.

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
# Fused Triton elementwise softmax-backward kernel:
#   Input:  dP [N_rows, skv] bfloat16  (flattened to 2D)
#           P  [N_rows, skv] bfloat16
#   Output: dS [N_rows, skv] bfloat16
#
#   For each row: dS = P * (dP - sum(dP * P))
#   2-pass: pass1 accumulates row_sum in registers, pass2 writes dS.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,             # [N_rows, skv]  bfloat16
    P_ptr,              # [N_rows, skv]  bfloat16
    dS_ptr,             # [N_rows, skv]  bfloat16  output
    skv,
    BLOCK_SKV: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)
    row_offset = row * skv

    skv_tiles = tl.cdiv(skv, BLOCK_SKV)

    # Pass 1: accumulate row_sum = sum_k(dP[row,k] * P[row,k])
    row_sum = tl.zeros([1], dtype=tl.float32)

    for tile_idx in range(skv_tiles):
        k_offs = tile_idx * BLOCK_SKV + tl.arange(0, BLOCK_SKV)
        k_mask = k_offs < skv

        dP_tile = tl.load(dP_ptr + row_offset + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        P_tile  = tl.load(P_ptr  + row_offset + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        row_sum += tl.sum(dP_tile * P_tile, axis=0)

    # Pass 2: write dS = P * (dP - row_sum)
    for tile_idx in range(skv_tiles):
        k_offs = tile_idx * BLOCK_SKV + tl.arange(0, BLOCK_SKV)
        k_mask = k_offs < skv

        dP_tile = tl.load(dP_ptr + row_offset + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        P_tile  = tl.load(P_ptr  + row_offset + k_offs, mask=k_mask, other=0.0).to(tl.float32)

        dS_tile = P_tile * (dP_tile - row_sum)
        tl.store(dS_ptr + row_offset + k_offs, dS_tile.to(tl.bfloat16), mask=k_mask)


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

    # --- Step 1: Make dO contiguous in [bs, 80, sq, 128] layout (one copy, reused twice) ---
    dO_c = grad_attn_output.transpose(1, 2).contiguous()  # [bs, 80, sq, 128] bf16

    # --- Step 2: GQA-native dP BMM — NO V expansion ---
    # Reshape dO: [bs, 80, sq, 128] -> [bs, 8, 10, sq, 128]
    dO_grouped = dO_c.view(bs, n_kv_heads, n_groups, seq_q, head_dim)

    # value_states: [bs, 8, skv, 128] -> unsqueeze to [bs, 8, 1, 128, skv] for broadcast
    V_t = value_states.transpose(-2, -1).unsqueeze(2)  # [bs, 8, 1, 128, skv]

    # Grouped matmul with broadcasting over n_groups:
    # [bs, 8, 10, sq, 128] @ [bs, 8, 1, 128, skv] -> [bs, 8, 10, sq, skv]
    # PyTorch broadcasts V across the 10 groups — V is read once per kv-head, not 10×
    dP_grouped = torch.matmul(dO_grouped, V_t)  # [bs, 8, 10, sq, skv] bf16

    # Reshape to [bs, 80, sq, skv]
    dP = dP_grouped.reshape(bs, n_heads, seq_q, seq_kv)

    # --- Step 3: Dropout backward ---
    dropout_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0
    dP = dP * dropout_mask * dropout_scale

    # --- Step 4: Fused Triton softmax backward ---
    # Ensure contiguous for row-stride kernel
    dP_c = dP.contiguous()
    P_c  = attn_weights.contiguous()
    grad_attn_scores = torch.empty_like(dP_c)

    n_rows   = bs * n_heads * seq_q
    BLOCK_SKV = 512

    softmax_bwd_kernel[(n_rows,)](
        dP_c, P_c, grad_attn_scores,
        seq_kv,
        BLOCK_SKV,
    )

    # --- Step 5: GQA-native dV BMM — NO Pd expansion ---
    # Reshape attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    Pd_c = attn_weights_dropped.contiguous()
    Pd_grouped = Pd_c.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)

    # dV: [bs, 8, 10, skv, sq] @ [bs, 8, 10, sq, 128] -> [bs, 8, 10, skv, 128]
    # dO_grouped: [bs, 8, 10, sq, 128]
    dV_grouped = torch.matmul(Pd_grouped.transpose(-2, -1), dO_grouped)  # [bs, 8, 10, skv, 128]

    # GQA reduction: sum over groups
    grad_value_states = dV_grouped.sum(dim=2).to(torch.bfloat16)  # [bs, 8, skv, 128]

    return grad_attn_scores, grad_value_states
