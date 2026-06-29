"""
Optimized attention-backward kernel using torch.compile with max-autotune.

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

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10


def _attn_backward_impl(
    grad_attn_output,   # [bs, sq, 80, 128]  bf16
    attn_weights,       # [bs, 80, sq, skv]  bf16
    attn_weights_dropped,  # [bs, 80, sq, skv]  bf16
    value_states,       # [bs, 8, skv, 128]  bf16
    dropout_mask,       # [bs, 80, sq, skv]  bool
    attention_dropout,  # float scalar
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # Transpose grad: [bs, sq, 80, d] -> [bs, 80, sq, d]
    # Make contiguous so cuBLAS sees a properly strided tensor
    dO = grad_attn_output.transpose(1, 2).contiguous()  # bf16, [bs, 80, sq, d]

    # --- Compute dP_dropped = dO @ V^T in bf16 (avoid float32 cast) ---
    # Expand value_states for GQA: [bs, 8, skv, d] -> [bs, 80, skv, d]
    # Make contiguous to avoid stride-0 cuBLAS slow path from expand
    vs_exp = value_states[:, :, None, :, :].expand(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_kv, HEAD_DIM
    ).reshape(bs, NUM_ATTENTION_HEADS, seq_kv, HEAD_DIM).contiguous()

    # dP_dropped = dO @ vs_exp^T  -> [bs, 80, sq, skv]  in bf16
    dP_dropped = torch.matmul(dO, vs_exp.transpose(-2, -1))

    # --- Dropout backward ---
    # dropout_mask is bool; attention_dropout=0.1 => scale = 1/0.9
    dP = dP_dropped * dropout_mask * (1.0 / (1.0 - attention_dropout))

    # --- Softmax backward: dS = P * (dP - sum(dP * P, dim=-1, keepdim)) ---
    P = attn_weights  # bf16
    dPP = dP * P
    dS = P * (dP - dPP.sum(dim=-1, keepdim=True))
    dS = dS.to(torch.bfloat16)

    # --- Compute dV using einsum for grouped contraction ---
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    # dO:                   [bs, 80, sq, d]   -> [bs, 8, 10, sq, d]
    # einsum directly contracts over groups (g) and queries (q):
    # dV[b, g, k, d] = sum_{h, q} awd[b, h, g, q, k] * dO[b, h, g, q, d]
    awd_r = attn_weights_dropped.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv)
    dO_r  = dO.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM)

    # einsum: sum over n_groups (n) and seq_q (q) simultaneously
    # Result: [bs, 8, skv, d]
    dV = torch.einsum('bgnqk,bgnqd->bgkd', awd_r, dO_r).to(torch.bfloat16)

    return dS, dV


# Compile once with max-autotune for best performance on B200
_compiled_attn_backward = torch.compile(
    _attn_backward_impl,
    mode="max-autotune",
    fullgraph=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    return _compiled_attn_backward(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        attention_dropout,
    )
