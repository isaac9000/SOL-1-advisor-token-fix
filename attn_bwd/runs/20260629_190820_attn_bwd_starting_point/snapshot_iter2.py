"""
Optimized attention-backward kernel using torch.compile with restructured
GQA computation that avoids materializing expanded tensors.

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
DROPOUT_SCALE = 1.0 / (1.0 - 0.1)  # precomputed for attention_dropout=0.1


def _attn_backward_compiled(
    grad_attn_output,   # [bs, sq, 80, 128]  bf16
    attn_weights,       # [bs, 80, sq, skv]  bf16
    attn_weights_dropped,  # [bs, 80, sq, skv]  bf16
    value_states,       # [bs, 8, skv, 128]  bf16
    dropout_mask,       # [bs, 80, sq, skv]  bool
    attention_dropout,  # scalar float
):
    bs = grad_attn_output.shape[0]
    seq_q = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # Transpose grad: [bs, sq, 80, d] -> [bs, 80, sq, d], then reshape to groups
    # [bs, 8, 10, sq, d]
    dO = grad_attn_output.transpose(1, 2).to(torch.float32)
    dO_grouped = dO.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM)

    # Reshape attn weights to grouped layout: [bs, 8, 10, sq, skv]
    attn_weights_grouped = attn_weights.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv)
    attn_weights_dropped_grouped = attn_weights_dropped.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv)
    dropout_mask_grouped = dropout_mask.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv)

    # value_states: [bs, 8, skv, d] -> for matmul we need [bs, 8, 1, d, skv]
    # dO_grouped: [bs, 8, 10, sq, d] @ V^T [bs, 8, 1, d, skv] -> [bs, 8, 10, sq, skv]
    V_t = value_states.to(torch.float32).transpose(-2, -1).unsqueeze(2)  # [bs, 8, 1, d, skv]
    dP_dropped = torch.matmul(dO_grouped, V_t)  # [bs, 8, 10, sq, skv]

    # Dropout backward: scale by mask / (1 - p)
    if attention_dropout > 0.0:
        scale = 1.0 / (1.0 - attention_dropout)
        dP = dP_dropped * dropout_mask_grouped * scale
    else:
        dP = dP_dropped

    # Softmax backward: dS = P * (dP - sum(dP * P, dim=-1, keepdim=True))
    P = attn_weights_grouped.to(torch.float32)
    dPP = dP * P
    dS = P * (dP - dPP.sum(dim=-1, keepdim=True))
    dS = dS.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv).to(torch.bfloat16)

    # dV: attn_weights_dropped_grouped^T @ dO_grouped -> [bs, 8, 10, skv, d]
    # sum over groups (dim=2) -> [bs, 8, skv, d]
    # attn_weights_dropped_grouped: [bs, 8, 10, sq, skv]
    # dO_grouped: [bs, 8, 10, sq, d]
    dV = torch.matmul(
        attn_weights_dropped_grouped.to(torch.float32).transpose(-2, -1),  # [bs, 8, 10, skv, sq]
        dO_grouped  # [bs, 8, 10, sq, d]
    ).sum(dim=2).to(torch.bfloat16)  # [bs, 8, skv, d]

    return dS, dV


# Cache the compiled function to avoid recompilation overhead
_compiled_fn = torch.compile(
    _attn_backward_compiled,
    mode="max-autotune",
    fullgraph=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    return _compiled_fn(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        attention_dropout,
    )
