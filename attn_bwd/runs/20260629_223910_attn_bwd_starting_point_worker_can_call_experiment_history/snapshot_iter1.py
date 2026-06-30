"""
Reference attention-backward kernel — pure PyTorch baseline.

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


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # Expand value_states for GQA: [bs, 8, skv, d] → [bs, 80, skv, d]
    vs_exp = value_states[:, :, None, :, :].expand(
        bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM
    ).reshape(bs, n_heads, seq_kv, HEAD_DIM)

    # 1. Transpose grad: [bs, sq, h, d] → [bs, h, sq, d]  (cast to f32)
    dO = grad_attn_output.transpose(1, 2).to(torch.float32)

    # 2. dP̃ = dO @ V^T  →  [bs, h, sq, skv]
    dP_dropped = torch.matmul(dO, vs_exp.to(torch.float32).transpose(-2, -1))

    # 3. Dropout backward
    if attention_dropout > 0.0:
        dP = dP_dropped * dropout_mask / (1.0 - attention_dropout)
    else:
        dP = dP_dropped

    # 4. Softmax backward: dS = P ⊙ (dP − sum(dP ⊙ P))
    P = attn_weights.to(torch.float32)
    dS = P * (dP - (dP * P).sum(dim=-1, keepdim=True))
    dS = dS.to(torch.bfloat16)

    # 5. dV_exp = P̃^T @ dO  →  [bs, h, skv, d]
    dV_exp = torch.matmul(
        attn_weights_dropped.to(torch.float32).transpose(-2, -1), dO
    )

    # 6. GQA aggregation: sum over groups  →  [bs, 8, skv, d]
    dV = dV_exp.reshape(bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM).sum(dim=2)
    dV = dV.to(torch.bfloat16)

    return dS, dV
