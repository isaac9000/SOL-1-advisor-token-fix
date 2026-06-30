"""
Optimized attention-backward kernel using torch.compile + GQA-aware computation.

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


def _attention_backward_impl(
    grad_attn_output, attn_weights, attn_weights_dropped,
    value_states, dropout_mask, attention_dropout
):
    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # 1. Transpose grad: [bs, sq, h, d] -> [bs, h, sq, d]  (cast to f32)
    dO = grad_attn_output.transpose(1, 2).to(torch.float32)
    # contiguous for reshape
    dO = dO.contiguous()

    # Reshape dO to exploit GQA: [bs, 80, sq, d] -> [bs, 8, 10, sq, d]
    dO_grouped = dO.reshape(bs, n_kv_heads, n_groups, seq_q, HEAD_DIM)

    # === Compute dP_dropped avoiding 10x V expansion ===
    # value_states: [bs, 8, skv, d] -> transpose to [bs, 8, d, skv]
    # then unsqueeze for group broadcast: [bs, 8, 1, d, skv]
    vs_f32 = value_states.to(torch.float32)
    vs_T = vs_f32.transpose(-2, -1).unsqueeze(2)  # [bs, 8, 1, d, skv]

    # dP_dropped_grouped: [bs, 8, 10, sq, skv] = [bs,8,10,sq,d] @ [bs,8,1,d,skv]
    dP_dropped_grouped = torch.matmul(dO_grouped, vs_T)
    # reshape to [bs, 80, sq, skv]
    dP_dropped = dP_dropped_grouped.reshape(bs, n_heads, seq_q, seq_kv)

    # 3. Dropout backward
    if attention_dropout > 0.0:
        dP = dP_dropped * dropout_mask / (1.0 - attention_dropout)
    else:
        dP = dP_dropped

    # 4. Softmax backward: dS = P * (dP - sum(dP * P, dim=-1, keepdim=True))
    P = attn_weights.to(torch.float32)
    dPP = dP * P
    dS = P * (dP - dPP.sum(dim=-1, keepdim=True))
    dS = dS.to(torch.bfloat16)

    # === Compute dV in [bs, 8, skv, d] space directly (no 10x expansion) ===
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    P_dropped_grouped = attn_weights_dropped.to(torch.float32).reshape(
        bs, n_kv_heads, n_groups, seq_q, seq_kv
    )
    # dV_grouped = P_dropped_grouped^T @ dO_grouped: [bs, 8, 10, skv, d]
    dV_grouped = torch.matmul(
        P_dropped_grouped.transpose(-2, -1),  # [bs, 8, 10, skv, sq]
        dO_grouped                             # [bs, 8, 10, sq, d]
    )
    # Sum over groups: [bs, 8, 10, skv, d] -> [bs, 8, skv, d]
    dV = dV_grouped.sum(dim=2).to(torch.bfloat16)

    return dS, dV


# Compile the inner function for better performance
_compiled_attention_backward = torch.compile(
    _attention_backward_impl,
    mode="max-autotune",
    fullgraph=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    return _compiled_attention_backward(
        grad_attn_output, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, attention_dropout
    )
