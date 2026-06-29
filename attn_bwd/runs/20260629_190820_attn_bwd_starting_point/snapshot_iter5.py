"""
Optimized attention-backward kernel using torch.compile with restructured
GQA computation that AVOIDS materializing the expanded V tensor.

For dP: reshape dO [bs, 80, sq, d] -> [bs*8, 10*sq, d], V [bs, 8, skv, d] -> [bs*8, d, skv],
        single BMM -> [bs*8, 10*sq, skv] -> reshape [bs, 80, sq, skv].
For dV: grouped BMM on [bs*8*10, sq, skv] x [bs*8*10, sq, d] -> sum over groups.

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


def _attn_backward_compiled(
    grad_attn_output,      # [bs, sq, 80, 128]     bf16
    attn_weights,          # [bs, 80, sq, skv]     bf16
    attn_weights_dropped,  # [bs, 80, sq, skv]     bf16
    value_states,          # [bs, 8, skv, 128]     bf16
    dropout_mask,          # [bs, 80, sq, skv]     bool
    attention_dropout,     # scalar float
):
    bs = grad_attn_output.shape[0]
    seq_q = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, sq, 80, d] -> [bs, 80, sq, d]
    dO = grad_attn_output.transpose(1, 2)  # bf16, [bs, 80, sq, d]

    # ------------------------------------------------------------------ #
    #  Compute dP_dropped = dO @ V^T  WITHOUT materializing expanded V
    #
    #  dO: [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    #  V:  [bs, 8, skv, d] -> [bs*8, d, skv]
    #  BMM: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
    #  reshape -> [bs, 8, 10, sq, skv] -> [bs, 80, sq, skv]
    # ------------------------------------------------------------------ #

    # Reshape dO for grouped BMM: [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    dO_grouped = dO.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM)
    dO_for_dP = dO_grouped.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS * seq_q, HEAD_DIM)  # [B8, 10*sq, d]

    # V: [bs, 8, skv, d] -> [bs*8, d, skv]
    V_flat_t = value_states.reshape(bs * NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM).transpose(-2, -1)  # [B8, d, skv]

    # dP_dropped grouped BMM: [B8, 10*sq, d] @ [B8, d, skv] -> [B8, 10*sq, skv]
    dP_dropped_grouped = torch.bmm(dO_for_dP, V_flat_t)  # [B8, 10*sq, skv] bf16

    # Reshape to [bs, 80, sq, skv]
    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv) \
                                   .reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ------------------------------------------------------------------ #
    #  Dropout backward: mask and scale
    # ------------------------------------------------------------------ #
    if attention_dropout > 0.0:
        scale = 1.0 / (1.0 - attention_dropout)
        dP = dP_dropped * dropout_mask * scale  # bf16
    else:
        dP = dP_dropped  # bf16

    # ------------------------------------------------------------------ #
    #  Softmax backward: dS = P * (dP - sum(dP * P, dim=-1, keepdim=True))
    #  Do in float32 for numerical stability
    # ------------------------------------------------------------------ #
    P_flat = attn_weights.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv).float()
    dP_flat_f32 = dP.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv).float()
    dPP = dP_flat_f32 * P_flat                                               # [B80, sq, skv]
    dS_flat = P_flat * (dP_flat_f32 - dPP.sum(dim=-1, keepdim=True))        # [B80, sq, skv]
    dS = dS_flat.to(torch.bfloat16).reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ------------------------------------------------------------------ #
    #  Compute dV = attn_weights_dropped^T @ dO  (grouped, no V expansion)
    #
    #  attn_weights_dropped: [bs, 80, sq, skv] -> [bs*8, 10, sq, skv]
    #                                           -> [bs*8*10, sq, skv]
    #  dO: [bs, 8, 10, sq, d] -> [bs*8*10, sq, d]
    #  BMM: [B80, skv, sq] @ [B80, sq, d] -> [B80, skv, d]
    #  Sum over 10 groups: [B8, 10, skv, d] -> sum -> [B8, skv, d] -> [bs, 8, skv, d]
    # ------------------------------------------------------------------ #
    aw_dropped_flat = attn_weights_dropped.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv)  # [B80, sq, skv]
    dO_flat_kv = dO_grouped.reshape(bs * NUM_ATTENTION_HEADS, seq_q, HEAD_DIM)              # [B80, sq, d]

    # dV per-group in bf16: [B80, skv, d]
    dV_flat = torch.bmm(aw_dropped_flat.transpose(-2, -1), dO_flat_kv)  # [B80, skv, d] bf16

    # Sum over 10 groups: [B8, 10, skv, d] -> sum -> [B8, skv, d]
    dV = dV_flat.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS, seq_kv, HEAD_DIM).sum(dim=1)   # [B8, skv, d]
    dV = dV.reshape(bs, NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM).to(torch.bfloat16)

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
