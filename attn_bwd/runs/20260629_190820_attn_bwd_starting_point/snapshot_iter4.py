"""
Optimized attention-backward kernel using torch.compile with restructured
GQA computation that uses bfloat16 BMMs (flat 3D batched) and float32
only for the elementwise softmax backward accumulation.

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
    #  Compute dP_dropped = dO @ V^T  (both bfloat16, flat 3D BMM)
    #  Need V expanded from [bs,8,skv,d] to [bs,80,skv,d]
    #  We do this without materializing via repeat_interleave then flat BMM
    # ------------------------------------------------------------------ #

    # Expand V: [bs, 8, skv, d] -> [bs, 80, skv, d] using interleave-expand
    # Use reshape trick: [bs,8,1,skv,d] -> broadcast -> [bs,8,10,skv,d] -> [bs,80,skv,d]
    V_exp = value_states.unsqueeze(2).expand(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_kv, HEAD_DIM).reshape(bs, NUM_ATTENTION_HEADS, seq_kv, HEAD_DIM)
    # V_exp: [bs, 80, skv, d] bf16

    # Flatten to 3D: [bs*80, sq, d] and [bs*80, d, skv]
    dO_flat = dO.reshape(bs * NUM_ATTENTION_HEADS, seq_q, HEAD_DIM)           # [B80, sq, d]
    V_flat_t = V_exp.transpose(-2, -1).reshape(bs * NUM_ATTENTION_HEADS, HEAD_DIM, seq_kv)  # [B80, d, skv]

    # dP_dropped in bf16
    dP_dropped_flat = torch.bmm(dO_flat, V_flat_t)  # [B80, sq, skv] bf16

    # ------------------------------------------------------------------ #
    #  Dropout backward: mask and scale
    # ------------------------------------------------------------------ #
    if attention_dropout > 0.0:
        scale = 1.0 / (1.0 - attention_dropout)
        dropout_mask_flat = dropout_mask.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv)
        dP_flat = dP_dropped_flat * dropout_mask_flat * scale  # bf16
    else:
        dP_flat = dP_dropped_flat  # bf16

    # ------------------------------------------------------------------ #
    #  Softmax backward: dS = P * (dP - sum(dP * P, dim=-1, keepdim=True))
    #  Do in float32 for numerical stability
    # ------------------------------------------------------------------ #
    P_flat = attn_weights.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv).float()
    dP_flat_f32 = dP_flat.float()
    dPP = dP_flat_f32 * P_flat                                      # [B80, sq, skv]
    dS_flat = P_flat * (dP_flat_f32 - dPP.sum(dim=-1, keepdim=True))  # [B80, sq, skv]
    dS = dS_flat.to(torch.bfloat16).reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ------------------------------------------------------------------ #
    #  Compute dV = attn_weights_dropped^T @ dO  (grouped, sum over groups)
    #  attn_weights_dropped: [bs, 80, sq, skv] -> grouped [bs, 8, 10, sq, skv]
    #  dO: [bs, 80, sq, d] -> grouped [bs, 8, 10, sq, d]
    #  dV_group[i] = sum_g(aw_dropped[:,i,g,:,:]^T @ dO[:,i,g,:,:])
    #
    #  Flatten groups into batch: merge (bs*8*10) -> flat 3D BMM, then sum groups
    # ------------------------------------------------------------------ #
    aw_dropped_grouped = attn_weights_dropped.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv)  # [B8, 10, sq, skv]
    dO_grouped = dO.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM)                          # [B8, 10, sq, d]

    # Flatten groups into batch dim: [B8*10, sq, skv] and [B8*10, sq, d]
    aw_dropped_flat = aw_dropped_grouped.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv)  # [B80, sq, skv] bf16
    dO_flat_kv = dO_grouped.reshape(bs * NUM_ATTENTION_HEADS, seq_q, HEAD_DIM)             # [B80, sq, d]   bf16

    # dV per-group in bf16: [B80, skv, d]
    dV_flat = torch.bmm(aw_dropped_flat.transpose(-2, -1), dO_flat_kv)  # [B80, skv, d] bf16

    # Sum over 10 groups: [B80, skv, d] -> [B8, 10, skv, d] -> sum -> [B8, skv, d]
    dV = dV_flat.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS, seq_kv, HEAD_DIM).sum(dim=1)  # [B8, skv, d] bf16
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
