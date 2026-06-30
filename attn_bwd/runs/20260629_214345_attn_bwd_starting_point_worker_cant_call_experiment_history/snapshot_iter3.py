"""
Optimized attention-backward kernel — torch.compile + GQA-avoiding dV path.

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


@torch.compile(mode="max-autotune", fullgraph=True)
def _attn_bwd_compiled(dO_in, attn_weights, attn_weights_dropped,
                        value_states, dropout_mask, attention_dropout):
    """
    Optimized attention backward:
    - GQA dV: reshape + einsum avoids full [bs,80,skv,d] expansion
    - Contiguous tensors for BLAS efficiency
    - torch.compile fuses elementwise ops
    """
    bs     = dO_in.shape[0]
    seq_q  = dO_in.shape[1]
    seq_kv = value_states.shape[2]
    n_kv   = NUM_KEY_VALUE_HEADS
    n_g    = N_GROUPS
    d      = HEAD_DIM

    # [bs, sq, 80, d] -> [bs, 80, sq, d], float32 for numerical stability
    dO = dO_in.transpose(1, 2).contiguous().to(torch.float32)  # [bs, 80, sq, d]

    # ── dP computation: dO @ V^T ──────────────────────────────────────────────
    # Expand value_states: [bs, 8, skv, d] -> [bs, 80, skv, d]
    vs_exp = value_states[:, :, None, :, :].expand(
        bs, n_kv, n_g, seq_kv, d
    ).reshape(bs, n_kv * n_g, seq_kv, d).contiguous().to(torch.float32)

    # dP_dropped: [bs, 80, sq, skv]
    dP_dropped = torch.matmul(dO, vs_exp.transpose(-2, -1))

    # Dropout backward
    scale = 1.0 / (1.0 - attention_dropout)
    dP = dP_dropped * dropout_mask.to(torch.float32) * scale

    # ── Softmax backward ──────────────────────────────────────────────────────
    P = attn_weights.to(torch.float32)
    dPP = dP * P
    dS = P * (dP - dPP.sum(dim=-1, keepdim=True))
    dS = dS.to(torch.bfloat16)

    # ── dV computation via GQA-aware grouped einsum ───────────────────────────
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    Pd = attn_weights_dropped.to(torch.float32).reshape(bs, n_kv, n_g, seq_q, seq_kv)
    # dO: [bs, 80, sq, d] -> [bs, 8, 10, sq, d]
    dO_grouped = dO.reshape(bs, n_kv, n_g, seq_q, d)

    # dV[b,g,s,d] = sum_h sum_q Pd[b,g,h,q,s] * dO_grouped[b,g,h,q,d]
    # einsum: 'bghqs,bghqd->bgsd'
    # Use bmm for efficiency: reshape to [bs*8, 10*skv, sq] @ [bs*8, 10*sq, d]?
    # Actually use einsum which torch.compile will optimize:
    dV = torch.einsum('bghqs,bghqd->bgsd', Pd, dO_grouped)
    dV = dV.to(torch.bfloat16)

    return dS, dV


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    return _attn_bwd_compiled(
        grad_attn_output, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, attention_dropout
    )
