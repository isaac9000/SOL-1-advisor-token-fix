"""
Optimized attention-backward kernel — torch.compile + GQA-aware BMMs.

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
    Optimized attention backward using GQA structure in both BMMs:

    - dP BMM: reshape dO to [bs*8, 10*sq, d] and use value_states as [bs*8, skv, d]
              → single batched GEMM [bs*8, 10*sq, d] @ [bs*8, d, skv] = [bs*8, 10*sq, skv]
              → avoids materializing [bs,80,skv,d] expanded tensor

    - dV BMM: reshape attn_weights_dropped to [bs*8, 10*sq, skv] and dO to [bs*8, 10*sq, d]
              → single batched GEMM [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] = [bs*8, skv, d]
              → directly gives dV without group-sum (mathematically equivalent)

    - BMMs in bfloat16, only softmax-backward in float32
    - torch.compile fuses elementwise ops
    """
    bs     = dO_in.shape[0]
    seq_q  = dO_in.shape[1]
    seq_kv = value_states.shape[2]
    n_kv   = NUM_KEY_VALUE_HEADS   # 8
    n_g    = N_GROUPS              # 10
    d      = HEAD_DIM              # 128

    # [bs, sq, 80, d] -> [bs, 80, sq, d] -> contiguous bfloat16
    dO = dO_in.transpose(1, 2).contiguous()  # [bs, 80, sq, d] bfloat16

    # ── dP computation: dO @ V^T  (GQA-aware, no expansion) ──────────────────
    # Reshape dO: [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    dO_reshaped = dO.reshape(bs, n_kv, n_g, seq_q, d).reshape(bs * n_kv, n_g * seq_q, d)

    # value_states: [bs, 8, skv, d] -> [bs*8, skv, d]
    vs_flat = value_states.reshape(bs * n_kv, seq_kv, d)

    # BMM: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]  (bfloat16)
    dP_dropped_flat = torch.bmm(dO_reshaped, vs_flat.transpose(-2, -1))  # [bs*8, 10*sq, skv]

    # Reshape back: [bs*8, 10*sq, skv] -> [bs, 80, sq, skv]
    dP_dropped = dP_dropped_flat.reshape(bs, n_kv * n_g, seq_q, seq_kv)

    # Dropout backward (scale, in float32 for softmax bwd)
    scale = 1.0 / (1.0 - attention_dropout)
    dP = dP_dropped.to(torch.float32) * dropout_mask.to(torch.float32) * scale

    # ── Softmax backward (float32) ────────────────────────────────────────────
    P = attn_weights.to(torch.float32)
    dPP = dP * P
    dS = P * (dP - dPP.sum(dim=-1, keepdim=True))
    dS = dS.to(torch.bfloat16)

    # ── dV computation: GQA-aware, no expansion, no group-sum ─────────────────
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs*8, 10*sq, skv]
    Pd_flat = attn_weights_dropped.reshape(bs * n_kv, n_g * seq_q, seq_kv)  # [bs*8, 10*sq, skv]

    # dO: [bs*8, 10*sq, d] already computed as dO_reshaped
    # BMM: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]  (bfloat16)
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_reshaped)  # [bs*8, skv, d]

    # Reshape to [bs, 8, skv, d]
    dV = dV_flat.reshape(bs, n_kv, seq_kv, d).to(torch.bfloat16)

    return dS, dV


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    return _attn_bwd_compiled(
        grad_attn_output, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, attention_dropout
    )
