"""
Optimized attention-backward using torch.compile with GQA-restructured GEMMs.

Key optimizations:
1. GQA structure exploited at GEMM level: instead of expanding [bs,8,skv,d] -> [bs,80,skv,d]
   and doing a bs*80-batch GEMM, reshape dO to [bs*8, 10, sq, d] and V as [bs*8, skv, d]
   so we get clean batched GEMMs with batch=bs*8 (not bs*80).
2. torch.compile with mode="max-autotune-no-cudagraphs" for static shape specialization.
3. avoid expand+reshape intermediates by working with proper batch structure.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool

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
    grad_attn_output,     # [bs, sq, 80, 128]  bf16
    attn_weights,         # [bs, 80, sq, skv]  bf16
    attn_weights_dropped, # [bs, 80, sq, skv]  bf16
    value_states,         # [bs,  8, skv, 128] bf16
    dropout_mask,         # [bs, 80, sq, skv]  bool
    attention_dropout,    # float scalar
):
    bs  = grad_attn_output.shape[0]
    sq  = grad_attn_output.shape[1]
    skv = value_states.shape[2]

    # dO: [bs, 80, sq, 128] float32
    dO = grad_attn_output.transpose(1, 2).to(torch.float32)

    # ---- GEMM 1: dP_dropped = dO @ V^T ----
    # Reshape to exploit GQA: batch over (bs*8), with 10 groups broadcasting V
    # dO_r: [bs*8, 10, sq, 128]
    dO_r = dO.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS, sq, HEAD_DIM)
    # V_r:  [bs*8, 1, skv, 128]  -> broadcast over groups
    V_r = value_states.reshape(bs * NUM_KEY_VALUE_HEADS, 1, skv, HEAD_DIM).to(torch.float32)
    # matmul: [bs*8, 10, sq, 128] @ [bs*8, 1, 128, skv] -> [bs*8, 10, sq, skv]
    dP_dropped = torch.matmul(dO_r, V_r.transpose(-2, -1))  # [bs*8, 10, sq, skv]
    # Reshape back to [bs, 80, sq, skv]
    dP_dropped = dP_dropped.reshape(bs, NUM_ATTENTION_HEADS, sq, skv)

    # ---- Elementwise: dropout backward + softmax backward ----
    inv_keep = 1.0 / (1.0 - attention_dropout)
    dP = dP_dropped * dropout_mask.to(torch.float32) * inv_keep

    P = attn_weights.to(torch.float32)
    # Di = sum(dP * P, dim=-1, keepdim=True)
    Di = (dP * P).sum(dim=-1, keepdim=True)
    dS = P * (dP - Di)
    dS = dS.to(torch.bfloat16)

    # ---- GEMM 2: dV = Pd^T @ dO ----
    # Pd = attn_weights_dropped: [bs, 80, sq, skv]
    # dO: [bs, 80, sq, 128]
    # Result: [bs, 80, skv, 128] -> sum over groups -> [bs, 8, skv, 128]
    #
    # Exploit GQA: reshape to [bs*8, 10, sq, skv] and [bs*8, 10, sq, 128]
    Pd_r = attn_weights_dropped.to(torch.float32).reshape(
        bs * NUM_KEY_VALUE_HEADS, N_GROUPS, sq, skv
    )
    # Sum over groups first to get [bs*8, sq, skv] before GEMM
    # Actually: dV = sum_g (Pd_g^T @ dO_g)  for each KV-head group
    # = [bs*8, skv, sq] @ [bs*8, sq, 128] with sum over 10 groups
    # Reshape dO_r already: [bs*8, 10, sq, 128]
    # Do batched matmul: [bs*8, 10, skv, sq] @ [bs*8, 10, sq, 128] -> [bs*8, 10, skv, 128]
    # then sum over groups (dim=1)
    dV_exp = torch.matmul(Pd_r.transpose(-2, -1), dO_r)  # [bs*8, 10, skv, 128]
    dV = dV_exp.sum(dim=1)  # [bs*8, skv, 128]
    dV = dV.reshape(bs, NUM_KEY_VALUE_HEADS, skv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV


# Compile with max-autotune for static shape specialization
_compiled_attn_backward = torch.compile(
    _attn_backward_impl,
    mode="max-autotune-no-cudagraphs",
    dynamic=False,
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
