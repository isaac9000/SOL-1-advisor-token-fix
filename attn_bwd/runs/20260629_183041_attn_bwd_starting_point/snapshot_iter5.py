"""
Optimized attention-backward using torch.compile with GQA-restructured GEMMs.

Key optimizations:
1. GQA structure exploited at GEMM level: instead of expanding [bs,8,skv,d] -> [bs,80,skv,d]
   and doing a bs*80-batch GEMM, reshape dO to [bs*8, 10*sq, d] and V as [bs*8, skv, d]
   so we get clean batched GEMMs with batch=bs*8 (not bs*80).
   The 10*sq stacked layout gives cuBLAS larger M dimension (better tile efficiency).
2. torch.compile with mode="max-autotune-no-cudagraphs" for static shape specialization.
3. GEMM 2 stacks 10 groups into one matmul [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d]
   which automatically sums over groups — no explicit reduction needed.

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
    # Stack 10 groups: dO [bs, 80, sq, 128] -> [bs*8, 10*sq, 128]
    # (reshape via [bs, 8, 10, sq, 128] -> [bs*8, 10*sq, 128])
    dO_stacked = dO.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS * sq, HEAD_DIM)  # [bs, 8, 10*sq, 128]
    dO_stacked = dO_stacked.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS * sq, HEAD_DIM)  # [bs*8, 10*sq, 128]

    # V: [bs, 8, skv, 128] -> [bs*8, skv, 128] -> [bs*8, 128, skv]
    V_r = value_states.reshape(bs * NUM_KEY_VALUE_HEADS, skv, HEAD_DIM).to(torch.float32)

    # GEMM: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] -> [bs*8, 10*sq, skv]
    dP_stacked = torch.matmul(dO_stacked, V_r.transpose(-2, -1))  # [bs*8, 10*sq, skv]

    # Reshape back to [bs, 80, sq, skv]
    dP_dropped = dP_stacked.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, sq, skv)
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
    # Stack: attn_weights_dropped [bs, 80, sq, skv] -> [bs*8, 10*sq, skv]
    Pd_stacked = attn_weights_dropped.to(torch.float32)
    Pd_stacked = Pd_stacked.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS * sq, skv)
    Pd_stacked = Pd_stacked.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS * sq, skv)  # [bs*8, 10*sq, skv]

    # GEMM: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
    # This automatically sums the 10 groups' contributions (equivalent to sum over groups)
    dV = torch.matmul(Pd_stacked.transpose(-2, -1), dO_stacked)  # [bs*8, skv, 128]

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
