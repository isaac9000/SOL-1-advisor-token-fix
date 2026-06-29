"""
Optimized attention-backward kernel using torch.compile with max-autotune.
- Wraps the entire computation (both BMMs + softmax backward elementwise ops)
  in a torch.compile'd function.
- Uses float32 for numerical correctness.
- Proper GQA grouping: explicitly reshape [bs,80,sq,skv] -> [bs,8,10,sq,skv]
  before any bmm, to allow batching over bs*8 with the 10 groups summed.
- Lets inductor fuse elementwise softmax-backward ops into a single kernel.
- cuBLAS handles both BMMs with its own internal pipelining.

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
import functools

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128

# ---------------------------------------------------------------------------
# Core computation expressed in clean PyTorch ops — to be compiled by inductor
# ---------------------------------------------------------------------------

def _attn_bwd_core(
    dO_in,               # [bs, seq_q, 80, 128]  bfloat16
    attn_weights,        # [bs, 80, seq_q, seq_kv] bfloat16
    attn_weights_dropped,# [bs, 80, seq_q, seq_kv] bfloat16
    value_states,        # [bs, 8, seq_kv, 128]  bfloat16
    dropout_mask,        # [bs, 80, seq_q, seq_kv] bool
    scale,               # scalar float: 1/(1-p_drop)
):
    bs       = dO_in.shape[0]
    seq_q    = dO_in.shape[1]
    seq_kv   = value_states.shape[2]
    n_heads     = 80
    n_kv_heads  = 8
    n_groups    = 10

    # dO: [bs, seq_q, 80, 128] -> [bs, 80, seq_q, 128] -> float32
    dO = dO_in.permute(0, 2, 1, 3).contiguous().float()  # [bs, 80, sq, 128]

    # value_states float32
    vs = value_states.float()  # [bs, 8, skv, 128]

    # GQA: expand vs to [bs, 80, skv, 128]
    # vs[:,k,:,:] is shared by dO heads [k*10 .. k*10+9]
    # vs_exp: [bs, 8, 1, skv, 128] -> [bs, 8, 10, skv, 128] -> [bs, 80, skv, 128]
    vs_exp = vs.unsqueeze(2).expand(bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM)
    vs_exp = vs_exp.reshape(bs * n_heads, seq_kv, HEAD_DIM)   # [bs*80, skv, 128]

    # dO flat for first bmm
    dO_flat = dO.reshape(bs * n_heads, seq_q, HEAD_DIM)       # [bs*80, sq, 128]

    # dP = dO @ V^T: [bs*80, sq, skv]
    dP_flat = torch.bmm(dO_flat, vs_exp.transpose(-2, -1))
    dP = dP_flat.reshape(bs, n_heads, seq_q, seq_kv)          # [bs, 80, sq, skv]

    # Apply dropout mask and scale
    dP_dropped = dP * dropout_mask.float() * scale

    # Softmax backward: dS = P * (dP - (dP*P).sum(-1, keepdim=True))
    P = attn_weights.float()
    dP_P_sum = (dP_dropped * P).sum(-1, keepdim=True)
    dS = P * (dP_dropped - dP_P_sum)

    # dV = attn_weights_dropped^T @ dO:
    # Need [bs, 80, skv, 128] then sum over groups -> [bs, 8, skv, 128]
    attn_flat = attn_weights_dropped.float().reshape(bs * n_heads, seq_q, seq_kv)
    # [bs*80, skv, sq] @ [bs*80, sq, 128] -> [bs*80, skv, 128]
    dV_flat = torch.bmm(attn_flat.transpose(-2, -1), dO_flat)
    # Sum over groups: [bs*80, skv, 128] -> [bs, 8, 10, skv, 128] -> sum -> [bs, 8, skv, 128]
    dV = dV_flat.reshape(bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM).sum(dim=2)

    return dS.to(torch.bfloat16), dV.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Compiled version with max-autotune
# ---------------------------------------------------------------------------
_compiled_fn = torch.compile(
    _attn_bwd_core,
    mode="max-autotune",
    fullgraph=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    dS, dV = _compiled_fn(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        scale,
    )

    return dS, dV
