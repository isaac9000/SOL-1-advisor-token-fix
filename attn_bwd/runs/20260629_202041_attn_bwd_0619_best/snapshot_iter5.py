"""
Optimized attention-backward kernel using torch.compile with max-autotune.
- Wraps the entire computation (both BMMs + softmax backward elementwise ops)
  in a torch.compile'd function.
- Removes manual dual-stream orchestration and custom Triton kernel.
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
    dO_in,              # [bs, seq_q, 80, 128]  bfloat16
    attn_weights,       # [bs, 80, seq_q, seq_kv] bfloat16
    attn_weights_dropped,  # [bs, 80, seq_q, seq_kv] bfloat16
    value_states,       # [bs, 8, seq_kv, 128]  bfloat16
    dropout_mask,       # [bs, 80, seq_q, seq_kv] bool
    scale,              # scalar float: 1/(1-p_drop)
    bs, seq_q, seq_kv,
    n_groups,           # 10
    n_kv_heads,         # 8
    n_heads,            # 80
):
    # -----------------------------------------------------------------
    # Reshape dO: [bs, seq_q, 80, 128] -> [bs*8, 10*seq_q, 128]
    # -----------------------------------------------------------------
    dO = dO_in.permute(0, 2, 1, 3)  # [bs, 80, seq_q, 128]
    dO_groups = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # -----------------------------------------------------------------
    # GQA expand value_states: [bs,8,seq_kv,128] -> [bs*8,seq_kv,128]
    # -----------------------------------------------------------------
    vs_grouped = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)

    # -----------------------------------------------------------------
    # dP = dO @ V^T: [bs*8, 10*seq_q, seq_kv]
    # -----------------------------------------------------------------
    dP_groups = torch.bmm(dO_groups, vs_grouped.transpose(-2, -1))  # bf16

    # Reshape to [bs, 80, seq_q, seq_kv]
    dP = dP_groups.reshape(bs, n_heads, seq_q, seq_kv)

    # -----------------------------------------------------------------
    # Apply dropout mask and scale
    # -----------------------------------------------------------------
    dP_dropped = dP * dropout_mask.to(dP.dtype) * scale

    # -----------------------------------------------------------------
    # Softmax backward: dS = P * (dP - (dP*P).sum(-1, keepdim=True))
    # -----------------------------------------------------------------
    P = attn_weights
    dP_P = (dP_dropped * P).sum(-1, keepdim=True)
    dS = P * (dP_dropped - dP_P)

    # -----------------------------------------------------------------
    # dV = attn_weights_dropped^T @ dO: [bs*8, seq_kv, 128]
    # -----------------------------------------------------------------
    attn_groups = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    dV_groups = torch.bmm(attn_groups.transpose(-2, -1), dO_groups)  # [bs*8, seq_kv, 128]
    dV = dV_groups.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS.to(torch.bfloat16), dV.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Cache compiled functions keyed by (bs, seq_q, seq_kv)
# ---------------------------------------------------------------------------
_compiled_cache = {}

def _get_compiled_fn(bs, seq_q, seq_kv):
    key = (bs, seq_q, seq_kv)
    if key not in _compiled_cache:
        compiled = torch.compile(
            _attn_bwd_core,
            mode="max-autotune",
            fullgraph=True,
        )
        _compiled_cache[key] = compiled
    return _compiled_cache[key]


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    compiled_fn = _get_compiled_fn(bs, seq_q, seq_kv)

    dS, dV = compiled_fn(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        scale,
        bs, seq_q, seq_kv,
        n_groups, n_kv_heads, n_heads,
    )

    return dS, dV
