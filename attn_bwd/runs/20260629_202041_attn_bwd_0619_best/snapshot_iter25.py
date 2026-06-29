"""
Optimized attention-backward kernel using torch.compile with reduce-overhead mode.
- Core computation compiled with torch.compile(mode="reduce-overhead", dynamic=True)
- Warm-up at module load time with representative tensor shapes to pre-compile
- Falls back gracefully if compile fails
- Full pipeline: dO permute, both BMMs in GQA-reshaped form, softmax-backward, dV group-sum
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128

# =========================================================================
# Core computation function (to be compiled)
# =========================================================================

def _core_attention_backward(
    grad_attn_output,   # [bs, sq, 80, 128]
    attn_weights,       # [bs, 80, sq, skv]
    attn_weights_dropped,  # [bs, 80, sq, skv]
    value_states,       # [bs, 8, skv, 128]
    dropout_mask,       # [bs, 80, sq, skv] bool
    scale,              # scalar float
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10

    # Reshape dO to [bs*8, 10*sq, 128]
    dO_groups_flat = (grad_attn_output
                      .reshape(bs, seq_q, n_kv_heads, n_groups, HEAD_DIM)
                      .permute(0, 2, 3, 1, 4)
                      .contiguous()
                      .reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM))

    # Prepare matmul operands (free views)
    vs_flat         = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # dP: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] -> [bs*8, 10*sq, skv]
    dP_groups = torch.bmm(dO_groups_flat, vs_flat.transpose(-2, -1))

    # dV: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
    dV_flat = torch.bmm(attn_groups_flat.transpose(-2, -1), dO_groups_flat)

    # Softmax backward
    total_rows = bs * NUM_ATTENTION_HEADS * seq_q
    dP_flat = dP_groups.reshape(total_rows, seq_kv).float()
    P_flat  = attn_weights.reshape(total_rows, seq_kv).float()
    dm_flat = dropout_mask.reshape(total_rows, seq_kv)

    # Apply dropout mask and scale
    dP = torch.where(dm_flat, dP_flat * scale, torch.zeros_like(dP_flat))

    # Softmax backward: dS = P * (dP - sum(dP * P))
    row_sum = (dP * P_flat).sum(dim=-1, keepdim=True)
    dS = P_flat * (dP - row_sum)

    dS = dS.to(torch.bfloat16).reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # dV group sum: [bs*8, skv, 128] -> [bs, 8, skv, 128]
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV


# Compile the core function with reduce-overhead mode
try:
    _compiled_core = torch.compile(
        _core_attention_backward,
        mode="reduce-overhead",
        dynamic=True,
        fullgraph=True,
    )
except Exception:
    _compiled_core = _core_attention_backward


# =========================================================================
# Warm-up at module load time with representative tensor shapes
# =========================================================================
def _warmup():
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        # Use a small but representative shape for warm-up
        bs, sq, skv = 1, 64, 64
        n_q = NUM_ATTENTION_HEADS
        n_kv = NUM_KEY_VALUE_HEADS

        dummy_dO   = torch.zeros(bs, sq, n_q, HEAD_DIM, dtype=torch.bfloat16, device=device)
        dummy_attn = torch.ones(bs, n_q, sq, skv, dtype=torch.bfloat16, device=device) / skv
        dummy_attn_d = dummy_attn.clone()
        dummy_vs   = torch.zeros(bs, n_kv, skv, HEAD_DIM, dtype=torch.bfloat16, device=device)
        dummy_mask = torch.ones(bs, n_q, sq, skv, dtype=torch.bool, device=device)
        dummy_scale = 1.0 / (1.0 - 0.1)

        # Run two warm-up passes to fully compile
        for _ in range(2):
            _compiled_core(dummy_dO, dummy_attn, dummy_attn_d, dummy_vs, dummy_mask, dummy_scale)

        torch.cuda.synchronize()
    except Exception:
        pass


_warmup()


# =========================================================================
# Entry point
# =========================================================================

def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    return _compiled_core(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        scale,
    )
