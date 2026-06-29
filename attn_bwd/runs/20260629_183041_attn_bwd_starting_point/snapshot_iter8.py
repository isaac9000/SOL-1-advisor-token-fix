"""
Optimized attention-backward kernel using a fused Triton elementwise kernel
for the softmax-backward + dropout-backward section, with torch.compile for
the two GEMM operations.

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
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10


# ---------------------------------------------------------------------------
# Triton kernel: fused softmax-backward + dropout-backward
#
# For each row (b, h, q) of shape [seq_kv]:
#   dP = dP_dropped * mask / (1 - p_drop)
#   dS = P * (dP - sum(dP * P))
#
# Grid: (bs * 80 * seq_q,)
# Each program handles one row of length seq_kv.
# ---------------------------------------------------------------------------
@triton.jit
def _softmax_dropout_bwd_kernel(
    dP_dropped_ptr,   # [bs, 80, sq, skv]  bf16
    P_ptr,            # [bs, 80, sq, skv]  bf16
    mask_ptr,         # [bs, 80, sq, skv]  bool (1 byte)
    dS_ptr,           # [bs, 80, sq, skv]  bf16  (output)
    seq_kv: tl.constexpr,
    inv_keep_prob: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one (b,h,q) combination)
    row_id = tl.program_id(0)
    row_start = row_id * seq_kv

    # We process the row in blocks of BLOCK_SIZE
    # First pass: compute sum(dP * P) across the row
    acc = tl.zeros([1], dtype=tl.float32)

    for block_start in range(0, seq_kv, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask_valid = offsets < seq_kv
        ptr = row_start + offsets

        dP_drop = tl.load(dP_dropped_ptr + ptr, mask=mask_valid, other=0.0).to(tl.float32)
        drop_mask = tl.load(mask_ptr + ptr, mask=mask_valid, other=0).to(tl.float32)
        P = tl.load(P_ptr + ptr, mask=mask_valid, other=0.0).to(tl.float32)

        dP = dP_drop * drop_mask * inv_keep_prob
        acc += tl.sum(dP * P, axis=0)

    # Second pass: compute dS = P * (dP - acc)
    for block_start in range(0, seq_kv, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask_valid = offsets < seq_kv
        ptr = row_start + offsets

        dP_drop = tl.load(dP_dropped_ptr + ptr, mask=mask_valid, other=0.0).to(tl.float32)
        drop_mask = tl.load(mask_ptr + ptr, mask=mask_valid, other=0).to(tl.float32)
        P = tl.load(P_ptr + ptr, mask=mask_valid, other=0.0).to(tl.float32)

        dP = dP_drop * drop_mask * inv_keep_prob
        dS = P * (dP - acc)

        tl.store(dS_ptr + ptr, dS.to(tl.bfloat16), mask=mask_valid)


def _softmax_dropout_bwd_triton(dP_dropped, P, dropout_mask, attention_dropout):
    """
    Fused triton kernel for softmax bwd + dropout bwd.
    Inputs:  dP_dropped [bs, 80, sq, skv] bf16 (contiguous)
             P          [bs, 80, sq, skv] bf16 (contiguous)
             dropout_mask [bs, 80, sq, skv] bool (contiguous)
    Output:  dS         [bs, 80, sq, skv] bf16
    """
    bs, nH, sq, skv = dP_dropped.shape
    dS = torch.empty_like(dP_dropped)

    n_rows = bs * nH * sq
    inv_keep_prob = float(1.0 / (1.0 - attention_dropout))

    # Choose BLOCK_SIZE: next power of 2 >= skv, capped at 4096
    BLOCK_SIZE = max(triton.next_power_of_2(skv), 64)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)

    # Ensure inputs are contiguous
    dP_dropped_c = dP_dropped.contiguous()
    P_c = P.contiguous()
    mask_c = dropout_mask.contiguous()

    grid = (n_rows,)
    _softmax_dropout_bwd_kernel[grid](
        dP_dropped_c, P_c, mask_c, dS,
        seq_kv=skv,
        inv_keep_prob=inv_keep_prob,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=min(max(BLOCK_SIZE // 32, 1), 32),
    )
    return dS


# ---------------------------------------------------------------------------
# Compiled matmul helpers (just the GEMMs, no fused elementwise)
# ---------------------------------------------------------------------------

def _matmul1(dO, vs_exp):
    """dP_dropped = dO @ vs_exp^T  -> [bs, 80, sq, skv]"""
    return torch.matmul(dO, vs_exp.transpose(-2, -1))


def _matmul2_and_dV(attn_weights_dropped, dO, bs):
    """dV = sum over groups of attn_weights_dropped^T @ dO"""
    # attn_weights_dropped: [bs, 80, sq, skv]
    # dO:                   [bs, 80, sq, d]
    # dV:                   [bs, 8, skv, d]
    awd_r = attn_weights_dropped.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, dO.shape[2], attn_weights_dropped.shape[3])
    dO_r  = dO.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, dO.shape[2], HEAD_DIM)
    dV = torch.einsum('bgnqk,bgnqd->bgkd', awd_r, dO_r).to(torch.bfloat16)
    return dV


_compiled_matmul1 = torch.compile(_matmul1, mode="max-autotune", fullgraph=True)
_compiled_matmul2_dV = torch.compile(_matmul2_and_dV, mode="max-autotune", fullgraph=True)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # [bs, sq, 80, d] -> [bs, 80, sq, d], contiguous
    dO = grad_attn_output.transpose(1, 2).contiguous()

    # GQA expand: [bs, 8, skv, d] -> [bs, 80, skv, d], contiguous
    vs_exp = value_states[:, :, None, :, :].expand(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_kv, HEAD_DIM
    ).reshape(bs, NUM_ATTENTION_HEADS, seq_kv, HEAD_DIM).contiguous()

    # GEMM 1: dP_dropped = dO @ vs_exp^T  [bs, 80, sq, skv]
    dP_dropped = _compiled_matmul1(dO, vs_exp)

    # Fused Triton: softmax bwd + dropout bwd
    dS = _softmax_dropout_bwd_triton(dP_dropped, attn_weights, dropout_mask, attention_dropout)

    # GEMM 2 + group reduction: dV [bs, 8, skv, d]
    dV = _compiled_matmul2_dV(attn_weights_dropped, dO, bs)

    return dS, dV
