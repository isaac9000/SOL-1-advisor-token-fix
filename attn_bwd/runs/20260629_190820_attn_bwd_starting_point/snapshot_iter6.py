"""
Optimized attention-backward kernel using torch.compile for BMMs +
a custom Triton kernel that fuses the softmax-backward elementwise chain:
  1. dropout mask + scale
  2. dP * P row-sum reduction  
  3. final P * (dP - row_sum)
All in one pass, reading dP_dropped, dropout_mask, attn_weights once
and writing grad_attn_scores once.

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


@triton.jit
def softmax_bwd_kernel(
    dP_dropped_ptr,   # [B80, sq, skv]  bfloat16
    dropout_mask_ptr, # [B80, sq, skv]  bool (uint8)
    P_ptr,            # [B80, sq, skv]  bfloat16
    dS_ptr,           # [B80, sq, skv]  bfloat16  (output)
    sq: tl.constexpr,
    skv: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """
    Each program handles one row: (batch*head, row_idx).
    Streams over the seq_kv dimension in tiles to compute the row sum,
    then writes dS in a second pass.
    """
    # pid 0 -> row index in [B80 * sq]
    row_id = tl.program_id(0)
    b80_idx = row_id // sq
    q_idx = row_id % sq

    row_offset = b80_idx * sq * skv + q_idx * skv

    # ---- First pass: compute row_sum = sum(dP * P) over skv ----
    row_sum = tl.zeros([1], dtype=tl.float32)

    for start in tl.range(0, skv, BLOCK_SKV):
        kv_ids = start + tl.arange(0, BLOCK_SKV)
        mask = kv_ids < skv
        offsets = row_offset + kv_ids

        dp_dropped = tl.load(dP_dropped_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        dmask = tl.load(dropout_mask_ptr + offsets, mask=mask, other=0).to(tl.float32)
        p = tl.load(P_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        dp = dp_dropped * dmask * scale
        row_sum += tl.sum(dp * p, axis=0)

    # ---- Second pass: compute dS = P * (dP - row_sum) and write ----
    for start in tl.range(0, skv, BLOCK_SKV):
        kv_ids = start + tl.arange(0, BLOCK_SKV)
        mask = kv_ids < skv
        offsets = row_offset + kv_ids

        dp_dropped = tl.load(dP_dropped_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        dmask = tl.load(dropout_mask_ptr + offsets, mask=mask, other=0).to(tl.float32)
        p = tl.load(P_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        dp = dp_dropped * dmask * scale
        ds = p * (dp - row_sum)

        tl.store(dS_ptr + offsets, ds.to(tl.bfloat16), mask=mask)


def fused_softmax_bwd(dP_dropped, dropout_mask, attn_weights, attention_dropout):
    """
    Fused softmax backward + dropout scaling.
    All inputs/output shaped [B80, sq, skv].
    """
    B80, sq, skv = dP_dropped.shape
    dS = torch.empty_like(dP_dropped)

    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Choose block size for skv dimension
    # Power of 2, at least covers common skv values
    if skv <= 64:
        BLOCK_SKV = 64
    elif skv <= 128:
        BLOCK_SKV = 128
    elif skv <= 256:
        BLOCK_SKV = 256
    elif skv <= 512:
        BLOCK_SKV = 512
    elif skv <= 1024:
        BLOCK_SKV = 1024
    else:
        BLOCK_SKV = 2048

    total_rows = B80 * sq
    grid = (total_rows,)

    softmax_bwd_kernel[grid](
        dP_dropped,
        dropout_mask,
        attn_weights,
        dS,
        sq=sq,
        skv=skv,
        scale=scale,
        BLOCK_SKV=BLOCK_SKV,
    )
    return dS


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

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
    #  reshape -> [bs, 80, sq, skv]
    # ------------------------------------------------------------------ #
    dO_grouped = dO.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM)
    dO_for_dP = dO_grouped.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS * seq_q, HEAD_DIM)

    V_flat_t = value_states.reshape(bs * NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM).transpose(-2, -1)

    dP_dropped_grouped = torch.bmm(dO_for_dP, V_flat_t)  # [B8, 10*sq, skv] bf16

    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv) \
                                   .reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ------------------------------------------------------------------ #
    #  Fused softmax backward via Triton kernel
    #  Reads dP_dropped, dropout_mask, attn_weights once; writes dS once
    # ------------------------------------------------------------------ #
    # Flatten to [B80, sq, skv] for the kernel
    dP_dropped_flat = dP_dropped.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv).contiguous()
    dropout_mask_flat = dropout_mask.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv).contiguous()
    attn_weights_flat = attn_weights.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv).contiguous()

    dS_flat = fused_softmax_bwd(dP_dropped_flat, dropout_mask_flat, attn_weights_flat, attention_dropout)
    dS = dS_flat.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ------------------------------------------------------------------ #
    #  Compute dV = attn_weights_dropped^T @ dO  (grouped, no V expansion)
    #
    #  attn_weights_dropped: [bs, 80, sq, skv] -> [bs*80, sq, skv]
    #  dO: [bs, 80, sq, d] -> [bs*80, sq, d]
    #  BMM: [B80, skv, sq] @ [B80, sq, d] -> [B80, skv, d]
    #  Sum over 10 groups: [B8, 10, skv, d] -> [B8, skv, d]
    # ------------------------------------------------------------------ #
    aw_dropped_flat = attn_weights_dropped.reshape(bs * NUM_ATTENTION_HEADS, seq_q, seq_kv)
    dO_flat_kv = dO_grouped.reshape(bs * NUM_ATTENTION_HEADS, seq_q, HEAD_DIM)

    dV_flat = torch.bmm(aw_dropped_flat.transpose(-2, -1), dO_flat_kv)  # [B80, skv, d] bf16

    dV = dV_flat.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS, seq_kv, HEAD_DIM).sum(dim=1)
    dV = dV.reshape(bs, NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV
