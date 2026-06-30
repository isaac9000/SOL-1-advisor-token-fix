"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- Keep both BMMs as bfloat16 torch.matmul (cuBLAS-optimized)
- Fused Triton kernel for elementwise dropout-bwd + softmax-bwd pass
  (single-pass when seq_kv <= BLOCK_KV, two-pass otherwise)
- GQA group-sum via torch.sum (float32 accumulation, bf16 output)
- Eliminate redundant .contiguous() calls to avoid memory copies

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


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel 1: fused dropout-bwd + softmax-bwd
#
# For each row (bs * n_heads * sq), length seq_kv:
#   dP = dP_dropped * mask / (1 - p)
#   dS = P * (dP - sum(P * dP))
#
# Single-pass variant: when seq_kv <= BLOCK_KV, load once, compute dot,
# then compute dS in the same pass (data stays in registers).
# Two-pass variant: for larger seq_kv.
#
# Grid: (total_rows,)
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _softmax_bwd_kernel(
    dP_dropped_ptr,    # [total_rows, seq_kv]  bfloat16
    attn_weights_ptr,  # [total_rows, seq_kv]  bfloat16
    dropout_mask_ptr,  # [total_rows, seq_kv]  bool (uint8)
    dS_ptr,            # [total_rows, seq_kv]  bfloat16  (output)
    seq_kv,            # runtime int
    inv_keep_prob,     # runtime float32
    BLOCK_KV: tl.constexpr,
    SINGLE_PASS: tl.constexpr,  # True when seq_kv <= BLOCK_KV
):
    row_idx = tl.program_id(0)
    base = row_idx * seq_kv

    if SINGLE_PASS:
        # ── Single pass: load once, compute dot, write dS ────────────────────
        offs = tl.arange(0, BLOCK_KV)
        valid = offs < seq_kv

        dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
        dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
        P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

        dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
        dot = tl.sum(P_vals * dP_vals, axis=0)
        dS_vals = P_vals * (dP_vals - dot)
        tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)
    else:
        # ── Pass 1: compute dot = sum(P * dP) ────────────────────────────────
        dot = tl.zeros([1], dtype=tl.float32)
        for blk_start in tl.range(0, seq_kv, BLOCK_KV):
            offs = blk_start + tl.arange(0, BLOCK_KV)
            valid = offs < seq_kv

            dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
            dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
            P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

            dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
            dot += tl.sum(P_vals * dP_vals, axis=0)

        # ── Pass 2: compute dS = P * (dP - dot) and store ────────────────────
        for blk_start in tl.range(0, seq_kv, BLOCK_KV):
            offs = blk_start + tl.arange(0, BLOCK_KV)
            valid = offs < seq_kv

            dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
            dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
            P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

            dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
            dS_vals = P_vals * (dP_vals - dot)
            tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # ── Step 1: Transpose grad and reshape for grouped computation ───────────
    # [bs, sq, 80, d] -> [bs, 80, sq, d] (contiguous bfloat16)
    # Must be contiguous for cuBLAS BMM — this copy is necessary
    dO = grad_attn_output.transpose(1, 2).contiguous()
    # [bs, 80, sq, d] -> [bs, 8, 10, sq, d]
    dO_grouped = dO.reshape(bs, n_kv_heads, n_groups, seq_q, HEAD_DIM)

    # ── Step 2: BMM1 — compute dP_dropped (bfloat16 matmul) ─────────────────
    # vs_T: [bs, 8, 1, d, skv]  (bfloat16) — no .contiguous() needed, cuBLAS handles strides
    vs_T = value_states.transpose(-2, -1).unsqueeze(2)
    # dP_dropped_grouped: [bs, 8, 10, sq, skv]  bfloat16
    dP_dropped_grouped = torch.matmul(dO_grouped, vs_T)
    # Reshape to [bs, 80, sq, skv] — no copy, just a view
    dP_dropped = dP_dropped_grouped.reshape(bs, n_heads, seq_q, seq_kv)

    # ── Step 3: Fused Triton kernel — dropout bwd + softmax bwd ─────────────
    total_rows = bs * n_heads * seq_q
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    # Flatten to [total_rows, seq_kv] — avoid .contiguous() when already contiguous
    # dP_dropped is result of matmul (contiguous), attn_weights and dropout_mask may be contiguous already
    dP_dropped_flat   = dP_dropped.reshape(total_rows, seq_kv)
    attn_weights_flat = attn_weights.reshape(total_rows, seq_kv)
    dropout_mask_flat = dropout_mask.reshape(total_rows, seq_kv)
    dS_flat           = dS.reshape(total_rows, seq_kv)

    # Ensure contiguity only if needed (Triton needs contiguous row access)
    if not dP_dropped_flat.is_contiguous():
        dP_dropped_flat = dP_dropped_flat.contiguous()
    if not attn_weights_flat.is_contiguous():
        attn_weights_flat = attn_weights_flat.contiguous()
    if not dropout_mask_flat.is_contiguous():
        dropout_mask_flat = dropout_mask_flat.contiguous()

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # BLOCK_KV: next power of 2 >= seq_kv, cap at 8192 for register pressure
    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 8192)
    # Use single-pass when seq_kv fits in one tile
    SINGLE_PASS = (seq_kv <= BLOCK_KV)

    _softmax_bwd_kernel[(total_rows,)](
        dP_dropped_flat,
        attn_weights_flat,
        dropout_mask_flat,
        dS_flat,
        seq_kv,
        inv_keep_prob,
        BLOCK_KV=BLOCK_KV,
        SINGLE_PASS=SINGLE_PASS,
    )

    # ── Step 4: BMM2 — compute dV_grouped (bfloat16 matmul) ─────────────────
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    # No .contiguous() — cuBLAS handles non-contiguous strides via view
    P_dropped_grouped = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    # dV_grouped: [bs, 8, 10, skv, d]
    dV_grouped = torch.matmul(
        P_dropped_grouped.transpose(-2, -1),  # [bs, 8, 10, skv, sq]
        dO_grouped                             # [bs, 8, 10, sq, d]
    )

    # ── Step 5: GQA sum reduction via torch.sum ──────────────────────────────
    # dV_grouped: [bs, 8, 10, skv, d] — sum over dim=2 (groups)
    # Use float32 accumulation for numerical accuracy, then cast to bf16
    dV = dV_grouped.float().sum(dim=2).to(torch.bfloat16)

    return dS, dV
