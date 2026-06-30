"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- Keep both BMMs as bfloat16 torch.matmul (cuBLAS-optimized)
- Single-pass Triton kernel for elementwise dropout-bwd + softmax-bwd:
  * When seq_kv fits in BLOCK_KV tiles: load dP_dropped+mask+P ONCE,
    compute partial sums, warp-reduce to get dot, write dS — NO second pass
  * For very large seq_kv: fall back to two-pass (rare)
- BMM2 fused with GQA reduction: single large GEMM with K = 10*sq
  instead of 10 separate GEMMs + Triton sum kernel

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
# Triton kernel 1: single-pass fused dropout-bwd + softmax-bwd
#
# For each row (bs * n_heads * sq), length seq_kv:
#   dP = dP_dropped * mask / (1 - p)
#   dS = P * (dP - sum(P * dP))
#
# Single-pass strategy (SINGLE_PASS=True, seq_kv <= BLOCK_KV):
#   - Load dP_dropped, mask, P once into registers
#   - Compute dot = sum(P * dP) using tl.sum (already a reduction over the tile)
#   - Immediately write dS = P * (dP - dot)
#   - Zero extra memory traffic vs. two-pass (which reads all three arrays twice)
#
# Two-pass fallback (SINGLE_PASS=False, seq_kv > BLOCK_KV):
#   - Used only for unusually large seq_kv
#
# Grid: (total_rows,)  one program per (bs, head, sq) row
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
        # All data for this row fits in BLOCK_KV registers — zero re-read
        offs = tl.arange(0, BLOCK_KV)
        valid = offs < seq_kv

        dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
        dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
        P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

        # Apply dropout scaling
        dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)

        # Compute dot product (scalar reduction over the tile)
        dot = tl.sum(P_vals * dP_vals, axis=0)

        # Compute and store dS — data already in registers, no re-load needed
        dS_vals = P_vals * (dP_vals - dot)
        tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)
    else:
        # ── Two-pass fallback for very large seq_kv ──────────────────────────
        # Pass 1: compute dot = sum(P * dP)
        dot = tl.zeros([1], dtype=tl.float32)
        for blk_start in tl.range(0, seq_kv, BLOCK_KV):
            offs = blk_start + tl.arange(0, BLOCK_KV)
            valid = offs < seq_kv

            dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
            dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
            P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

            dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
            dot += tl.sum(P_vals * dP_vals, axis=0)

        # Pass 2: compute dS = P * (dP - dot) and store
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
    dO = grad_attn_output.transpose(1, 2).contiguous()
    # [bs, 80, sq, d] -> [bs, 8, 10, sq, d]
    dO_grouped = dO.reshape(bs, n_kv_heads, n_groups, seq_q, HEAD_DIM)

    # ── Step 2: BMM1 — compute dP_dropped (bfloat16 matmul) ─────────────────
    # vs_T: [bs, 8, 1, d, skv]  (bfloat16)
    vs_T = value_states.transpose(-2, -1).unsqueeze(2)
    # dP_dropped_grouped: [bs, 8, 10, sq, skv]  bfloat16
    dP_dropped_grouped = torch.matmul(dO_grouped, vs_T)
    # Reshape to [bs, 80, sq, skv] — zero-copy view
    dP_dropped = dP_dropped_grouped.reshape(bs, n_heads, seq_q, seq_kv)

    # ── Step 3: Single-pass Triton kernel — dropout bwd + softmax bwd ────────
    total_rows = bs * n_heads * seq_q
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    # Flatten to [total_rows, seq_kv]
    # dP_dropped was reshaped from a contiguous grouped tensor — check if already contiguous
    dP_dropped_flat   = dP_dropped.reshape(total_rows, seq_kv)
    if not dP_dropped_flat.is_contiguous():
        dP_dropped_flat = dP_dropped_flat.contiguous()
    attn_weights_flat = attn_weights.reshape(total_rows, seq_kv)
    if not attn_weights_flat.is_contiguous():
        attn_weights_flat = attn_weights_flat.contiguous()
    dropout_mask_flat = dropout_mask.reshape(total_rows, seq_kv)
    if not dropout_mask_flat.is_contiguous():
        dropout_mask_flat = dropout_mask_flat.contiguous()
    dS_flat           = dS.reshape(total_rows, seq_kv)

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # Choose BLOCK_KV: next power of 2 >= seq_kv, capped at 16384 for registers
    # Larger BLOCK_KV enables single-pass for bigger seq_kv values
    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 16384)
    # SINGLE_PASS: True when all seq_kv elements fit in one tile
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

    # ── Step 4: Fused BMM2 + GQA reduction — single large GEMM ──────────────
    #
    # Instead of:
    #   dV_grouped = bmm(P_dropped_grouped.T, dO_grouped)  [bs,8,10,skv,d]
    #   dV = dV_grouped.sum(dim=2)                          [bs,8,skv,d]
    #
    # We reshape to merge the groups into the K dimension:
    #   P_dropped: [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv] -> transpose -> [bs*8, skv, 10*sq]
    #   dO:        [bs, 8, 10, sq, d]   -> [bs*8, 10*sq, d]
    #   result:    [bs*8, skv, d]  (one GEMM with K = 10*sq, replaces 10 GEMMs + reduction)
    #
    # This is mathematically equivalent:
    #   dV[b,kv,s,d] = sum_{g,q} P_dropped[b,kv,g,q,s] * dO[b,kv,g,q,d]
    #                = P_dropped_2d^T @ dO_2d   where K = n_groups * seq_q

    # attn_weights_dropped: [bs, 80, sq, skv] — reshape to [bs*8, 10*sq, skv]
    # Need contiguous for the reshape+transpose
    P_dropped_2d = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    # Make contiguous so transpose+bmm works efficiently
    if not P_dropped_2d.is_contiguous():
        P_dropped_2d = P_dropped_2d.contiguous()
    # Transpose: [bs*8, skv, 10*sq]
    P_dropped_2d_T = P_dropped_2d.transpose(-2, -1)  # non-contiguous view, bmm handles it

    # dO: [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    dO_2d = dO_grouped.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)
    if not dO_2d.is_contiguous():
        dO_2d = dO_2d.contiguous()

    # Single BMM: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
    dV_flat = torch.bmm(P_dropped_2d_T, dO_2d)

    # Reshape to final output shape [bs, 8, skv, d]
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV
