"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- Both BMMs as clean 3D batched GEMMs (cuBLAS-optimized, no broadcasting)
- BMM1 restructured: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
  then reshape to [bs, 80, sq, skv] — same K-merging trick as BMM2
- BMM2 fused with GQA reduction: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
- dO_2d [bs*8, 10*sq, d] reused across both BMMs (computed once)
- Row-batched Triton kernel for elementwise dropout-bwd + softmax-bwd:
  * ROWS_PER_CTA rows processed per program to increase SM occupancy
  * Each row is handled independently within the CTA, amortizing launch overhead

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
# Triton kernel: row-batched fused dropout-bwd + softmax-bwd
#
# For each row (bs * n_heads * sq), length seq_kv:
#   dP = dP_dropped * mask / (1 - p)
#   dS = P * (dP - sum(P * dP))
#
# Row-batching strategy (ROWS_PER_CTA > 1):
#   - Each program handles ROWS_PER_CTA consecutive rows
#   - Grid size = ceil(total_rows / ROWS_PER_CTA)
#   - Increases SM occupancy by reducing kernel launch overhead and
#     giving each CTA more work (better warp utilization)
#
# For single-pass (seq_kv <= BLOCK_KV):
#   - Load dP_dropped, mask, P once per row into registers
#   - Compute dot via tl.sum, write dS — no re-read
#
# Two-pass fallback (seq_kv > BLOCK_KV):
#   - Pass 1: compute partial dot sums, Pass 2: write dS
#
# Grid: (ceil(total_rows / ROWS_PER_CTA),)
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _softmax_bwd_kernel(
    dP_dropped_ptr,    # [total_rows, seq_kv]  bfloat16
    attn_weights_ptr,  # [total_rows, seq_kv]  bfloat16
    dropout_mask_ptr,  # [total_rows, seq_kv]  bool (uint8)
    dS_ptr,            # [total_rows, seq_kv]  bfloat16  (output)
    total_rows,        # runtime int
    seq_kv,            # runtime int
    inv_keep_prob,     # runtime float32
    BLOCK_KV: tl.constexpr,
    SINGLE_PASS: tl.constexpr,  # True when seq_kv <= BLOCK_KV
    ROWS_PER_CTA: tl.constexpr,  # number of rows per CTA
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_CTA

    # Process ROWS_PER_CTA rows per CTA
    for i in tl.static_range(ROWS_PER_CTA):
        row_idx = row_start + i
        # Guard: skip rows beyond total_rows
        if row_idx < total_rows:
            base = row_idx * seq_kv

            if SINGLE_PASS:
                # ── Single pass: load once, compute dot, write dS ────────────
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
                # ── Two-pass fallback for very large seq_kv ──────────────────
                dot = tl.zeros([1], dtype=tl.float32)
                for blk_start in tl.range(0, seq_kv, BLOCK_KV):
                    offs = blk_start + tl.arange(0, BLOCK_KV)
                    valid = offs < seq_kv

                    dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
                    dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
                    P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

                    dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
                    dot += tl.sum(P_vals * dP_vals, axis=0)

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

    # ── Step 1: Transpose grad and build dO_2d for both BMMs ─────────────────
    # [bs, sq, 80, d] -> [bs, 80, sq, d] (contiguous bfloat16)
    dO = grad_attn_output.transpose(1, 2).contiguous()
    # [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    # This single reshape is reused for BOTH BMM1 and BMM2
    dO_2d = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)
    if not dO_2d.is_contiguous():
        dO_2d = dO_2d.contiguous()

    # ── Step 2: BMM1 — clean 3D batched GEMM (no broadcasting) ──────────────
    #
    # vs_T_2d: [bs, 8, skv, d] -> transpose -> [bs, 8, d, skv] -> [bs*8, d, skv]
    vs_T_2d = value_states.transpose(-2, -1).reshape(bs * n_kv_heads, HEAD_DIM, seq_kv)
    if not vs_T_2d.is_contiguous():
        vs_T_2d = vs_T_2d.contiguous()

    # Single clean 3D BMM: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
    dP_dropped_2d = torch.bmm(dO_2d, vs_T_2d)

    # Reshape to [bs, 80, sq, skv] for the Triton softmax-bwd kernel
    dP_dropped = dP_dropped_2d.reshape(bs, n_heads, seq_q, seq_kv)

    # ── Step 3: Row-batched Triton kernel — dropout bwd + softmax bwd ────────
    total_rows = bs * n_heads * seq_q
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    # Flatten to [total_rows, seq_kv]
    dP_dropped_flat = dP_dropped.reshape(total_rows, seq_kv)
    if not dP_dropped_flat.is_contiguous():
        dP_dropped_flat = dP_dropped_flat.contiguous()
    attn_weights_flat = attn_weights.reshape(total_rows, seq_kv)
    if not attn_weights_flat.is_contiguous():
        attn_weights_flat = attn_weights_flat.contiguous()
    dropout_mask_flat = dropout_mask.reshape(total_rows, seq_kv)
    if not dropout_mask_flat.is_contiguous():
        dropout_mask_flat = dropout_mask_flat.contiguous()
    dS_flat = dS.reshape(total_rows, seq_kv)

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # Choose BLOCK_KV: next power of 2 >= seq_kv, capped at 16384 for registers
    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 16384)
    SINGLE_PASS = (seq_kv <= BLOCK_KV)

    # ROWS_PER_CTA: batch multiple rows per CTA to increase SM occupancy
    # Using 4 rows per CTA — amortizes launch overhead, increases warp utilization
    ROWS_PER_CTA = 4
    grid_size = triton.cdiv(total_rows, ROWS_PER_CTA)

    _softmax_bwd_kernel[(grid_size,)](
        dP_dropped_flat,
        attn_weights_flat,
        dropout_mask_flat,
        dS_flat,
        total_rows,
        seq_kv,
        inv_keep_prob,
        BLOCK_KV=BLOCK_KV,
        SINGLE_PASS=SINGLE_PASS,
        ROWS_PER_CTA=ROWS_PER_CTA,
        num_warps=4,
    )

    # ── Step 4: Fused BMM2 + GQA reduction — single large GEMM ──────────────
    #
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs*8, 10*sq, skv]
    # dO_2d: [bs*8, 10*sq, d]  (already computed above)
    #
    # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]

    P_dropped_2d = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    if not P_dropped_2d.is_contiguous():
        P_dropped_2d = P_dropped_2d.contiguous()
    # Transpose: [bs*8, skv, 10*sq]
    P_dropped_2d_T = P_dropped_2d.transpose(-2, -1)  # non-contiguous view, bmm handles it

    # Single BMM: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
    dV_flat = torch.bmm(P_dropped_2d_T, dO_2d)

    # Reshape to final output shape [bs, 8, skv, d]
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV
