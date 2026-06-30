"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- Keep both BMMs as bfloat16 torch.matmul (cuBLAS-optimized)
- Fused Triton kernel for elementwise dropout-bwd + softmax-bwd pass
- Triton kernel for GQA group-sum reduction (float32 accumulation, bf16 output)

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
# Grid: (total_rows,)
# seq_kv passed as runtime arg; BLOCK_KV is constexpr tile size
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
):
    row_idx = tl.program_id(0)
    base = row_idx * seq_kv

    # ── Pass 1: compute dot = sum(P * dP) ────────────────────────────────────
    dot = tl.zeros([1], dtype=tl.float32)
    for blk_start in tl.range(0, seq_kv, BLOCK_KV):
        offs = blk_start + tl.arange(0, BLOCK_KV)
        valid = offs < seq_kv

        dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
        dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
        P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

        dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
        dot += tl.sum(P_vals * dP_vals, axis=0)

    # ── Pass 2: compute dS = P * (dP - dot) and store ────────────────────────
    for blk_start in tl.range(0, seq_kv, BLOCK_KV):
        offs = blk_start + tl.arange(0, BLOCK_KV)
        valid = offs < seq_kv

        dP_d_vals = tl.load(dP_dropped_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
        dmask_vals = tl.load(dropout_mask_ptr + base + offs, mask=valid, other=0).to(tl.int1)
        P_vals = tl.load(attn_weights_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

        dP_vals = tl.where(dmask_vals, dP_d_vals * inv_keep_prob, 0.0)
        dS_vals = P_vals * (dP_vals - dot)
        tl.store(dS_ptr + base + offs, dS_vals.to(tl.bfloat16), mask=valid)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel 2: GQA group sum reduction
#
# dV_grouped: [bs*8, n_groups, skv, d]  bfloat16
# dV:         [bs*8,           skv, d]  bfloat16
#
# Grid: (bs*8*skv,)  — each program handles one (bs_kv, skv_pos) row
# Accumulates over n_groups in float32, stores bf16.
# HEAD_DIM=128 fits in one BLOCK_D=128 tile.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _gqa_sum_kernel(
    dV_grouped_ptr,      # [bs*8, n_groups, skv, d]  bfloat16
    dV_ptr,              # [bs*8, skv, d]             bfloat16
    n_groups: tl.constexpr,
    skv,                 # runtime int
    HEAD_DIM: tl.constexpr,
):
    row_idx = tl.program_id(0)   # in [0, bs*8*skv)

    bs_kv_idx = row_idx // skv   # index into (bs*8) space
    skv_pos   = row_idx % skv

    offs_d = tl.arange(0, HEAD_DIM)

    # Accumulate over n_groups in float32
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for g in tl.range(0, n_groups):
        # dV_grouped layout: [bs*8, n_groups, skv, d]
        # flat index: (bs_kv_idx * n_groups + g) * skv + skv_pos  — then * HEAD_DIM
        grouped_row = (bs_kv_idx * n_groups + g) * skv + skv_pos
        ptr = dV_grouped_ptr + grouped_row * HEAD_DIM + offs_d
        val = tl.load(ptr).to(tl.float32)
        acc += val

    # Output: [bs*8, skv, d] — row = bs_kv_idx * skv + skv_pos
    out_row = bs_kv_idx * skv + skv_pos
    tl.store(dV_ptr + out_row * HEAD_DIM + offs_d, acc.to(tl.bfloat16))


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
    # Reshape to [bs, 80, sq, skv]
    dP_dropped = dP_dropped_grouped.reshape(bs, n_heads, seq_q, seq_kv)

    # ── Step 3: Fused Triton kernel — dropout bwd + softmax bwd ─────────────
    total_rows = bs * n_heads * seq_q
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    # Flatten to [total_rows, seq_kv]
    dP_dropped_flat   = dP_dropped.reshape(total_rows, seq_kv).contiguous()
    attn_weights_flat = attn_weights.reshape(total_rows, seq_kv).contiguous()
    dropout_mask_flat = dropout_mask.reshape(total_rows, seq_kv).contiguous()
    dS_flat           = dS.reshape(total_rows, seq_kv)

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # BLOCK_KV: next power of 2 >= seq_kv, max 4096 (loop handles larger)
    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 4096)

    _softmax_bwd_kernel[(total_rows,)](
        dP_dropped_flat,
        attn_weights_flat,
        dropout_mask_flat,
        dS_flat,
        seq_kv,
        inv_keep_prob,
        BLOCK_KV=BLOCK_KV,
    )

    # ── Step 4: BMM2 — compute dV_grouped (bfloat16 matmul) ─────────────────
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    P_dropped_grouped = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    # dV_grouped: [bs, 8, 10, skv, d]
    dV_grouped = torch.matmul(
        P_dropped_grouped.transpose(-2, -1),  # [bs, 8, 10, skv, sq]
        dO_grouped                             # [bs, 8, 10, sq, d]
    )

    # ── Step 5: Triton GQA sum reduction ─────────────────────────────────────
    dV = torch.empty((bs, n_kv_heads, seq_kv, HEAD_DIM), dtype=torch.bfloat16, device=dO.device)

    # Flatten: dV_grouped [bs, 8, 10, skv, d] -> [bs*8, 10, skv, d]
    dV_grouped_flat = dV_grouped.reshape(bs * n_kv_heads, n_groups, seq_kv, HEAD_DIM).contiguous()
    dV_flat = dV.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)

    total_out_rows = bs * n_kv_heads * seq_kv

    _gqa_sum_kernel[(total_out_rows,)](
        dV_grouped_flat,
        dV_flat,
        n_groups=n_groups,
        skv=seq_kv,
        HEAD_DIM=HEAD_DIM,
    )

    return dS, dV
