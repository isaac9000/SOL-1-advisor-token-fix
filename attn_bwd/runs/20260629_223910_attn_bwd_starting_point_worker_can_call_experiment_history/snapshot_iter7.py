"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- BMM1 (dO @ V^T) fused with dropout-bwd + softmax-bwd in a single Triton kernel
  → eliminates the [bs,80,sq,skv] dP_dropped intermediate tensor
- BMM2 (attn_weights_dropped^T @ dO) remains as cuBLAS bfloat16 matmul
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
N_GROUPS = 10


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused BMM1 + dropout-bwd + softmax-bwd
#
# Each program handles one row (b, head, sq_pos):
#   1. Load dO_row [HEAD_DIM] once into registers
#   2. Pass 1 over seq_kv: compute dP[j] = dO_row . V[kv_head, j, :],
#      apply dropout mask, accumulate dot = sum(P[j] * dP[j])
#   3. Pass 2 over seq_kv: compute dS[j] = P[j] * (dP[j] - dot), store
#
# GQA: head_idx maps to kv_head = head_idx // 10
#
# Grid: (bs * n_heads * seq_q,)
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _fused_bmm1_softmax_bwd_kernel(
    # Inputs
    dO_ptr,            # [bs, n_heads, seq_q, HEAD_DIM]  bfloat16 (contiguous after transpose)
    V_ptr,             # [bs, n_kv_heads, seq_kv, HEAD_DIM]  bfloat16
    attn_weights_ptr,  # [bs, n_heads, seq_q, seq_kv]  bfloat16
    dropout_mask_ptr,  # [bs, n_heads, seq_q, seq_kv]  bool (uint8)
    # Output
    dS_ptr,            # [bs, n_heads, seq_q, seq_kv]  bfloat16
    # Dimensions
    seq_q,             # runtime int
    seq_kv,            # runtime int
    inv_keep_prob,     # runtime float32
    n_heads: tl.constexpr,      # 80
    n_kv_heads: tl.constexpr,   # 8
    n_groups: tl.constexpr,     # 10
    HEAD_DIM: tl.constexpr,     # 128
    BLOCK_KV: tl.constexpr,     # tile size over seq_kv
):
    # Decode program id
    prog_id = tl.program_id(0)
    bs_idx  = prog_id // (n_heads * seq_q)
    rem     = prog_id % (n_heads * seq_q)
    head_idx = rem // seq_q
    sq_idx   = rem % seq_q

    kv_head_idx = head_idx // n_groups  # GQA mapping

    # ── Load dO_row [HEAD_DIM] into registers ─────────────────────────────────
    # dO layout: [bs, n_heads, seq_q, HEAD_DIM]
    dO_row_base = ((bs_idx * n_heads + head_idx) * seq_q + sq_idx) * HEAD_DIM
    offs_d = tl.arange(0, HEAD_DIM)
    dO_row = tl.load(dO_ptr + dO_row_base + offs_d).to(tl.float32)  # [HEAD_DIM]

    # Base pointers for attn_weights and dropout_mask rows
    # Layout: [bs, n_heads, seq_q, seq_kv]
    row_base = ((bs_idx * n_heads + head_idx) * seq_q + sq_idx) * seq_kv

    # V layout: [bs, n_kv_heads, seq_kv, HEAD_DIM]
    V_kv_base = (bs_idx * n_kv_heads + kv_head_idx) * seq_kv * HEAD_DIM

    # ── Pass 1: compute dot = sum_j(P[j] * dP[j]) ────────────────────────────
    dot = tl.zeros([1], dtype=tl.float32)

    for blk_start in tl.range(0, seq_kv, BLOCK_KV):
        offs_kv = blk_start + tl.arange(0, BLOCK_KV)
        valid_kv = offs_kv < seq_kv

        # Load P[j] values
        P_vals = tl.load(
            attn_weights_ptr + row_base + offs_kv,
            mask=valid_kv, other=0.0
        ).to(tl.float32)  # [BLOCK_KV]

        # Load dropout mask
        dmask_vals = tl.load(
            dropout_mask_ptr + row_base + offs_kv,
            mask=valid_kv, other=0
        ).to(tl.int1)  # [BLOCK_KV]

        # Compute dP[j] = dO_row . V[kv_head, j, :] for each j in block
        # V[kv_head, j, :] is at V_kv_base + j * HEAD_DIM + offs_d
        # We need [BLOCK_KV] dot products: each is sum over HEAD_DIM
        # Load V block: [BLOCK_KV, HEAD_DIM]
        V_block_base = V_kv_base + offs_kv[:, None] * HEAD_DIM + offs_d[None, :]
        V_block = tl.load(
            V_ptr + V_block_base,
            mask=valid_kv[:, None],
            other=0.0
        ).to(tl.float32)  # [BLOCK_KV, HEAD_DIM]

        # dP_raw[j] = sum_d(dO_row[d] * V_block[j, d])
        dP_raw = tl.sum(V_block * dO_row[None, :], axis=1)  # [BLOCK_KV]

        # Apply dropout mask
        dP_vals = tl.where(dmask_vals, dP_raw * inv_keep_prob, 0.0)

        # Accumulate dot product
        dot += tl.sum(P_vals * dP_vals, axis=0)

    # ── Pass 2: compute dS[j] = P[j] * (dP[j] - dot) and store ──────────────
    for blk_start in tl.range(0, seq_kv, BLOCK_KV):
        offs_kv = blk_start + tl.arange(0, BLOCK_KV)
        valid_kv = offs_kv < seq_kv

        # Load P[j] values
        P_vals = tl.load(
            attn_weights_ptr + row_base + offs_kv,
            mask=valid_kv, other=0.0
        ).to(tl.float32)

        # Load dropout mask
        dmask_vals = tl.load(
            dropout_mask_ptr + row_base + offs_kv,
            mask=valid_kv, other=0
        ).to(tl.int1)

        # Compute dP[j] again (recompute, cheaper than storing)
        V_block_base = V_kv_base + offs_kv[:, None] * HEAD_DIM + offs_d[None, :]
        V_block = tl.load(
            V_ptr + V_block_base,
            mask=valid_kv[:, None],
            other=0.0
        ).to(tl.float32)

        dP_raw = tl.sum(V_block * dO_row[None, :], axis=1)
        dP_vals = tl.where(dmask_vals, dP_raw * inv_keep_prob, 0.0)

        # dS = P * (dP - dot)
        dS_vals = P_vals * (dP_vals - dot)
        tl.store(
            dS_ptr + row_base + offs_kv,
            dS_vals.to(tl.bfloat16),
            mask=valid_kv
        )


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
    device = grad_attn_output.device

    # ── Step 1: Transpose grad to [bs, n_heads, seq_q, HEAD_DIM] ─────────────
    # [bs, sq, 80, d] -> [bs, 80, sq, d] (contiguous bfloat16)
    dO = grad_attn_output.transpose(1, 2).contiguous()

    # ── Step 2: Fused Triton kernel — BMM1 + dropout bwd + softmax bwd ───────
    # No intermediate dP_dropped tensor needed!
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=device)

    # Make inputs contiguous for Triton
    attn_weights_c  = attn_weights.contiguous()
    dropout_mask_c  = dropout_mask.contiguous()
    value_states_c  = value_states.contiguous()

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # BLOCK_KV: reasonable tile size (power of 2, cap at 128 for register pressure)
    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 128)

    total_programs = bs * n_heads * seq_q

    _fused_bmm1_softmax_bwd_kernel[(total_programs,)](
        dO,                 # [bs, n_heads, seq_q, HEAD_DIM]
        value_states_c,     # [bs, n_kv_heads, seq_kv, HEAD_DIM]
        attn_weights_c,     # [bs, n_heads, seq_q, seq_kv]
        dropout_mask_c,     # [bs, n_heads, seq_q, seq_kv]
        dS,                 # [bs, n_heads, seq_q, seq_kv]  output
        seq_q,
        seq_kv,
        inv_keep_prob,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_groups=n_groups,
        HEAD_DIM=HEAD_DIM,
        BLOCK_KV=BLOCK_KV,
    )

    # ── Step 3: BMM2 — compute dV_grouped (bfloat16 matmul) ──────────────────
    # Reshape dO for GQA grouping: [bs, 80, sq, d] -> [bs, 8, 10, sq, d]
    dO_grouped = dO.reshape(bs, n_kv_heads, n_groups, seq_q, HEAD_DIM)

    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    P_dropped_grouped = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    # dV_grouped: [bs, 8, 10, skv, d]
    dV_grouped = torch.matmul(
        P_dropped_grouped.transpose(-2, -1),  # [bs, 8, 10, skv, sq]
        dO_grouped                             # [bs, 8, 10, sq, d]
    )

    # ── Step 4: Triton GQA sum reduction ─────────────────────────────────────
    dV = torch.empty((bs, n_kv_heads, seq_kv, HEAD_DIM), dtype=torch.bfloat16, device=device)

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
