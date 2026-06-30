"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- BMM1 + softmax-bwd FUSED in a single Triton kernel:
  * Reads grad_attn_output directly in [bs, sq, 80, d] layout (no transpose copy)
  * Reads value_states in [bs, 8, skv, d] layout (no expand)
  * For each (bs, head, sq) row: computes dP = dO_row @ V^T (d=128 dot products
    against skv key positions), applies dropout mask, computes softmax-bwd, writes dS
  * Eliminates: dO.contiguous() transpose, dP_dropped intermediate tensor, separate
    softmax-bwd kernel launch — fuses them all into one memory pass
- BMM2 kept as clean 3D batched cuBLAS GEMM (near-optimal)

Fused kernel grid: (bs * n_heads * seq_q,) — one CTA per row
Each CTA iterates over skv tiles (BLOCK_KV per tile), accumulating d=128 dot products.
Two-pass within the kernel: pass1 compute dP and sum(P*dP), pass2 write dS.
For single-pass (seq_kv <= BLOCK_KV): one pass only.

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
# Fused Triton kernel: BMM1 + dropout-bwd + softmax-bwd
#
# For each row (bs_idx, head_idx, sq_pos):
#   dO_row = grad_attn_output[bs_idx, sq_pos, head_idx, :]   (d=128)
#   kv_head = head_idx // n_groups
#   For each kv_pos in [0, seq_kv):
#     dP[kv_pos] = dot(dO_row, value_states[bs_idx, kv_head, kv_pos, :])
#   Apply dropout: dP_scaled[kv_pos] = mask[...] ? dP[kv_pos] * inv_keep_prob : 0
#   P_row = attn_weights[bs_idx, head_idx, sq_pos, :]
#   dot_val = sum(P_row * dP_scaled)
#   dS[bs_idx, head_idx, sq_pos, kv_pos] = P_row[kv_pos] * (dP_scaled[kv_pos] - dot_val)
#
# Access patterns:
#   grad_attn_output: [bs, sq, 80, d]  — stride_bs=sq*80*d, stride_sq=80*d, stride_h=d
#   value_states:     [bs, 8, skv, d]  — stride_bs=8*skv*d, stride_kv=skv*d, stride_pos=d
#   attn_weights:     [bs, 80, sq, skv]
#   dropout_mask:     [bs, 80, sq, skv]
#   dS output:        [bs, 80, sq, skv]
#
# Grid: (bs * n_heads * seq_q,)
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _fused_bmm1_softmax_bwd_kernel(
    # Inputs
    dO_ptr,            # [bs, sq, 80, d]    bfloat16   — original layout, strided
    value_ptr,         # [bs, 8,  skv, d]   bfloat16
    attn_w_ptr,        # [bs, 80, sq, skv]  bfloat16
    mask_ptr,          # [bs, 80, sq, skv]  bool
    # Output
    dS_ptr,            # [bs, 80, sq, skv]  bfloat16
    # Strides for dO [bs, sq, 80, d]
    dO_stride_bs,      # sq * 80 * d
    dO_stride_sq,      # 80 * d
    dO_stride_h,       # d
    # Strides for value [bs, 8, skv, d]
    vs_stride_bs,      # 8 * skv * d
    vs_stride_kv,      # skv * d
    vs_stride_pos,     # d
    # Dimensions
    seq_kv,
    inv_keep_prob,
    n_groups: tl.constexpr,   # 10
    HEAD_DIM: tl.constexpr,   # 128
    BLOCK_KV: tl.constexpr,   # tile size over seq_kv
    SINGLE_PASS: tl.constexpr,
    n_heads: tl.constexpr,    # 80
    n_kv_heads: tl.constexpr, # 8
):
    # Each program handles one row: (bs_idx, head_idx, sq_pos)
    row_idx = tl.program_id(0)

    # Decompose row_idx -> (bs_idx, head_idx, sq_pos)
    sq_pos   = row_idx % tl.num_programs(1)  # won't work — use arithmetic
    # Actually decompose manually:
    # row_idx = bs_idx * (n_heads * seq_q) + head_idx * seq_q + sq_pos
    # But seq_q is runtime — need to pass it
    # We'll pass a combined approach differently below
    # This kernel signature needs seq_q; let's restructure.
    # See the actual kernel below (_fused_bmm1_softmax_bwd_kernel2) which has seq_q.
    pass


# Clean implementation with all needed parameters
@triton.jit
def _fused_bmm1_softmax_bwd(
    # Inputs
    dO_ptr,            # [bs, sq, 80, d]    bfloat16   — original layout
    value_ptr,         # [bs, 8,  skv, d]   bfloat16
    attn_w_ptr,        # [bs, 80, sq, skv]  bfloat16
    mask_ptr,          # [bs, 80, sq, skv]  bool
    # Output
    dS_ptr,            # [bs, 80, sq, skv]  bfloat16
    # Strides for dO [bs, sq, 80, d]
    dO_stride_bs,
    dO_stride_sq,
    dO_stride_h,
    # Strides for value [bs, 8, skv, d]
    vs_stride_bs,
    vs_stride_kv,      # = skv * d
    vs_stride_pos,     # = d
    # Dimensions
    seq_q,
    seq_kv,
    inv_keep_prob,
    n_heads: tl.constexpr,     # 80
    n_groups: tl.constexpr,    # 10
    HEAD_DIM: tl.constexpr,    # 128
    BLOCK_KV: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    # Each program handles one (bs_idx, head_idx, sq_pos) row
    row_idx = tl.program_id(0)

    # Decompose: row_idx in [0, bs * n_heads * seq_q)
    sq_pos   = row_idx % seq_q
    tmp      = row_idx // seq_q
    head_idx = tmp % n_heads
    bs_idx   = tmp // n_heads

    kv_head  = head_idx // n_groups

    # ── Pointer to dO row: dO[bs_idx, sq_pos, head_idx, :] ──────────────────
    dO_row_ptr = dO_ptr + bs_idx * dO_stride_bs + sq_pos * dO_stride_sq + head_idx * dO_stride_h
    offs_d = tl.arange(0, HEAD_DIM)
    # Load dO row (d=128 elements) — stays in registers for all skv iterations
    dO_row = tl.load(dO_row_ptr + offs_d).to(tl.float32)  # [HEAD_DIM]

    # ── Base pointer for value: value[bs_idx, kv_head, :, :] ────────────────
    vs_base_ptr = value_ptr + bs_idx * vs_stride_bs + kv_head * vs_stride_kv

    # ── Base pointers for attn_w and mask (flat [bs,80,sq,skv] layout) ───────
    # Row offset in [bs, 80, sq, skv]: bs_idx * n_heads * seq_q * seq_kv + head_idx * seq_q * seq_kv + sq_pos * seq_kv
    attn_row_base = (bs_idx * n_heads * seq_q + head_idx * seq_q + sq_pos) * seq_kv
    dS_row_base   = attn_row_base

    if SINGLE_PASS:
        # ── Single pass: compute all dP, then dot, then write dS ─────────────
        # For seq_kv <= BLOCK_KV: process entire row in one tile
        offs_kv = tl.arange(0, BLOCK_KV)
        valid_kv = offs_kv < seq_kv

        # Load P and mask
        P_vals = tl.load(attn_w_ptr + attn_row_base + offs_kv, mask=valid_kv, other=0.0).to(tl.float32)
        mask_vals = tl.load(mask_ptr + attn_row_base + offs_kv, mask=valid_kv, other=0).to(tl.int1)

        # Compute dP = dot(dO_row, V[kv_pos, :]) for each kv_pos in tile
        # value layout: [skv, d] starting at vs_base_ptr
        # We need to load value[offs_kv, :] — shape [BLOCK_KV, HEAD_DIM]
        vs_ptrs = vs_base_ptr + offs_kv[:, None] * vs_stride_pos + offs_d[None, :]
        V_tile = tl.load(vs_ptrs, mask=valid_kv[:, None], other=0.0).to(tl.float32)  # [BLOCK_KV, HEAD_DIM]

        # dP[kv_pos] = sum_d(dO_row[d] * V_tile[kv_pos, d])
        dP_raw = tl.sum(dO_row[None, :] * V_tile, axis=1)  # [BLOCK_KV]

        # Apply dropout
        dP_vals = tl.where(mask_vals, dP_raw * inv_keep_prob, 0.0)

        # Softmax backward: dot = sum(P * dP)
        dot = tl.sum(P_vals * dP_vals, axis=0)

        # dS = P * (dP - dot)
        dS_vals = P_vals * (dP_vals - dot)
        tl.store(dS_ptr + dS_row_base + offs_kv, dS_vals.to(tl.bfloat16), mask=valid_kv)

    else:
        # ── Two-pass: first accumulate dot, then write dS ────────────────────
        dot = tl.zeros([1], dtype=tl.float32)

        # Pass 1: compute dot = sum(P * dP) across all kv tiles
        for blk_start in tl.range(0, seq_kv, BLOCK_KV):
            offs_kv = blk_start + tl.arange(0, BLOCK_KV)
            valid_kv = offs_kv < seq_kv

            P_vals = tl.load(attn_w_ptr + attn_row_base + offs_kv, mask=valid_kv, other=0.0).to(tl.float32)
            mask_vals = tl.load(mask_ptr + attn_row_base + offs_kv, mask=valid_kv, other=0).to(tl.int1)

            vs_ptrs = vs_base_ptr + offs_kv[:, None] * vs_stride_pos + offs_d[None, :]
            V_tile = tl.load(vs_ptrs, mask=valid_kv[:, None], other=0.0).to(tl.float32)

            dP_raw = tl.sum(dO_row[None, :] * V_tile, axis=1)
            dP_vals = tl.where(mask_vals, dP_raw * inv_keep_prob, 0.0)
            dot += tl.sum(P_vals * dP_vals, axis=0)

        # Pass 2: write dS
        for blk_start in tl.range(0, seq_kv, BLOCK_KV):
            offs_kv = blk_start + tl.arange(0, BLOCK_KV)
            valid_kv = offs_kv < seq_kv

            P_vals = tl.load(attn_w_ptr + attn_row_base + offs_kv, mask=valid_kv, other=0.0).to(tl.float32)
            mask_vals = tl.load(mask_ptr + attn_row_base + offs_kv, mask=valid_kv, other=0).to(tl.int1)

            vs_ptrs = vs_base_ptr + offs_kv[:, None] * vs_stride_pos + offs_d[None, :]
            V_tile = tl.load(vs_ptrs, mask=valid_kv[:, None], other=0.0).to(tl.float32)

            dP_raw = tl.sum(dO_row[None, :] * V_tile, axis=1)
            dP_vals = tl.where(mask_vals, dP_raw * inv_keep_prob, 0.0)

            dS_vals = P_vals * (dP_vals - dot)
            tl.store(dS_ptr + dS_row_base + offs_kv, dS_vals.to(tl.bfloat16), mask=valid_kv)


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

    # ── Step 1: Compute strides for dO [bs, sq, 80, d] ───────────────────────
    # grad_attn_output is [bs, sq, 80, d] — use it directly without transposing
    dO_stride_bs = seq_q * n_heads * HEAD_DIM
    dO_stride_sq = n_heads * HEAD_DIM
    dO_stride_h  = HEAD_DIM

    # value_states [bs, 8, skv, d] strides
    vs_stride_bs  = n_kv_heads * seq_kv * HEAD_DIM
    vs_stride_kv  = seq_kv * HEAD_DIM
    vs_stride_pos = HEAD_DIM

    # Ensure inputs are contiguous for predictable strides
    if not grad_attn_output.is_contiguous():
        grad_attn_output = grad_attn_output.contiguous()
    if not value_states.is_contiguous():
        value_states = value_states.contiguous()
    if not attn_weights.is_contiguous():
        attn_weights = attn_weights.contiguous()
    if not dropout_mask.is_contiguous():
        dropout_mask = dropout_mask.contiguous()

    # ── Step 2: Launch fused Triton kernel — BMM1 + dropout-bwd + softmax-bwd
    total_rows = bs * n_heads * seq_q
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=device)

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # BLOCK_KV: tile size over seq_kv dimension
    # Cap at 512 to keep register pressure manageable (each tile loads [BLOCK_KV, HEAD_DIM] V)
    # For BLOCK_KV=512, HEAD_DIM=128: 512*128 = 65536 f32 elements = 256KB — too large
    # Use BLOCK_KV=64 to keep V tile small: 64*128=8192 f32 = 32KB in registers
    BLOCK_KV = 64
    SINGLE_PASS = (seq_kv <= BLOCK_KV)

    # num_warps: with BLOCK_KV=64, HEAD_DIM=128, use 4 warps
    num_warps = 4

    _fused_bmm1_softmax_bwd[(total_rows,)](
        grad_attn_output,
        value_states,
        attn_weights,
        dropout_mask,
        dS,
        dO_stride_bs, dO_stride_sq, dO_stride_h,
        vs_stride_bs, vs_stride_kv, vs_stride_pos,
        seq_q, seq_kv,
        inv_keep_prob,
        n_heads=n_heads,
        n_groups=n_groups,
        HEAD_DIM=HEAD_DIM,
        BLOCK_KV=BLOCK_KV,
        SINGLE_PASS=SINGLE_PASS,
        num_warps=num_warps,
    )

    # ── Step 3: BMM2 — clean 3D batched GEMM (kept as cuBLAS) ────────────────
    # Need dO in [bs*8, 10*sq, d] for BMM2 — must still transpose
    dO = grad_attn_output.transpose(1, 2).contiguous()
    dO_2d = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)
    if not dO_2d.is_contiguous():
        dO_2d = dO_2d.contiguous()

    P_dropped_2d = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    if not P_dropped_2d.is_contiguous():
        P_dropped_2d = P_dropped_2d.contiguous()
    P_dropped_2d_T = P_dropped_2d.transpose(-2, -1)

    dV_flat = torch.bmm(P_dropped_2d_T, dO_2d)
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV
