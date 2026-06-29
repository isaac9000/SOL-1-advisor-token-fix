"""
Optimized attention-backward kernel:
- Avoids the expensive dO.permute(0,2,1,3).contiguous() [bs,80,sq,128] flat copy.
- Instead, reshapes dO to [bs, sq, 8, 10, 128] and does ONE permute to
  [bs, 8, 10, sq, 128] contiguous — same size but structured for 5D matmuls.
- dP: attn_5d^T not needed; compute as dO_perm5 @ V^T broadcasting.
  dO_perm5 [bs, 8, 10, sq, 128] @ vs_T [bs, 8, 128, skv] -> [bs, 8, 10, sq, skv]
  -> reshape to [bs*8, 10*sq, skv]
- dV: attn_5d^T [bs, 8, 10, skv, sq] @ dO_perm5 [bs, 8, 10, sq, 128]
  -> [bs, 8, 10, skv, 128] -> sum dim=2 -> [bs, 8, skv, 128]
- ONE contiguous copy of dO (as [bs,8,10,sq,128]) serves both BMMs.
- Triton softmax-backward with row batching.
- Sequential execution.

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


@triton.jit
def fused_softmax_bwd_batched(
    dP_dropped_ptr,
    P_ptr,
    dropout_mask_ptr,
    dS_ptr,
    total_rows,
    scale: tl.constexpr,
    seq_kv: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    """
    Batched softmax-backward kernel: each program handles ROWS_PER_BLOCK rows.
    Grid: ceil(total_rows / ROWS_PER_BLOCK)
    """
    block_id = tl.program_id(0)
    row_start = block_id * ROWS_PER_BLOCK

    for r in tl.static_range(ROWS_PER_BLOCK):
        row_id = row_start + r
        if row_id < total_rows:
            base = row_id * seq_kv

            if BLOCK_SKV >= seq_kv:
                offsets = tl.arange(0, BLOCK_SKV)
                mask_bounds = offsets < seq_kv

                dp_dropped = tl.load(dP_dropped_ptr + base + offsets,
                                     mask=mask_bounds, other=0.0).to(tl.float32)
                dm = tl.load(dropout_mask_ptr + base + offsets,
                             mask=mask_bounds, other=0).to(tl.int1)
                p = tl.load(P_ptr + base + offsets,
                            mask=mask_bounds, other=0.0).to(tl.float32)

                dp = tl.where(dm, dp_dropped * scale, 0.0)
                row_sum = tl.sum(dp * p, axis=0)
                ds = p * (dp - row_sum)

                tl.store(dS_ptr + base + offsets, ds.to(tl.bfloat16), mask=mask_bounds)
            else:
                row_sum = tl.zeros([1], dtype=tl.float32)
                for block_start in range(0, seq_kv, BLOCK_SKV):
                    offsets = block_start + tl.arange(0, BLOCK_SKV)
                    mask_bounds = offsets < seq_kv
                    dp_dropped = tl.load(dP_dropped_ptr + base + offsets,
                                         mask=mask_bounds, other=0.0).to(tl.float32)
                    dm = tl.load(dropout_mask_ptr + base + offsets,
                                 mask=mask_bounds, other=0).to(tl.int1)
                    p = tl.load(P_ptr + base + offsets,
                                mask=mask_bounds, other=0.0).to(tl.float32)
                    dp = tl.where(dm, dp_dropped * scale, 0.0)
                    row_sum += tl.sum(dp * p, axis=0)

                for block_start in range(0, seq_kv, BLOCK_SKV):
                    offsets = block_start + tl.arange(0, BLOCK_SKV)
                    mask_bounds = offsets < seq_kv
                    dp_dropped = tl.load(dP_dropped_ptr + base + offsets,
                                         mask=mask_bounds, other=0.0).to(tl.float32)
                    dm = tl.load(dropout_mask_ptr + base + offsets,
                                 mask=mask_bounds, other=0).to(tl.int1)
                    p = tl.load(P_ptr + base + offsets,
                                mask=mask_bounds, other=0.0).to(tl.float32)
                    dp = tl.where(dm, dp_dropped * scale, 0.0)
                    ds = p * (dp - row_sum)
                    tl.store(dS_ptr + base + offsets, ds.to(tl.bfloat16), mask=mask_bounds)


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

    # =========================================================================
    # Step 1: ONE contiguous copy of dO, in [bs, 8, 10, sq, 128] layout.
    # This avoids the [bs, 80, sq, 128] permute of the baseline while exposing
    # GQA structure. Cost is identical in bytes but the layout directly feeds
    # 5D BMMs without additional reshaping.
    # grad_attn_output: [bs, sq, 80, 128] -> view as [bs, sq, 8, 10, 128]
    # -> permute(0,2,3,1,4) -> [bs, 8, 10, sq, 128] contiguous
    # =========================================================================
    dO_gqa_raw = grad_attn_output.reshape(bs, seq_q, n_kv_heads, n_groups, HEAD_DIM)
    dO_perm5 = dO_gqa_raw.permute(0, 2, 3, 1, 4).contiguous()
    # dO_perm5: [bs, 8, 10, sq, 128], bfloat16, contiguous

    # =========================================================================
    # Step 2: dP = dO @ V^T
    # dO_perm5: [bs, 8, 10, sq, 128]
    # vs_T:     [bs, 8, 128, skv]  -> broadcast as [bs, 8, 1, 128, skv]
    # Result:   [bs, 8, 10, sq, skv]
    # =========================================================================
    vs_T_bc = value_states.transpose(-2, -1).unsqueeze(2)  # [bs, 8, 1, 128, skv]
    # matmul broadcasts over dim 2 (groups): [bs,8,10,sq,128] @ [bs,8,1,128,skv]
    dP_5d = torch.matmul(dO_perm5, vs_T_bc)  # [bs, 8, 10, sq, skv]

    # Reshape for Triton softmax: [bs*8, 10*sq, skv]
    dP_groups = dP_5d.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # =========================================================================
    # Step 3: dV = attn_dropped^T @ dO
    # attn_5d:  [bs, 8, 10, sq, skv] (free view of attn_weights_dropped)
    # dO_perm5: [bs, 8, 10, sq, 128]
    # Result:   [bs, 8, 10, skv, 128] -> sum dim=2 -> [bs, 8, skv, 128]
    # =========================================================================
    attn_5d = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    # attn_5d^T: [bs, 8, 10, skv, sq]
    dV_5d = torch.matmul(attn_5d.transpose(-2, -1), dO_perm5)  # [bs, 8, 10, skv, 128]
    # Sum over groups: [bs, 8, skv, 128]
    dV_flat = dV_5d.sum(dim=2).to(torch.bfloat16)  # [bs, 8, skv, 128]

    # =========================================================================
    # Step 4: Fused softmax backward + dropout correction via Triton.
    # dP_groups: [bs*8, 10*sq, skv] needs to be aligned with attn_weights layout.
    # attn_weights: [bs, 80, sq, skv] = [bs, 8, 10, sq, skv]
    # dP_5d is already [bs, 8, 10, sq, skv] and dP_groups = reshape [bs*8, 10*sq, skv].
    # attn_weights reshape [bs*8, 10*sq, skv] — check alignment.
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    total_rows = bs * n_heads * seq_q

    dP_dropped_flat = dP_groups.reshape(total_rows, seq_kv)
    # attn_weights is [bs, 80, sq, skv] = [bs, 8, 10, sq, skv] — same logical order
    P_flat  = attn_weights.reshape(total_rows, seq_kv)
    dm_flat = dropout_mask.reshape(total_rows, seq_kv)

    dS_flat = torch.empty((total_rows, seq_kv), dtype=torch.bfloat16, device=device)

    # Choose BLOCK_SKV and ROWS_PER_BLOCK based on seq_kv
    if seq_kv <= 128:
        BLOCK_SKV = 128
        ROWS_PER_BLOCK = 16
    elif seq_kv <= 256:
        BLOCK_SKV = 256
        ROWS_PER_BLOCK = 8
    elif seq_kv <= 512:
        BLOCK_SKV = 512
        ROWS_PER_BLOCK = 4
    elif seq_kv <= 1024:
        BLOCK_SKV = 1024
        ROWS_PER_BLOCK = 2
    elif seq_kv <= 2048:
        BLOCK_SKV = 2048
        ROWS_PER_BLOCK = 1
    else:
        BLOCK_SKV = 2048
        ROWS_PER_BLOCK = 1

    num_blocks = (total_rows + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK

    fused_softmax_bwd_batched[(num_blocks,)](
        dP_dropped_flat, P_flat, dm_flat, dS_flat,
        total_rows=total_rows,
        scale=scale,
        seq_kv=seq_kv,
        BLOCK_SKV=BLOCK_SKV,
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
    )

    dS = dS_flat.reshape(bs, n_heads, seq_q, seq_kv)

    return dS, dV_flat
