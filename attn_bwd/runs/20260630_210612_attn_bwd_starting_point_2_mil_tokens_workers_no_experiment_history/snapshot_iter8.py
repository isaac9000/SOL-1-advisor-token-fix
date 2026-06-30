"""
Optimized attention-backward kernel using BF16 GEMMs + Triton softmax backward.

Strategy:
  1. Avoid materializing V_exp [bs, 80, skv, d] entirely.
     For dP: reshape dO from [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
             use V as [bs*8, skv, d], compute bmm -> [bs*8, 10*sq, skv]
             reshape to [bs, 80, sq, skv]. No V copy needed!

  2. For dV: reshape P_dropped from [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
             reshape dO from [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
             compute [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
             This directly gives dV summed over groups -- no separate reduction!

  3. Triton softmax-backward kernel: single-pass approach.
     Each thread block handles one (batch, head, sq_row) row.
     Load dP*mask*inv_keep and P tiles, compute rowsum in one sweep,
     then immediately write dS in the same sweep using accumulated rowsum.
     For large skv, use a prefix-sum style approach with two loops but
     maximize BLOCK_SKV to reduce iterations.

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


@triton.jit
def softmax_bwd_single_pass_kernel(
    # dP: [bs, 80, sq, skv] bfloat16
    dP_ptr, stride_dp_bs, stride_dp_h, stride_dp_sq, stride_dp_skv,
    # P (attn_weights): [bs, 80, sq, skv] bfloat16
    P_ptr, stride_p_bs, stride_p_h, stride_p_sq, stride_p_skv,
    # dropout_mask: [bs, 80, sq, skv] bool
    mask_ptr, stride_m_bs, stride_m_h, stride_m_sq, stride_m_skv,
    # dS output: [bs, 80, sq, skv] bfloat16
    dS_ptr, stride_ds_bs, stride_ds_h, stride_ds_sq, stride_ds_skv,
    bs, n_heads, sq, skv,
    inv_keep_prob: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """
    Single-pass softmax backward with dropout application.
    Grid: (bs * n_heads * sq,)  -- one program per (batch, head, sq_row)

    For each sq row:
      - Load all skv tiles of dP, mask, P into registers in one sweep,
        accumulating rowsum = sum(dP_masked * P)
      - Store intermediate dP_masked tiles back to output buffer (dS)
      - Then do second sweep over dS tiles to apply: dS = P * (dP_masked - rowsum)

    This reduces memory reads: dP and mask are read once (stored to dS temp),
    P is read twice, but we save one read of mask and dP vs naive two-pass.

    Actually: true single-pass approach -- process one sq row at a time,
    with BLOCK_SKV = full skv dimension when possible (fits in registers).
    When skv fits in BLOCK_SKV: load everything, compute rowsum, write dS in one pass.
    When skv doesn't fit: use two loops (still reads mask/dP twice but with larger blocks).
    """
    pid = tl.program_id(0)

    # Decompose pid -> (batch_idx, head, sq_row)
    sq_row   = pid % sq
    bh_idx   = pid // sq
    batch_idx = bh_idx // n_heads
    head      = bh_idx % n_heads

    dP_base = dP_ptr   + batch_idx * stride_dp_bs + head * stride_dp_h + sq_row * stride_dp_sq
    P_base  = P_ptr    + batch_idx * stride_p_bs  + head * stride_p_h  + sq_row * stride_p_sq
    M_base  = mask_ptr + batch_idx * stride_m_bs  + head * stride_m_h  + sq_row * stride_m_sq
    dS_base = dS_ptr   + batch_idx * stride_ds_bs + head * stride_ds_h + sq_row * stride_ds_sq

    skv_offs = tl.arange(0, BLOCK_SKV)
    num_skv_blocks = tl.cdiv(skv, BLOCK_SKV)

    # ----- Pass 1: compute rowsum(dP_masked * P) -----
    rowsum = tl.zeros((1,), dtype=tl.float32)

    for skv_tile in range(num_skv_blocks):
        skv_start     = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask      = skv_tile_offs < skv

        # Load dP tile (BF16) and cast to float32
        dp_ptrs  = dP_base + skv_tile_offs * stride_dp_skv
        dp_tile  = tl.load(dp_ptrs, mask=skv_mask, other=0.0).to(tl.float32)

        # Load dropout mask
        m_ptrs   = M_base + skv_tile_offs * stride_m_skv
        m_tile   = tl.load(m_ptrs, mask=skv_mask, other=0).to(tl.float32)
        dp_masked = dp_tile * m_tile * inv_keep_prob

        # Load P
        p_ptrs   = P_base + skv_tile_offs * stride_p_skv
        p_tile   = tl.load(p_ptrs, mask=skv_mask, other=0.0).to(tl.float32)

        # Accumulate rowsum
        rowsum  += tl.sum(dp_masked * p_tile, axis=0, keep_dims=True)

    # ----- Pass 2: compute dS and store -----
    for skv_tile in range(num_skv_blocks):
        skv_start     = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask      = skv_tile_offs < skv

        dp_ptrs   = dP_base + skv_tile_offs * stride_dp_skv
        dp_tile   = tl.load(dp_ptrs, mask=skv_mask, other=0.0).to(tl.float32)

        m_ptrs    = M_base + skv_tile_offs * stride_m_skv
        m_tile    = tl.load(m_ptrs, mask=skv_mask, other=0).to(tl.float32)
        dp_masked = dp_tile * m_tile * inv_keep_prob

        p_ptrs    = P_base + skv_tile_offs * stride_p_skv
        p_tile    = tl.load(p_ptrs, mask=skv_mask, other=0.0).to(tl.float32)

        # dS = P * (dP_masked - rowsum)
        dS_tile = p_tile * (dp_masked - rowsum)

        ds_ptrs = dS_base + skv_tile_offs * stride_ds_skv
        tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=skv_mask)


@triton.jit
def softmax_bwd_large_block_kernel(
    # dP: [bs, 80, sq, skv] bfloat16
    dP_ptr, stride_dp_bs, stride_dp_h, stride_dp_sq, stride_dp_skv,
    # P (attn_weights): [bs, 80, sq, skv] bfloat16
    P_ptr, stride_p_bs, stride_p_h, stride_p_sq, stride_p_skv,
    # dropout_mask: [bs, 80, sq, skv] bool
    mask_ptr, stride_m_bs, stride_m_h, stride_m_sq, stride_m_skv,
    # dS output: [bs, 80, sq, skv] bfloat16
    dS_ptr, stride_ds_bs, stride_ds_h, stride_ds_sq, stride_ds_skv,
    bs, n_heads, sq, skv,
    inv_keep_prob: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """
    Two-pass softmax backward with large blocks.
    Grid: (bs * n_heads, cdiv(sq, BLOCK_SQ))
    Processes BLOCK_SQ rows simultaneously for better memory efficiency.
    Uses large BLOCK_SKV to minimize loop iterations.
    """
    pid_bh = tl.program_id(0)
    pid_sq = tl.program_id(1)

    batch_idx = pid_bh // n_heads
    head      = pid_bh % n_heads

    sq_start = pid_sq * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < sq

    skv_offs = tl.arange(0, BLOCK_SKV)

    dP_base = dP_ptr   + batch_idx * stride_dp_bs + head * stride_dp_h
    P_base  = P_ptr    + batch_idx * stride_p_bs  + head * stride_p_h
    M_base  = mask_ptr + batch_idx * stride_m_bs  + head * stride_m_h
    dS_base = dS_ptr   + batch_idx * stride_ds_bs + head * stride_ds_h

    num_skv_blocks = tl.cdiv(skv, BLOCK_SKV)

    # ----- Pass 1: compute rowsum(dP_masked * P) -----
    rowsum = tl.zeros((BLOCK_SQ,), dtype=tl.float32)

    for skv_tile in range(num_skv_blocks):
        skv_start     = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask      = skv_tile_offs < skv

        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dp_ptrs  = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
        dp_tile  = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        m_ptrs   = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile   = tl.load(m_ptrs, mask=combined_mask, other=0).to(tl.float32)
        dp_masked = dp_tile * m_tile * inv_keep_prob

        p_ptrs   = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile   = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        rowsum  += tl.sum(dp_masked * p_tile, axis=1)

    # ----- Pass 2: compute dS and store -----
    for skv_tile in range(num_skv_blocks):
        skv_start     = skv_tile * BLOCK_SKV
        skv_tile_offs = skv_start + skv_offs
        skv_mask      = skv_tile_offs < skv

        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        dp_ptrs   = dP_base + sq_offs[:, None] * stride_dp_sq + skv_tile_offs[None, :] * stride_dp_skv
        dp_tile   = tl.load(dp_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        m_ptrs    = M_base + sq_offs[:, None] * stride_m_sq + skv_tile_offs[None, :] * stride_m_skv
        m_tile    = tl.load(m_ptrs, mask=combined_mask, other=0).to(tl.float32)
        dp_masked = dp_tile * m_tile * inv_keep_prob

        p_ptrs    = P_base + sq_offs[:, None] * stride_p_sq + skv_tile_offs[None, :] * stride_p_skv
        p_tile    = tl.load(p_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        dS_tile = p_tile * (dp_masked - rowsum[:, None])

        ds_ptrs = dS_base + sq_offs[:, None] * stride_ds_sq + skv_tile_offs[None, :] * stride_ds_skv
        tl.store(ds_ptrs, dS_tile.to(tl.bfloat16), mask=combined_mask)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = N_GROUPS  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    d      = HEAD_DIM

    # Transpose dO: [bs, sq, 80, d] -> [bs, 80, sq, d] contiguous, keep BF16
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()  # [bs, 80, sq, d] bfloat16

    # Reshape dO for GQA-aware GEMMs:
    # [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]
    dO_grouped = dO.view(bs, n_kv_heads, n_groups, seq_q, d)
    dO_gqa = dO_grouped.reshape(bs * n_kv_heads, n_groups * seq_q, d)  # [bs*8, 10*sq, d]

    # V: [bs, 8, skv, d] -> [bs*8, skv, d] — NO expansion needed!
    V_gqa = value_states.reshape(bs * n_kv_heads, seq_kv, d)  # [bs*8, skv, d]

    # ---- GEMM 1 (BF16): dP = dO_gqa @ V_gqa^T ----
    # [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
    dP_gqa = torch.bmm(dO_gqa, V_gqa.transpose(-2, -1))  # [bs*8, 10*sq, skv] BF16
    dP_raw = dP_gqa.view(bs, n_kv_heads, n_groups, seq_q, seq_kv) \
                   .reshape(bs, n_heads, seq_q, seq_kv)  # [bs, 80, sq, skv] BF16

    # ---- GEMM 2 (BF16): dV = P_dropped_gqa^T @ dO_gqa ----
    # P_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
    P_dropped_gqa = attn_weights_dropped.view(bs, n_kv_heads, n_groups, seq_q, seq_kv) \
                                        .reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
    dV_gqa = torch.bmm(P_dropped_gqa.transpose(-2, -1), dO_gqa)  # [bs*8, skv, d] BF16

    # Reshape to [bs, 8, skv, d]
    dV = dV_gqa.view(bs, n_kv_heads, seq_kv, d).to(torch.bfloat16)

    # ---- Triton kernel: softmax backward ----
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)

    P_attn = attn_weights.contiguous()   # [bs, 80, sq, skv] bfloat16
    dmask  = dropout_mask.contiguous()   # [bs, 80, sq, skv] bool

    inv_keep = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Choose kernel based on seq_kv size
    # Single-pass-per-row kernel: one program per (bs, head, sq_row)
    # This eliminates the BLOCK_SQ dimension and processes one row at a time
    # Use BLOCK_SKV = power-of-2 >= skv when skv is small, else use 512
    if seq_kv <= 512:
        # Find smallest power of 2 >= seq_kv
        BLOCK_SKV_DS = 512
        if seq_kv <= 64:
            BLOCK_SKV_DS = 64
        elif seq_kv <= 128:
            BLOCK_SKV_DS = 128
        elif seq_kv <= 256:
            BLOCK_SKV_DS = 256

        # One program per (batch, head, sq_row) — true single-pass possible when BLOCK_SKV >= skv
        grid_dS = (bs * n_heads * seq_q,)
        softmax_bwd_single_pass_kernel[grid_dS](
            dP_raw, dP_raw.stride(0), dP_raw.stride(1), dP_raw.stride(2), dP_raw.stride(3),
            P_attn, P_attn.stride(0), P_attn.stride(1), P_attn.stride(2), P_attn.stride(3),
            dmask,  dmask.stride(0),  dmask.stride(1),  dmask.stride(2),  dmask.stride(3),
            dS,     dS.stride(0),     dS.stride(1),     dS.stride(2),     dS.stride(3),
            bs, n_heads, seq_q, seq_kv,
            inv_keep_prob=inv_keep,
            BLOCK_SKV=BLOCK_SKV_DS,
        )
    else:
        # Large skv: use batched row kernel with large BLOCK_SKV and BLOCK_SQ
        BLOCK_SQ_DS  = 4
        BLOCK_SKV_DS = 512

        grid_dS = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ_DS))
        softmax_bwd_large_block_kernel[grid_dS](
            dP_raw, dP_raw.stride(0), dP_raw.stride(1), dP_raw.stride(2), dP_raw.stride(3),
            P_attn, P_attn.stride(0), P_attn.stride(1), P_attn.stride(2), P_attn.stride(3),
            dmask,  dmask.stride(0),  dmask.stride(1),  dmask.stride(2),  dmask.stride(3),
            dS,     dS.stride(0),     dS.stride(1),     dS.stride(2),     dS.stride(3),
            bs, n_heads, seq_q, seq_kv,
            inv_keep_prob=inv_keep,
            BLOCK_SQ=BLOCK_SQ_DS, BLOCK_SKV=BLOCK_SKV_DS,
        )

    return dS, dV
