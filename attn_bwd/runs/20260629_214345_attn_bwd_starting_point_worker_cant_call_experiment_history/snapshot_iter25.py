"""
Optimized attention-backward kernel — fused BMM1+softmax-bwd Triton kernel.

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
import math

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10


@triton.jit
def _fused_bmm1_softmax_bwd_kernel(
    # dO: [bs, n_heads, seq_q, d]  (already transposed, contiguous)
    dO_ptr,
    dO_stride_b, dO_stride_h, dO_stride_sq, dO_stride_d,
    # V: [bs, n_kv, seq_kv, d]
    V_ptr,
    V_stride_b, V_stride_kv, V_stride_skv, V_stride_d,
    # P (attn_weights): [bs, n_heads, seq_q, seq_kv]
    P_ptr,
    P_stride_b, P_stride_h, P_stride_sq, P_stride_skv,
    # dropout_mask: [bs, n_heads, seq_q, seq_kv] bool
    Mask_ptr,
    M_stride_b, M_stride_h, M_stride_sq, M_stride_skv,
    # dS output: [bs, n_heads, seq_q, seq_kv]
    dS_ptr,
    dS_stride_b, dS_stride_h, dS_stride_sq, dS_stride_skv,
    # dims
    seq_q: tl.constexpr,
    seq_kv,
    n_heads: tl.constexpr,
    n_kv: tl.constexpr,
    d: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Each program instance handles one (batch, head, query_row).
    - Loads dO row [d] once.
    - Pass 1: iterate over seq_kv tiles, compute dP = dO @ V^T tile,
              apply dropout mask and scale, accumulate row_sum = sum(dP_scaled * P).
    - Pass 2: iterate again, compute dS = P * (dP_scaled - row_sum), write out.
    """
    pid = tl.program_id(0)
    # Decode: pid = batch * n_heads * seq_q + head * seq_q + q_row
    n_heads_sq = n_heads * seq_q
    b   = pid // n_heads_sq
    rem = pid % n_heads_sq
    h   = rem // seq_q
    q   = rem %  seq_q

    # kv-head index (for GQA: n_heads=80, n_kv=8, group=10)
    kv_h = h // (n_heads // n_kv)

    # Pointers to dO row: dO[b, h, q, :]
    dO_row_base = (b * dO_stride_b + h * dO_stride_h +
                   q * dO_stride_sq)

    # Load dO row into registers — d=128, load in BLOCK_D chunks
    d_offs = tl.arange(0, BLOCK_D)
    dO_row = tl.load(dO_ptr + dO_row_base + d_offs * dO_stride_d,
                     mask=d_offs < d, other=0.0).to(tl.float32)

    # Base pointers for P, Mask, dS, V
    P_row_base  = (b * P_stride_b  + h * P_stride_h  + q * P_stride_sq)
    M_row_base  = (b * M_stride_b  + h * M_stride_h  + q * M_stride_sq)
    dS_row_base = (b * dS_stride_b + h * dS_stride_h + q * dS_stride_sq)
    V_base      = (b * V_stride_b  + kv_h * V_stride_kv)

    # ── Pass 1: compute row_sum = sum(dP_scaled * P) ─────────────────────────
    row_sum = tl.zeros([1], dtype=tl.float32)

    for kv_start in tl.range(0, seq_kv, BLOCK_KV):
        kv_offs = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offs < seq_kv

        # Load P tile
        p_tile = tl.load(P_ptr + P_row_base + kv_offs * P_stride_skv,
                         mask=kv_mask, other=0.0).to(tl.float32)

        # Load dropout mask tile
        m_tile = tl.load(Mask_ptr + M_row_base + kv_offs * M_stride_skv,
                         mask=kv_mask, other=0).to(tl.float32)

        # Load V tile: V[b, kv_h, kv_offs, :] -> shape [BLOCK_KV, BLOCK_D]
        v_tile = tl.load(
            V_ptr + V_base + kv_offs[:, None] * V_stride_skv + d_offs[None, :] * V_stride_d,
            mask=kv_mask[:, None] & (d_offs[None, :] < d), other=0.0
        ).to(tl.float32)

        # dP tile = dO_row @ V_tile^T  -> dot product for each kv position
        # dO_row: [BLOCK_D], v_tile: [BLOCK_KV, BLOCK_D]
        # Result: [BLOCK_KV]  -- dot product of each v row with dO
        dp_tile = tl.sum(dO_row[None, :] * v_tile, axis=1)  # [BLOCK_KV]

        # Apply dropout scale
        dp_scaled = dp_tile * m_tile * scale

        # Accumulate row sum
        row_sum += tl.sum(dp_scaled * p_tile, axis=0)

    row_sum_val = tl.sum(row_sum, axis=0)

    # ── Pass 2: compute dS and write out ─────────────────────────────────────
    for kv_start in tl.range(0, seq_kv, BLOCK_KV):
        kv_offs = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offs < seq_kv

        # Reload P tile
        p_tile = tl.load(P_ptr + P_row_base + kv_offs * P_stride_skv,
                         mask=kv_mask, other=0.0).to(tl.float32)

        # Reload dropout mask
        m_tile = tl.load(Mask_ptr + M_row_base + kv_offs * M_stride_skv,
                         mask=kv_mask, other=0).to(tl.float32)

        # Reload V tile
        v_tile = tl.load(
            V_ptr + V_base + kv_offs[:, None] * V_stride_skv + d_offs[None, :] * V_stride_d,
            mask=kv_mask[:, None] & (d_offs[None, :] < d), other=0.0
        ).to(tl.float32)

        # Recompute dp
        dp_tile   = tl.sum(dO_row[None, :] * v_tile, axis=1)
        dp_scaled = dp_tile * m_tile * scale

        # Softmax backward
        ds_tile = p_tile * (dp_scaled - row_sum_val)

        # Write dS
        tl.store(dS_ptr + dS_row_base + kv_offs * dS_stride_skv,
                 ds_tile.to(tl.bfloat16),
                 mask=kv_mask)


# Persistent side stream for BMM2 overlap
_side_stream = None

def _get_side_stream():
    global _side_stream
    if _side_stream is None:
        _side_stream = torch.cuda.Stream()
    return _side_stream


def custom_kernel(data):
    (dO_in, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs      = dO_in.shape[0]
    seq_q   = dO_in.shape[1]
    seq_kv  = value_states.shape[2]
    n_kv    = NUM_KEY_VALUE_HEADS   # 8
    n_g     = N_GROUPS              # 10
    d       = HEAD_DIM              # 128
    n_heads = NUM_ATTENTION_HEADS   # 80

    # ── Prepare dO: [bs, sq, 80, d] -> [bs, 80, sq, d] contiguous ────────────
    dO = dO_in.permute(0, 2, 1, 3).contiguous()  # [bs, 80, sq, d]

    # ── Value states: ensure contiguous [bs, 8, skv, d] ──────────────────────
    vs = value_states.contiguous()  # [bs, 8, skv, d]

    # ── Allocate dS output ────────────────────────────────────────────────────
    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=dO.device)

    # ── Launch fused kernel ───────────────────────────────────────────────────
    scale = 1.0 / (1.0 - attention_dropout)

    N_programs = bs * n_heads * seq_q
    BLOCK_KV   = 64
    BLOCK_D    = 128  # must be >= d (128), constexpr power-of-2

    grid = (N_programs,)

    _fused_bmm1_softmax_bwd_kernel[grid](
        dO,
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        vs,
        vs.stride(0), vs.stride(1), vs.stride(2), vs.stride(3),
        attn_weights,
        attn_weights.stride(0), attn_weights.stride(1),
        attn_weights.stride(2), attn_weights.stride(3),
        dropout_mask,
        dropout_mask.stride(0), dropout_mask.stride(1),
        dropout_mask.stride(2), dropout_mask.stride(3),
        dS,
        dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
        seq_q=seq_q, seq_kv=seq_kv,
        n_heads=n_heads, n_kv=n_kv, d=d,
        scale=scale,
        BLOCK_KV=BLOCK_KV, BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )

    # ── dV via torch.bmm on side stream ──────────────────────────────────────
    # dV = Pd^T @ dO   (GQA-grouped)
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
    # dO: [bs, 80, sq, d] -> [bs, 8, 10, sq, d] -> [bs*8, 10*sq, d]

    side_stream    = _get_side_stream()
    default_stream = torch.cuda.current_stream()
    side_stream.wait_stream(default_stream)

    with torch.cuda.stream(side_stream):
        Pd_grouped = attn_weights_dropped \
                        .reshape(bs, n_kv, n_g, seq_q, seq_kv) \
                        .reshape(bs * n_kv, n_g * seq_q, seq_kv) \
                        .contiguous()
        dO_grouped = dO.reshape(bs, n_kv, n_g, seq_q, d) \
                       .reshape(bs * n_kv, n_g * seq_q, d) \
                       .contiguous()
        # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
        dV_flat = torch.bmm(
            Pd_grouped.float().transpose(-2, -1),
            dO_grouped.float()
        )
        dV_fp32 = dV_flat.reshape(bs, n_kv, seq_kv, d)

    default_stream.wait_stream(side_stream)
    dV = dV_fp32.to(torch.bfloat16)

    return dS, dV
