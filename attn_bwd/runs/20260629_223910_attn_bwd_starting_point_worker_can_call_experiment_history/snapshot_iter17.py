"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach
with CUDA graph capture for reduced launch overhead.

Strategy:
- Both BMMs as clean 3D batched GEMMs (cuBLAS-optimized, no broadcasting)
- BMM1 restructured: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
  then reshape to [bs, 80, sq, skv] — same K-merging trick as BMM2
- BMM2 fused with GQA reduction: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
- dO_2d [bs*8, 10*sq, d] reused across both BMMs (computed once)
- Row-batched Triton kernel for elementwise dropout-bwd + softmax-bwd
- CUDA graph capture: cache graphs keyed on (bs, seq_q, seq_kv) to amortize
  kernel launch overhead across repeated calls with same shapes

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
    SINGLE_PASS: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_CTA

    for i in tl.static_range(ROWS_PER_CTA):
        row_idx = row_start + i
        if row_idx < total_rows:
            base = row_idx * seq_kv

            if SINGLE_PASS:
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


# ─────────────────────────────────────────────────────────────────────────────
# CUDA Graph cache: keyed on (bs, seq_q, seq_kv)
# Each entry stores: (graph, static_inputs_dict, static_outputs_dict)
# ─────────────────────────────────────────────────────────────────────────────
_cuda_graph_cache = {}


def _compute_core(
    dO_2d, vs_T_2d, P_dropped_2d_T,
    attn_weights_flat, dropout_mask_flat,
    dP_dropped_2d, dV_flat, dS_flat,
    bs, n_heads, n_kv_heads, n_groups, seq_q, seq_kv,
    inv_keep_prob, BLOCK_KV, SINGLE_PASS, ROWS_PER_CTA,
):
    """Core computation: BMM1 + softmax-bwd + BMM2."""
    # BMM1: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
    torch.bmm(dO_2d, vs_T_2d, out=dP_dropped_2d)

    # Softmax-bwd Triton kernel
    total_rows = bs * n_heads * seq_q
    grid_size = triton.cdiv(total_rows, ROWS_PER_CTA)
    _softmax_bwd_kernel[(grid_size,)](
        dP_dropped_2d.reshape(total_rows, seq_kv),
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

    # BMM2: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
    torch.bmm(P_dropped_2d_T, dO_2d, out=dV_flat)


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

    # ── Prepare shared inputs ─────────────────────────────────────────────────
    # [bs, sq, 80, d] -> [bs, 80, sq, d] -> [bs*8, 10*sq, d]
    dO = grad_attn_output.transpose(1, 2).contiguous()
    dO_2d = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)
    if not dO_2d.is_contiguous():
        dO_2d = dO_2d.contiguous()

    # vs_T_2d: [bs*8, d, skv]
    vs_T_2d = value_states.transpose(-2, -1).reshape(bs * n_kv_heads, HEAD_DIM, seq_kv)
    if not vs_T_2d.is_contiguous():
        vs_T_2d = vs_T_2d.contiguous()

    # P_dropped_2d_T: [bs*8, skv, 10*sq]
    P_dropped_2d = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    if not P_dropped_2d.is_contiguous():
        P_dropped_2d = P_dropped_2d.contiguous()
    P_dropped_2d_T = P_dropped_2d.transpose(-2, -1).contiguous()

    # Flatten attn_weights and dropout_mask for Triton kernel
    total_rows = bs * n_heads * seq_q
    attn_weights_flat = attn_weights.reshape(total_rows, seq_kv)
    if not attn_weights_flat.is_contiguous():
        attn_weights_flat = attn_weights_flat.contiguous()
    dropout_mask_flat = dropout_mask.reshape(total_rows, seq_kv)
    if not dropout_mask_flat.is_contiguous():
        dropout_mask_flat = dropout_mask_flat.contiguous()

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 16384)
    SINGLE_PASS = (seq_kv <= BLOCK_KV)
    ROWS_PER_CTA = 4

    # ── CUDA Graph capture / replay ───────────────────────────────────────────
    shape_key = (bs, seq_q, seq_kv)

    if shape_key not in _cuda_graph_cache:
        # Allocate static buffers for graph capture
        static_dO_2d           = dO_2d.clone()
        static_vs_T_2d         = vs_T_2d.clone()
        static_P_dropped_2d_T  = P_dropped_2d_T.clone()
        static_attn_w_flat     = attn_weights_flat.clone()
        static_mask_flat       = dropout_mask_flat.clone()
        static_dP_dropped_2d   = torch.empty((bs * n_kv_heads, n_groups * seq_q, seq_kv),
                                              dtype=torch.bfloat16, device=device)
        static_dV_flat         = torch.empty((bs * n_kv_heads, seq_kv, HEAD_DIM),
                                              dtype=torch.bfloat16, device=device)
        static_dS_flat         = torch.empty((total_rows, seq_kv),
                                              dtype=torch.bfloat16, device=device)

        # Warm-up pass (also compiles Triton kernel)
        torch.cuda.synchronize()
        for _ in range(3):
            _compute_core(
                static_dO_2d, static_vs_T_2d, static_P_dropped_2d_T,
                static_attn_w_flat, static_mask_flat,
                static_dP_dropped_2d, static_dV_flat, static_dS_flat,
                bs, n_heads, n_kv_heads, n_groups, seq_q, seq_kv,
                inv_keep_prob, BLOCK_KV, SINGLE_PASS, ROWS_PER_CTA,
            )
        torch.cuda.synchronize()

        # Capture the graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _compute_core(
                static_dO_2d, static_vs_T_2d, static_P_dropped_2d_T,
                static_attn_w_flat, static_mask_flat,
                static_dP_dropped_2d, static_dV_flat, static_dS_flat,
                bs, n_heads, n_kv_heads, n_groups, seq_q, seq_kv,
                inv_keep_prob, BLOCK_KV, SINGLE_PASS, ROWS_PER_CTA,
            )

        _cuda_graph_cache[shape_key] = {
            'graph': g,
            'static_dO_2d': static_dO_2d,
            'static_vs_T_2d': static_vs_T_2d,
            'static_P_dropped_2d_T': static_P_dropped_2d_T,
            'static_attn_w_flat': static_attn_w_flat,
            'static_mask_flat': static_mask_flat,
            'static_dP_dropped_2d': static_dP_dropped_2d,
            'static_dV_flat': static_dV_flat,
            'static_dS_flat': static_dS_flat,
        }

    entry = _cuda_graph_cache[shape_key]
    g = entry['graph']

    # Copy live inputs into static buffers
    entry['static_dO_2d'].copy_(dO_2d)
    entry['static_vs_T_2d'].copy_(vs_T_2d)
    entry['static_P_dropped_2d_T'].copy_(P_dropped_2d_T)
    entry['static_attn_w_flat'].copy_(attn_weights_flat)
    entry['static_mask_flat'].copy_(dropout_mask_flat)

    # Replay the graph
    g.replay()

    # Reshape outputs
    dS = entry['static_dS_flat'].reshape(bs, n_heads, seq_q, seq_kv).clone()
    dV = entry['static_dV_flat'].reshape(bs, n_kv_heads, seq_kv, HEAD_DIM).clone()

    return dS, dV
