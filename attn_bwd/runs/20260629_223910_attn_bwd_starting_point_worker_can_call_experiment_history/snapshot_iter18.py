"""
Optimized attention-backward kernel using hybrid Triton + torch.matmul approach.

Strategy:
- Both BMMs as clean 3D batched GEMMs (cuBLAS-optimized, no broadcasting)
- BMM1 restructured: [bs*8, 10*sq, d] @ [bs*8, d, skv] -> [bs*8, 10*sq, skv]
  then reshape to [bs, 80, sq, skv] — same K-merging trick as BMM2
- BMM2 fused with GQA reduction: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, d] -> [bs*8, skv, d]
- dO_2d [bs*8, 10*sq, d] reused across both BMMs (computed once)
- Dual-stream pipelining: BMM1 on stream A, BMM2 on stream B (launched concurrently)
  Triton softmax-bwd runs on stream A after BMM1 (overlaps with BMM2 on stream B)
  Final sync waits for both streams to complete
- Row-batched Triton kernel for elementwise dropout-bwd + softmax-bwd
- Removed unnecessary .contiguous() copies:
  * vs_T_2d: pass value_states as [bs*8, skv, d] and let cuBLAS handle the transpose
  * P_dropped_2d: already contiguous as a reshape of contiguous attn_weights_dropped
  * P_dropped_2d_T: passed directly as non-contiguous to torch.bmm (cuBLAS handles it)

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

    # ── Step 1: Transpose grad and build dO_2d for both BMMs ─────────────────
    # [bs, sq, 80, d] -> [bs, 80, sq, d] (contiguous bfloat16) -> [bs*8, 10*sq, d]
    dO = grad_attn_output.transpose(1, 2).contiguous()
    dO_2d = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)
    # dO_2d is contiguous since dO is contiguous and reshape preserves it

    # ── Step 2: Prepare BMM inputs without unnecessary copies ────────────────

    # BMM1: dO_2d [bs*8, 10*sq, d] @ vs_T [bs*8, d, skv] -> [bs*8, 10*sq, skv]
    # value_states is [bs, 8, skv, d] contiguous.
    # Reshape to [bs*8, skv, d] (zero-copy view, both dims are contiguous).
    # Then pass the transposed view [bs*8, d, skv] directly to bmm —
    # PyTorch/cuBLAS handles non-contiguous transposes natively without copying.
    vs_2d = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    # vs_2d is contiguous (reshape of contiguous tensor with compatible strides)
    vs_T_2d = vs_2d.transpose(-2, -1)  # [bs*8, d, skv] — non-contiguous, cuBLAS OK

    # BMM2: P_dropped_2d_T [bs*8, skv, 10*sq] @ dO_2d [bs*8, 10*sq, d] -> [bs*8, skv, d]
    # attn_weights_dropped is [bs, 80, sq, skv] contiguous.
    # Reshape to [bs*8, 10*sq, skv] is a zero-copy view (all dims fit contiguously).
    P_dropped_2d = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    # P_dropped_2d is contiguous (zero-copy reshape of contiguous input)
    P_dropped_2d_T = P_dropped_2d.transpose(-2, -1)  # [bs*8, skv, 10*sq] — non-contiguous, cuBLAS OK

    # Flatten attn_weights and dropout_mask for Triton kernel
    total_rows = bs * n_heads * seq_q
    attn_weights_flat = attn_weights.reshape(total_rows, seq_kv)
    # attn_weights is [bs, 80, sq, skv] contiguous -> reshape is zero-copy
    dropout_mask_flat = dropout_mask.reshape(total_rows, seq_kv)
    # dropout_mask is [bs, 80, sq, skv] contiguous -> reshape is zero-copy

    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    BLOCK_KV = min(triton.next_power_of_2(seq_kv), 16384)
    SINGLE_PASS = (seq_kv <= BLOCK_KV)
    ROWS_PER_CTA = 4
    grid_size = triton.cdiv(total_rows, ROWS_PER_CTA)

    # Allocate output tensors
    dP_dropped_2d = torch.empty((bs * n_kv_heads, n_groups * seq_q, seq_kv),
                                 dtype=torch.bfloat16, device=device)
    dV_flat = torch.empty((bs * n_kv_heads, seq_kv, HEAD_DIM),
                           dtype=torch.bfloat16, device=device)
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=device)

    # ── Step 3: Launch BMM1 on stream A, BMM2 on stream B concurrently ───────
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    default_stream = torch.cuda.current_stream()
    start_event = torch.cuda.Event()
    start_event.record(default_stream)

    # Stream A: BMM1
    with torch.cuda.stream(stream_a):
        stream_a.wait_event(start_event)
        torch.bmm(dO_2d, vs_T_2d, out=dP_dropped_2d)

    # Stream B: BMM2
    with torch.cuda.stream(stream_b):
        stream_b.wait_event(start_event)
        torch.bmm(P_dropped_2d_T, dO_2d, out=dV_flat)

    # ── Step 4: After BMM1 completes, run Triton softmax-bwd on stream A ─────
    dS_flat = dS.reshape(total_rows, seq_kv)
    dP_dropped_flat = dP_dropped_2d.reshape(total_rows, seq_kv)
    # dP_dropped_2d is contiguous -> reshape is zero-copy

    with torch.cuda.stream(stream_a):
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

    # ── Step 5: Sync both streams back to the default stream ─────────────────
    event_a = torch.cuda.Event()
    event_b = torch.cuda.Event()
    event_a.record(stream_a)
    event_b.record(stream_b)

    default_stream.wait_event(event_a)
    default_stream.wait_event(event_b)

    # ── Step 6: Reshape outputs ───────────────────────────────────────────────
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV
