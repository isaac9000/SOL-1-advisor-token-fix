"""
Optimized attention-backward kernel:
- Both matmuls use group-reshape [bs*8, 10*sq, ...] — no GQA expansion anywhere.
- dP and dV bmms launched concurrently on separate CUDA streams.
- Module-level cached stream to avoid creation overhead in hot path.
- Pre-allocated output tensors before any stream switching.
- dV: direct attn.T @ dO -> [bs*8, skv, 128] (no post-transpose copy).
- Triton softmax-backward with correct two-pass dropout unmasking.
- All in bfloat16.

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

# Module-level cached CUDA stream and event (created once, reused every call)
_side_stream = None
_dO_ready_event = None

def _get_side_stream(device):
    global _side_stream, _dO_ready_event
    if _side_stream is None:
        _side_stream = torch.cuda.Stream(device)
        _dO_ready_event = torch.cuda.Event()
    return _side_stream, _dO_ready_event


@triton.jit
def fused_softmax_bwd_kernel(
    # Inputs
    dP_raw_ptr,       # [total_rows, seq_kv]  bfloat16  — raw dO @ V^T output
    P_ptr,            # [total_rows, seq_kv]  bfloat16  — attn_weights (post-softmax)
    mask_ptr,         # [total_rows, seq_kv]  bool      — dropout mask
    # Output
    dS_ptr,           # [total_rows, seq_kv]  bfloat16
    # Params
    total_rows,
    scale,            # 1/(1-p_drop)
    seq_kv,
    BLOCK_SKV: tl.constexpr,  # power-of-2 >= seq_kv for single-pass case
    USE_SINGLE_PASS: tl.constexpr,  # True if BLOCK_SKV >= seq_kv
):
    """
    Each program handles exactly ONE row.
    Grid = total_rows.

    Algorithm per row:
      1. Load dP_raw[row], mask[row], P[row]
      2. dp = dP_raw * mask * scale   (dropout unmask)
      3. row_sum = sum(dp * P)
      4. dS = P * (dp - row_sum)
      5. Store dS[row]

    Two-pass path used when seq_kv > BLOCK_SKV.
    """
    row_id = tl.program_id(0)
    if row_id >= total_rows:
        return

    base = row_id * seq_kv

    if USE_SINGLE_PASS:
        # Single pass: BLOCK_SKV >= seq_kv
        offsets = tl.arange(0, BLOCK_SKV)
        mask_bounds = offsets < seq_kv

        dp_raw = tl.load(dP_raw_ptr + base + offsets,
                         mask=mask_bounds, other=0.0).to(tl.float32)
        drop_m = tl.load(mask_ptr + base + offsets,
                         mask=mask_bounds, other=0).to(tl.int1)
        p = tl.load(P_ptr + base + offsets,
                    mask=mask_bounds, other=0.0).to(tl.float32)

        dp = tl.where(drop_m, dp_raw * scale, 0.0)
        row_sum = tl.sum(dp * p, axis=0)
        ds = p * (dp - row_sum)

        tl.store(dS_ptr + base + offsets, ds.to(tl.bfloat16), mask=mask_bounds)
    else:
        # Two-pass: seq_kv > BLOCK_SKV
        # Pass 1: compute row_sum = sum(dp * P)
        row_sum = 0.0
        for block_start in range(0, seq_kv, BLOCK_SKV):
            offsets = block_start + tl.arange(0, BLOCK_SKV)
            mask_bounds = offsets < seq_kv
            dp_raw = tl.load(dP_raw_ptr + base + offsets,
                             mask=mask_bounds, other=0.0).to(tl.float32)
            drop_m = tl.load(mask_ptr + base + offsets,
                             mask=mask_bounds, other=0).to(tl.int1)
            p = tl.load(P_ptr + base + offsets,
                        mask=mask_bounds, other=0.0).to(tl.float32)
            dp = tl.where(drop_m, dp_raw * scale, 0.0)
            row_sum = row_sum + tl.sum(dp * p, axis=0)

        # Pass 2: compute dS and store
        for block_start in range(0, seq_kv, BLOCK_SKV):
            offsets = block_start + tl.arange(0, BLOCK_SKV)
            mask_bounds = offsets < seq_kv
            dp_raw = tl.load(dP_raw_ptr + base + offsets,
                             mask=mask_bounds, other=0.0).to(tl.float32)
            drop_m = tl.load(mask_ptr + base + offsets,
                             mask=mask_bounds, other=0).to(tl.int1)
            p = tl.load(P_ptr + base + offsets,
                        mask=mask_bounds, other=0.0).to(tl.float32)
            dp = tl.where(drop_m, dp_raw * scale, 0.0)
            ds = p * (dp - row_sum)
            tl.store(dS_ptr + base + offsets, ds.to(tl.bfloat16), mask=mask_bounds)


def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


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
    # Step 1: Make dO contiguous in [bs, 80, sq, 128] layout (bfloat16).
    # =========================================================================
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()
    # dO: [bs, 80, sq, 128], bfloat16, contiguous

    # Shared group-reshape for both matmuls: [bs*8, 10*sq, 128] — free view
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # Prepare matmul operands (all free views, no copies)
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # =========================================================================
    # Pre-allocate output tensors on the CURRENT stream before any switching.
    # =========================================================================
    # dP output: [bs*8, 10*sq, skv] — raw dO @ V^T
    dP_groups = torch.empty(
        (bs * n_kv_heads, n_groups * seq_q, seq_kv),
        dtype=torch.bfloat16, device=device
    )
    # dV output: [bs*8, skv, 128]
    dV_flat = torch.empty(
        (bs * n_kv_heads, seq_kv, HEAD_DIM),
        dtype=torch.bfloat16, device=device
    )

    # =========================================================================
    # Step 2: Concurrent stream execution.
    # - Side stream: dV bmm (attn.T @ dO → directly contiguous [bs*8, skv, 128])
    # - Main stream: dP bmm → Triton softmax backward
    # =========================================================================
    main_stream = torch.cuda.current_stream(device)
    side_stream, dO_ready_event = _get_side_stream(device)

    # Record event: dO is ready on the main stream
    dO_ready_event.record(main_stream)

    # Side stream waits for dO to be ready, then launches dV
    side_stream.wait_event(dO_ready_event)
    with torch.cuda.stream(side_stream):
        # dV: bmm([bs*8, skv, 10*sq], [bs*8, 10*sq, 128]) -> [bs*8, skv, 128]
        torch.bmm(attn_groups_flat.transpose(-2, -1), dO_groups_flat, out=dV_flat)

    # Launch dP on main stream (concurrent with dV on side stream)
    # dP_raw: bmm([bs*8, 10*sq, 128], [bs*8, 128, skv]) -> [bs*8, 10*sq, skv]
    torch.bmm(dO_groups_flat, vs_flat.transpose(-2, -1), out=dP_groups)

    # =========================================================================
    # Step 3: Triton softmax backward + dropout correction.
    # Runs on main stream — overlaps with dV on side stream.
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    total_rows = bs * n_heads * seq_q

    # Note: dP_groups is [bs*8, 10*sq, skv] — reshape to [bs*80, sq, skv] matches
    # attn_weights [bs, 80, sq, skv] and dropout_mask [bs, 80, sq, skv].
    # Both are already contiguous with matching row order.
    dP_raw_flat = dP_groups.reshape(total_rows, seq_kv)
    P_flat      = attn_weights.reshape(total_rows, seq_kv)
    dm_flat     = dropout_mask.reshape(total_rows, seq_kv)

    dS_flat = torch.empty((total_rows, seq_kv), dtype=torch.bfloat16, device=device)

    # Determine BLOCK_SKV: smallest power-of-2 >= seq_kv (capped at 4096 for memory)
    # If seq_kv fits in a single block, use single-pass; otherwise two-pass.
    MAX_BLOCK = 4096
    p2 = _next_power_of_2(seq_kv)
    if p2 <= MAX_BLOCK:
        BLOCK_SKV = p2
        USE_SINGLE_PASS = True
    else:
        # Two-pass with a fixed block size
        BLOCK_SKV = MAX_BLOCK
        USE_SINGLE_PASS = False

    grid = (total_rows,)

    fused_softmax_bwd_kernel[grid](
        dP_raw_flat, P_flat, dm_flat, dS_flat,
        total_rows=total_rows,
        scale=scale,
        seq_kv=seq_kv,
        BLOCK_SKV=BLOCK_SKV,
        USE_SINGLE_PASS=USE_SINGLE_PASS,
    )

    dS = dS_flat.reshape(bs, n_heads, seq_q, seq_kv)

    # Wait for side stream (dV) to complete
    main_stream.wait_stream(side_stream)

    # dV_flat is already [bs*8, skv, 128] contiguous — just reshape
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV
