"""
Optimized attention-backward kernel:
- Fused Triton kernel: computes dP = dO @ V^T AND softmax backward in one pass,
  eliminating the large intermediate dP_groups tensor entirely.
- dV BMM remains on a separate CUDA stream for overlap.
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
def fused_dp_softmax_bwd_kernel(
    # Inputs
    dO_ptr,           # [total_rows, HEAD_DIM]  bfloat16  — grad output, one row per query
    V_ptr,            # [total_rows_v, HEAD_DIM] bfloat16  — value states (kv-head grouped)
    P_ptr,            # [total_rows, seq_kv]    bfloat16  — attn_weights (post-softmax)
    mask_ptr,         # [total_rows, seq_kv]    bool      — dropout mask
    # Output
    dS_ptr,           # [total_rows, seq_kv]    bfloat16
    # Strides for dO: [total_rows, HEAD_DIM]
    stride_dO_row, stride_dO_dim,
    # Strides for V: rows are seq_kv, each row is one kv token
    # V is laid out as [n_kv_batch * seq_kv, HEAD_DIM]
    stride_V_row, stride_V_dim,
    # Strides for P and mask: [total_rows, seq_kv]
    stride_P_row, stride_P_col,
    stride_M_row, stride_M_col,
    # Strides for dS: [total_rows, seq_kv]
    stride_dS_row, stride_dS_col,
    # Params
    seq_kv,
    n_groups,         # 10 — queries per KV head
    scale,            # 1/(1-p_drop)
    HEAD_DIM: tl.constexpr,   # 128
    BLOCK_SKV: tl.constexpr,  # tile size over seq_kv dimension
):
    """
    Each program handles ONE query row (one of total_rows = bs*n_heads*seq_q).

    For this query row:
      1. Load dO[row] (shape [HEAD_DIM]) into registers.
      2. For each tile of seq_kv of size BLOCK_SKV:
         a. Load V[kv_start:kv_start+BLOCK_SKV, :] -> [BLOCK_SKV, HEAD_DIM]
         b. Compute dP_tile = dO @ V_tile^T -> [BLOCK_SKV]  (dot products)
         c. Store dP_tile temporarily... but we need the full row sum first.
      
      Two-pass strategy:
        Pass 1: compute dP tile by tile, accumulate row_sum = sum(dP * P)
        Pass 2: compute dS = P * (dP - row_sum) and store

      For correctness, we do two passes over seq_kv.
    """
    row_id = tl.program_id(0)

    # Which KV-head batch does this row belong to?
    # total_rows = bs * n_heads * seq_q
    # The V tensor is grouped: for row_id in [bs*n_heads*seq_q],
    # the corresponding KV batch index is row_id // (n_groups * seq_q)
    # but we can derive: kv_batch = row_id // (n_groups * seq_q)
    # where n_groups = n_heads / n_kv_heads
    # seq_q rows per head → kv_head_idx = (row_id // seq_q) // n_groups

    # Load dO row into registers: shape [HEAD_DIM]
    dO_base = row_id * stride_dO_row
    dim_offsets = tl.arange(0, HEAD_DIM)
    dO_row = tl.load(dO_ptr + dO_base + dim_offsets * stride_dO_dim).to(tl.float32)

    # Compute which KV batch row this query belongs to
    # row layout: [bs * n_heads * seq_q] with n_heads = n_groups * n_kv_heads
    # kv_batch_id = row_id // (n_groups * seq_q)  -- but n_groups*seq_q may not be power-of-2
    # We'll compute: head_id = row_id // seq_q; kv_batch_id = head_id // n_groups
    # But for the V pointer, we need: V[kv_batch_id * seq_kv + kv_token, :]
    # The V tensor is flat: [bs * n_kv_heads * seq_kv, HEAD_DIM]
    # So V_base_for_this_row = kv_batch_id * seq_kv * stride_V_row

    # We receive V_ptr as the start of V for the kv batch corresponding to row_id=0
    # We need to pass an offset. Instead, compute it here.
    # n_groups is a runtime param (not constexpr), so we use integer division.
    # seq_q is also not constexpr — we derive it from row_id patterns... 
    # Actually, let's pass it directly.
    # We handle this by accepting a pre-computed kv_batch_offset via a separate pointer.
    # Simpler: pass seq_q as a runtime param too.
    pass


@triton.jit
def fused_dp_softmax_bwd_kernel_v2(
    # Inputs - all flat 2D
    dO_ptr,           # [total_rows, HEAD_DIM]  bfloat16
    V_ptr,            # [n_kv_batches * seq_kv, HEAD_DIM] bfloat16
    P_ptr,            # [total_rows, seq_kv]    bfloat16
    mask_ptr,         # [total_rows, seq_kv]    bool
    # Output
    dS_ptr,           # [total_rows, seq_kv]    bfloat16
    # Params
    seq_q,            # number of query positions (runtime)
    seq_kv,           # number of key/value positions (runtime)
    n_groups,         # 10
    scale,            # 1/(1-p_drop)
    HEAD_DIM: tl.constexpr,   # 128
    BLOCK_SKV: tl.constexpr,  # tile size over seq_kv
):
    """
    Each Triton program handles ONE query row.
    total_rows = bs * n_heads * seq_q = bs * n_kv_heads * n_groups * seq_q

    Row layout matches attn_weights: [bs, n_heads, seq_q, seq_kv]
    flattened to [bs * n_heads * seq_q, seq_kv].

    V layout: [bs * n_kv_heads, seq_kv, HEAD_DIM] reshaped to
              [bs * n_kv_heads * seq_kv, HEAD_DIM].

    For row_id:
      head_id     = row_id // seq_q          (which attention head, 0..bs*n_heads-1)
      kv_batch_id = head_id // n_groups      (which KV batch, 0..bs*n_kv_heads-1)
      V_base      = kv_batch_id * seq_kv     (first V row for this KV head)
    """
    row_id = tl.program_id(0)

    # Compute KV batch for this row
    head_id     = row_id // seq_q
    kv_batch_id = head_id // n_groups
    V_base_row  = kv_batch_id * seq_kv   # first V row index for this KV head

    # Load dO for this row: [HEAD_DIM]
    dO_base = row_id * HEAD_DIM
    dim_offsets = tl.arange(0, HEAD_DIM)
    dO_row = tl.load(dO_ptr + dO_base + dim_offsets).to(tl.float32)

    # Base pointers for P, mask, dS
    P_base  = row_id * seq_kv
    M_base  = row_id * seq_kv
    dS_base = row_id * seq_kv

    # -------------------------------------------------------------------------
    # Pass 1: compute row_sum = sum_over_kv(dP_kv * P_kv)
    # where dP_kv = dO_row dot V[kv] (dot product of HEAD_DIM vectors)
    # -------------------------------------------------------------------------
    row_sum = tl.zeros([1], dtype=tl.float32)

    kv_offsets_base = tl.arange(0, BLOCK_SKV)

    for kv_start in range(0, seq_kv, BLOCK_SKV):
        kv_offsets = kv_start + kv_offsets_base
        kv_mask    = kv_offsets < seq_kv

        # Load V tile: [BLOCK_SKV, HEAD_DIM]
        V_row_indices = V_base_row + kv_offsets  # [BLOCK_SKV]
        # V is stored as [n_kv_batches*seq_kv, HEAD_DIM], row-major
        # V[V_row_idx, :] starts at V_ptr + V_row_idx * HEAD_DIM
        V_base_offsets = V_row_indices[:, None] * HEAD_DIM + dim_offsets[None, :]
        # V_base_offsets shape: [BLOCK_SKV, HEAD_DIM]
        V_tile = tl.load(
            V_ptr + V_base_offsets,
            mask=kv_mask[:, None] & tl.full([BLOCK_SKV, HEAD_DIM], True, dtype=tl.int1),
            other=0.0
        ).to(tl.float32)

        # dP_tile[k] = dot(dO_row, V_tile[k]) for each k in BLOCK_SKV
        # = sum over d of dO_row[d] * V_tile[k, d]
        dP_tile = tl.sum(dO_row[None, :] * V_tile, axis=1)  # [BLOCK_SKV]

        # Apply dropout: dP = dP_raw * mask * scale
        drop_m = tl.load(mask_ptr + M_base + kv_offsets, mask=kv_mask, other=0).to(tl.int1)
        dP_tile = tl.where(drop_m, dP_tile * scale, tl.zeros([BLOCK_SKV], dtype=tl.float32))

        # Load P tile
        p_tile = tl.load(P_ptr + P_base + kv_offsets, mask=kv_mask, other=0.0).to(tl.float32)

        # Accumulate row_sum
        row_sum += tl.sum(dP_tile * p_tile, axis=0)

    # -------------------------------------------------------------------------
    # Pass 2: compute dS = P * (dP - row_sum) and store
    # -------------------------------------------------------------------------
    rs = row_sum  # scalar

    for kv_start in range(0, seq_kv, BLOCK_SKV):
        kv_offsets = kv_start + kv_offsets_base
        kv_mask    = kv_offsets < seq_kv

        # Recompute V tile and dP_tile
        V_row_indices = V_base_row + kv_offsets
        V_base_offsets = V_row_indices[:, None] * HEAD_DIM + dim_offsets[None, :]
        V_tile = tl.load(
            V_ptr + V_base_offsets,
            mask=kv_mask[:, None] & tl.full([BLOCK_SKV, HEAD_DIM], True, dtype=tl.int1),
            other=0.0
        ).to(tl.float32)

        dP_tile = tl.sum(dO_row[None, :] * V_tile, axis=1)  # [BLOCK_SKV]

        drop_m = tl.load(mask_ptr + M_base + kv_offsets, mask=kv_mask, other=0).to(tl.int1)
        dP_tile = tl.where(drop_m, dP_tile * scale, tl.zeros([BLOCK_SKV], dtype=tl.float32))

        p_tile = tl.load(P_ptr + P_base + kv_offsets, mask=kv_mask, other=0.0).to(tl.float32)

        dS_tile = p_tile * (dP_tile - rs)
        tl.store(dS_ptr + dS_base + kv_offsets, dS_tile.to(tl.bfloat16), mask=kv_mask)


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
    # Step 1: Prepare dO in [bs*n_heads, seq_q, HEAD_DIM] layout.
    # =========================================================================
    # grad_attn_output: [bs, seq_q, 80, 128] -> permute -> [bs, 80, seq_q, 128]
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()
    # dO: [bs, 80, seq_q, 128], bfloat16, contiguous

    total_rows = bs * n_heads * seq_q

    # Flat views for the Triton kernel
    # dO_flat: [total_rows, HEAD_DIM]
    dO_flat = dO.reshape(total_rows, HEAD_DIM)

    # V_flat: [bs * n_kv_heads * seq_kv, HEAD_DIM]
    # value_states: [bs, 8, seq_kv, 128] -> [bs*8, seq_kv, 128] -> [bs*8*seq_kv, 128]
    vs_flat = value_states.reshape(bs * n_kv_heads * seq_kv, HEAD_DIM).contiguous()

    # P_flat: [total_rows, seq_kv]
    P_flat = attn_weights.reshape(total_rows, seq_kv)

    # mask_flat: [total_rows, seq_kv]
    dm_flat = dropout_mask.reshape(total_rows, seq_kv)

    # =========================================================================
    # Step 2: Launch dV on side stream concurrently.
    # dV: attn_weights_dropped.T @ dO -> [bs*8, skv, 128]
    # =========================================================================
    main_stream = torch.cuda.current_stream(device)
    side_stream, dO_ready_event = _get_side_stream(device)

    # Group reshape for dV bmm: [bs*8, 10*sq, skv]^T @ [bs*8, 10*sq, 128]
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    dV_flat = torch.empty(
        (bs * n_kv_heads, seq_kv, HEAD_DIM),
        dtype=torch.bfloat16, device=device
    )

    # Record event: inputs are ready on the main stream
    dO_ready_event.record(main_stream)

    # Side stream waits for inputs, then launches dV bmm
    side_stream.wait_event(dO_ready_event)
    with torch.cuda.stream(side_stream):
        torch.bmm(attn_groups_flat.transpose(-2, -1), dO_groups_flat, out=dV_flat)

    # =========================================================================
    # Step 3: Fused Triton kernel: dP = dO @ V^T + softmax backward.
    # Runs on main stream, overlapping with dV on side stream.
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    dS_flat = torch.empty((total_rows, seq_kv), dtype=torch.bfloat16, device=device)

    # Choose tile size for seq_kv dimension
    # Larger BLOCK_SKV = more registers but fewer kernel launches per row
    # 32 is a reasonable starting point — can be tuned
    BLOCK_SKV = 32

    grid = (total_rows,)

    fused_dp_softmax_bwd_kernel_v2[grid](
        dO_flat,         # [total_rows, HEAD_DIM]
        vs_flat,         # [bs*n_kv_heads*seq_kv, HEAD_DIM]
        P_flat,          # [total_rows, seq_kv]
        dm_flat,         # [total_rows, seq_kv]
        dS_flat,         # [total_rows, seq_kv]
        seq_q=seq_q,
        seq_kv=seq_kv,
        n_groups=n_groups,
        scale=scale,
        HEAD_DIM=HEAD_DIM,
        BLOCK_SKV=BLOCK_SKV,
    )

    dS = dS_flat.reshape(bs, n_heads, seq_q, seq_kv)

    # Wait for side stream (dV) to complete
    main_stream.wait_stream(side_stream)

    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV
