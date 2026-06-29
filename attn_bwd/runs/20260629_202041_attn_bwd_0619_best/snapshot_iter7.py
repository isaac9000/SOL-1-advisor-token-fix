"""
Optimized attention-backward kernel:
- Two concurrent CUDA streams for the two BMMs (dP and dV)
- Triton softmax-backward kernel: one program per row, single-pass vectorized,
  with block sizes tuned for powers-of-2 at/above actual seq_kv.
- GQA handled via reshape/expand (no actual data copy for read-only expansion).

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    value_states           [bs,  8, seq_kv, 128]    bfloat16
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


# ---------------------------------------------------------------------------
# Triton softmax-backward kernel: one program per row
# Each program handles exactly one (bs, head, seq_q) row of length seq_kv.
# BLOCK_SKV is a constexpr power-of-2 >= seq_kv, chosen at launch time.
# Single-pass: load dP, mask, P -> compute reduction -> write dS.
# ---------------------------------------------------------------------------
@triton.jit
def _softmax_bwd_kernel(
    # dP = dP_dropped (already dropout-masked and scaled)
    dP_ptr,       # [N_rows, seq_kv]  float32
    P_ptr,        # [N_rows, seq_kv]  float32
    dS_ptr,       # [N_rows, seq_kv]  float32 (output)
    seq_kv: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    row = tl.program_id(0)

    row_start = row * seq_kv
    offs = tl.arange(0, BLOCK_SKV)
    mask = offs < seq_kv

    # Load dP and P for this row
    dP = tl.load(dP_ptr + row_start + offs, mask=mask, other=0.0).to(tl.float32)
    P  = tl.load(P_ptr  + row_start + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute sum(dP * P) for softmax backward
    dP_P = dP * P
    sum_dP_P = tl.sum(dP_P, axis=0)

    # dS = P * (dP - sum_dP_P)
    dS = P * (dP - sum_dP_P)

    tl.store(dS_ptr + row_start + offs, dS, mask=mask)


def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


# Pre-create two persistent CUDA streams for concurrent kernel execution
_stream1 = None
_stream2 = None


def _get_streams():
    global _stream1, _stream2
    if _stream1 is None:
        _stream1 = torch.cuda.Stream()
        _stream2 = torch.cuda.Stream()
    return _stream1, _stream2


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs    = grad_attn_output.shape[0]
    seq_q = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_heads    = NUM_ATTENTION_HEADS    # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS    # 8
    n_groups   = 10

    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # dO: [bs, seq_q, 80, 128] -> [bs, 80, seq_q, 128], float32, contiguous
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous().float()  # [bs,80,sq,128]

    # Flat views for BMMs
    dO_flat = dO.reshape(bs * n_heads, seq_q, HEAD_DIM)              # [bs*80, sq, 128]

    # GQA expand for dP BMM: vs [bs,8,skv,128] -> [bs*80, skv, 128]
    vs = value_states.float()  # [bs, 8, skv, 128]
    vs_exp = vs.unsqueeze(2).expand(bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM)
    vs_exp = vs_exp.reshape(bs * n_heads, seq_kv, HEAD_DIM)  # contiguous not needed for expand read

    # attn_weights_dropped flat
    attn_flat = attn_weights_dropped.float().reshape(bs * n_heads, seq_q, seq_kv)

    stream1, stream2 = _get_streams()
    current_stream = torch.cuda.current_stream()

    # Record event on current stream so both sub-streams wait for inputs
    evt_start = torch.cuda.Event()
    evt_start.record(current_stream)

    # ---- Stream 1: compute dP = dO @ V^T ----
    with torch.cuda.stream(stream1):
        stream1.wait_event(evt_start)
        # dP_flat: [bs*80, sq, skv]
        dP_flat = torch.bmm(dO_flat, vs_exp.transpose(-2, -1))

    # ---- Stream 2: compute dV = attn_weights_dropped^T @ dO ----
    with torch.cuda.stream(stream2):
        stream2.wait_event(evt_start)
        # dV_flat: [bs*80, skv, 128]
        dV_flat = torch.bmm(attn_flat.transpose(-2, -1), dO_flat)

    # Wait for dP before running softmax-bwd Triton kernel
    evt_dp = torch.cuda.Event()
    evt_dp.record(stream1)
    current_stream.wait_event(evt_dp)

    # Apply dropout mask and scale to get dP_dropped (float32)
    dP = dP_flat.reshape(bs, n_heads, seq_q, seq_kv)
    dP_dropped = dP * dropout_mask.float() * scale  # [bs, 80, sq, skv]

    # P in float32
    P = attn_weights.float()  # [bs, 80, sq, skv]

    # Flatten rows for Triton kernel
    N_rows = bs * n_heads * seq_q
    dP_2d = dP_dropped.reshape(N_rows, seq_kv).contiguous()
    P_2d  = P.reshape(N_rows, seq_kv).contiguous()
    dS_2d = torch.empty_like(dP_2d)

    BLOCK_SKV = _next_power_of_2(seq_kv)
    # Cap at 65536 (max Triton block)
    if BLOCK_SKV > 65536:
        BLOCK_SKV = 65536

    grid = (N_rows,)
    _softmax_bwd_kernel[grid](
        dP_2d, P_2d, dS_2d,
        seq_kv=seq_kv,
        BLOCK_SKV=BLOCK_SKV,
        num_warps=min(16, max(1, BLOCK_SKV // 32)),
    )

    dS = dS_2d.reshape(bs, n_heads, seq_q, seq_kv).to(torch.bfloat16)

    # Wait for dV from stream 2
    evt_dv = torch.cuda.Event()
    evt_dv.record(stream2)
    current_stream.wait_event(evt_dv)

    dV = dV_flat.reshape(bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM).sum(dim=2).to(torch.bfloat16)

    return dS, dV
