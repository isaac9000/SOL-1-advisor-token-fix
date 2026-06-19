# Attn-Bwd Autoresearch

An advisor-worker agent pair that iteratively optimizes a CUDA kernel for the attention backward pass on NVIDIA B200. Each iteration the **advisor** reviews experiment history and proposes a strategic direction; the **worker** implements exactly one change to `submission.py`, evaluates it on a B200 via Modal, logs the result, and stops. The outer loop drives the next iteration.

## Task

Implement the fastest possible **attention backward pass** for a GQA transformer layer (NVIDIA SOL-ExecBench kernel #1):

```
dO = grad_attn_output.transpose(1, 2).float()          # [bs, 80, sq, 128]
dP̃ = dO @ value_states_expanded^T                      # [bs, 80, sq, skv]
dP = dP̃ * dropout_mask / (1 - p)
dS = P * (dP - sum(dP * P, dim=-1))                    # softmax backward
dV_exp = attn_weights_dropped^T @ dO                   # [bs, 80, skv, 128]
grad_value_states = dV_exp.reshape(bs,8,10,skv,128).sum(dim=2)
return grad_attn_scores.to(bf16), grad_value_states.to(bf16)
```

`custom_kernel` receives a tuple `(grad_attn_output, attn_weights, attn_weights_dropped, value_states, dropout_mask, attention_dropout)` and returns `(grad_attn_scores, grad_value_states)`:

| Argument | Shape | Dtype |
|---|---|---|
| `grad_attn_output` | `[bs, seq_q, 80, 128]` | bfloat16 |
| `attn_weights` | `[bs, 80, seq_q, seq_kv]` | bfloat16 |
| `attn_weights_dropped` | `[bs, 80, seq_q, seq_kv]` | bfloat16 |
| `value_states` | `[bs, 8, seq_kv, 128]` | bfloat16 |
| `dropout_mask` | `[bs, 80, seq_q, seq_kv]` | bool |
| `attention_dropout` | scalar | float32 |
| return `grad_attn_scores` | `[bs, 80, seq_q, seq_kv]` | bfloat16 |
| return `grad_value_states` | `[bs, 8, seq_kv, 128]` | bfloat16 |

Fixed architecture: 80 attention heads, 8 KV heads, 10 groups/KV head, head_dim=128, dropout=0.1.

**Benchmark cases (16 total) — from NVIDIA SOL-ExecBench:**

| # | bs | seq_q | seq_kv | Baseline (μs) | SOL (μs) |
|---|-----|-------|--------|---------------|----------|
| 1 | 4   | 256   | 256    | 89.7          | 20.1     |
| 2 | 8   | 373   | 449    | 840.8         | 94.2     |
| 3 | 4   | 1024  | 2048   | 3208.3        | 540.9    |
| 4 | 64  | 128   | 128    | 1641.4        | 92.3     |
| 5 | 2   | 256   | 512    | 211.1         | 18.7     |
| 6 | 32  | 691   | 773    | 9273.8        | 1142.7   |
| 7 | 8   | 128   | 128    | 256.4         | 11.9     |
| 8 | 32  | 512   | 512    | 4250.2        | 578.1    |
| 9 | 4   | 211   | 293    | 266.7         | 18.8     |
| 10 | 8  | 256   | 256    | 509.0         | 39.8     |
| 11 | 16 | 128   | 256    | 485.2         | 40.9     |
| 12 | 1  | 1024  | 1024   | 354.7         | 69.3     |
| 13 | 16 | 256   | 512    | 1109.1        | 147.0    |
| 14 | 32 | 128   | 128    | 840.4         | 46.4     |
| 15 | 1  | 512   | 512    | 133.3         | 18.5     |
| 16 | 1  | 4096  | 4096   | 4567.9        | 1063.8   |

All 16 cases are used for both correctness testing and benchmarking. Correctness tolerance: `rtol=1e-2, atol=1e-2`. Score = `756 / geomean_us` (≈1.0 at baseline, ≈9.3 at SOL).

## Setup

```bash
uv sync

# Configure Modal credentials
uv run modal token set --token-id <token-id> --token-secret <token-secret>

# Deploy the B200 evaluator (once, before any agent runs)
uv run modal deploy eval_modal_attn_bwd.py
```

Create a `.env` file in the repo root:

```
ANTHROPIC_API_KEY=...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
AUTORESEARCH_MODEL=claude-opus-4-7     # optional, this is the default
```

## Running the agent

```bash
bash run_agent.sh
```

Or directly:

```bash
uv run attn_bwd/agent.py --baseline attn_bwd/starting_point.py --iterations 25
```

Quick correctness check without a full benchmark:

```bash
cd attn_bwd
uv run python run_eval.py submission.py -o results.json --mode test
```

## Structure

```
eval_modal_attn_bwd.py   — deployable Modal B200 evaluator
attn_bwd/
├── agent.py             — advisor-worker agentic loop (direct Anthropic SDK)
├── advisor_prompt.md    — advisor system prompt: strategy, comparison discipline
├── worker_prompt.md     — worker system prompt: task spec, mandatory sequence, rules
├── submission.py        — the kernel file the worker edits each iteration
├── starting_point.py    — baseline PyTorch kernel to seed each run
├── run_eval.py          — submits submission.py to the deployed Modal evaluator
├── tools.py             — logging, plotting, and get_experiment_history tool
└── runs/                — one directory per run: history, TSV log, plots, best submission
```

Each run directory contains:
- `experiment_history.md` — full log of every attempt with code and result
- `results.tsv` — tab-separated summary for plotting
- `progress.png` — latency scatter plot updated each experiment; shows keep/discard/crash points, best-time step line, baseline and SOL reference lines, and cumulative LLM call count
- `iterations.png` — best latency per advisor iteration
- `best_submission.py` — snapshot of the fastest kernel found so far
- `proposals.md` — advisor proposals for every iteration
- `snapshot_iter{N}.py` — per-iteration snapshot of submission.py before the worker edits it
