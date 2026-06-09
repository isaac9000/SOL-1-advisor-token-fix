# TriMul Autoresearch

An advisor-worker agent pair that iteratively optimizes a CUDA kernel for the Triangle Multiplicative Update (TriMul) operator on NVIDIA H100. Each iteration the **advisor** reviews experiment history and proposes a strategic direction; the **worker** implements it, evaluates on an H100 via Modal, and logs the result.

## Task

Implement the fastest possible **outgoing** TriMul operator from AlphaFold3 — a core operation in protein structure prediction models (AlphaFold3, Chai, Protenix).

`custom_kernel` receives `(input_tensor, mask, weights, config)` and must return a float32 tensor of shape `[bs, seqlen, seqlen, dim]`:

```
x     = LayerNorm(input_tensor)
left  = left_proj(x) * left_gate(x).sigmoid()
right = right_proj(x) * right_gate(x).sigmoid()
left  = left  * mask.unsqueeze(-1)
right = right * mask.unsqueeze(-1)
out   = einsum("... i k d, ... j k d -> ... i j d", left, right)
out   = LayerNorm(out) * out_gate(x).sigmoid()
return to_out(out)
```

**Benchmark shapes:**

| seqlen | bs | dim | hiddendim | distribution |
|--------|-----|-----|-----------|--------------|
| 256    | 2   | 128 | 128       | normal       |
| 768    | 1   | 128 | 128       | cauchy       |
| 256    | 2   | 384 | 128       | normal       |
| 512    | 1   | 128 | 128       | normal       |
| 1024   | 1   | 128 | 128       | cauchy       |
| 768    | 1   | 384 | 128       | normal       |
| 1024   | 1   | 384 | 128       | normal       |

Ranked by geometric mean latency across all seven benchmark shapes (lower is better). Score = 3000 / geomean_us.

## Setup

```bash
uv sync
```

Create a `.env` file in the repo root:

```
ANTHROPIC_API_KEY=...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
AUTORESEARCH_MODEL=claude-sonnet-4-6   # optional, this is the default
```

Deploy the H100 evaluator (once, before any agent runs):

```bash
uv run modal deploy eval_modal_trimul.py
```

## Running the agent

```bash
uv run trimul/agent.py --iterations 20
```

Start from the provided starting point:

```bash
uv run trimul/agent.py --baseline trimul/starting_point.py --iterations 20
```

Use different models for advisor and worker:

```bash
uv run trimul/agent.py --advisor-model claude-opus-4-8 --worker-model claude-sonnet-4-6 --iterations 20
```

Or use the provided script (checks for H100 then launches in tmux):

```bash
./run_agent.sh
```

Evaluate a kernel file without running the agent:

```bash
cd trimul
python run_eval.py submission.py -o results.json
python run_eval.py submission.py -o results.json --mode test   # correctness only
```

## Structure

```
eval_modal_trimul.py      — deployable Modal H100 evaluator
run_agent.sh              — H100 check + tmux agent launcher
trimul/
├── agent.py              — advisor-worker agentic loop
├── advisor_prompt.md     — advisor system prompt: strategy, comparison discipline
├── worker_prompt.md      — worker system prompt: mandatory sequence, rules
├── submission.py         — the kernel file the worker edits each iteration
├── starting_point.py     — original PyTorch/BF16 baseline
├── run_eval.py           — submits submission.py to the deployed Modal evaluator
├── tools.py              — log_experiment and get_experiment_history tools
└── runs/                 — one directory per run: history, TSV log, plots, best submission
```

Each run directory contains:
- `experiment_history.md` — full log of every attempt with code and result
- `results.tsv` — tab-separated summary for plotting
- `progress.png` — latency scatter plot updated each experiment; shows keep/discard/crash points, best-time step line, and cumulative LLM call count
- `iterations.png` — best latency per advisor iteration
- `best_submission.py` — snapshot of the fastest kernel found so far
- `proposals.md` — advisor proposals for every iteration
- `snapshot_iter{N}.py` — per-iteration snapshot of submission.py before the worker edits it

## LLM Call Counter

The agent tracks how many times the LLM is invoked across both the advisor and worker agents (each tool-calling turn and each plain response counts as one call). This is reported:

- **Per-iteration** in the console: `[advisor]` and `[worker]` call counts accumulated into a running total
- **At each checkpoint** (every `--checkpoint-every` iterations): `LLM calls (total): T`
- **In the final report**: `LLM calls (total): T`
- **On `progress.png`**: displayed as a badge in the bottom-right corner of every plot, updated live as experiments are logged
