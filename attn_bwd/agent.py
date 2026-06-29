"""
Advisor-Worker agentic loop for attention-backward kernel optimization.

Direct Anthropic SDK implementation — no LangGraph or deepagents.
Evaluation runs outside the loop: the orchestrator calls run_eval.py
after each worker turn and handles all logging.

Usage:
    uv run attn_bwd/agent.py
    uv run attn_bwd/agent.py --iterations 20 --baseline attn_bwd/starting_point.py
    uv run attn_bwd/agent.py --advisor-model claude-opus-4-7 --worker-model claude-sonnet-4-6
"""

import argparse
import copy
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

import tools as _tools
from tools import (
    _update_plot,
    _log_experiment_direct,
    set_run_directory,
    set_agent_iteration,
    set_llm_call_count,
)

PROJECT_DIR    = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT      = os.path.dirname(PROJECT_DIR)
SUBMISSION_FILE = os.path.join(PROJECT_DIR, "submission.py")
RESULTS_FILE   = os.path.join(PROJECT_DIR, "results.json")


def load_prompt(filename: str) -> str:
    with open(os.path.join(PROJECT_DIR, filename)) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Token usage tracking (consistent with SOL-1-BES)
# ---------------------------------------------------------------------------

_token_lock = threading.Lock()
_token_totals: dict = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}
_token_file: Path | None = None


def _flush_tokens() -> None:
    if _token_file is not None:
        _token_file.write_text(json.dumps(_token_totals, indent=2))


def _record(usage) -> None:
    with _token_lock:
        _token_totals["input_tokens"] += getattr(usage, "input_tokens", 0)
        _token_totals["output_tokens"] += getattr(usage, "output_tokens", 0)
        _token_totals["api_calls"] += 1
        _flush_tokens()


def _patch_anthropic() -> None:
    _orig_sync = anthropic.resources.messages.Messages.create

    def _sync_create(self, *args, **kwargs):
        response = _orig_sync(self, *args, **kwargs)
        if getattr(response, "usage", None):
            _record(response.usage)
        return response

    anthropic.resources.messages.Messages.create = _sync_create

    _orig_async = anthropic.resources.messages.AsyncMessages.create

    async def _async_create(self, *args, **kwargs):
        response = await _orig_async(self, *args, **kwargs)
        if getattr(response, "usage", None):
            _record(response.usage)
        return response

    anthropic.resources.messages.AsyncMessages.create = _async_create


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

ADVISOR_TOOLS = [
    {
        "name": "get_experiment_history",
        "description": (
            "Read the full experiment history markdown. "
            "Returns every prior kernel attempt, its code, hypothesis, and result. "
            "Call this before proposing a new approach to avoid repeating failures."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }
]

WORKER_TOOLS = [
    {
        "name": "read_file",
        "description": "Read any file in the workspace by absolute or relative path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write new content to submission.py, replacing it entirely. "
            "Only submission.py may be written."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Complete new content for submission.py.",
                }
            },
            "required": ["content"],
        },
    },
]


def _dispatch_tool(name: str, args: dict) -> str:
    if name == "get_experiment_history":
        return _tools.get_experiment_history()
    elif name == "read_file":
        path = args.get("path", "")
        if not os.path.isabs(path):
            path = os.path.join(PROJECT_DIR, path)
        try:
            with open(path) as f:
                return f.read()
        except Exception as e:
            return f"Error reading {path}: {e}"
    elif name == "write_file":
        content = args.get("content", "")
        try:
            with open(SUBMISSION_FILE, "w") as f:
                f.write(content)
            return "submission.py written successfully."
        except Exception as e:
            return f"Error writing submission.py: {e}"
    else:
        return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Agent turn runner
# ---------------------------------------------------------------------------


def run_agent_turn(
    client: anthropic.Anthropic,
    history: list,
    system: str,
    tools: list,
    model: str,
    label: str,
    max_tokens: int = 8096,
) -> tuple[str, int]:
    """Run one agent turn to completion with tool-calling loop."""
    n_calls = 0

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=history,
        )
        n_calls += 1

        content_blocks = []
        for block in response.content:
            if block.type == "text":
                content_blocks.append({"type": "text", "text": block.text})
                if block.text.strip():
                    print(f"  [{label}] {block.text[:200]}", flush=True)
            elif block.type == "tool_use":
                content_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                print(f"  [{label}] {block.name}({str(block.input)[:120]})", flush=True)

        history.append({"role": "assistant", "content": content_blocks})

        if response.stop_reason == "end_turn":
            text = " ".join(
                b.text for b in response.content if b.type == "text"
            ).strip()
            return text, n_calls

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = _dispatch_tool(block.name, block.input)
                    print(f"  [{label}] → {str(result)[:200]}", flush=True)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })
            history.append({"role": "user", "content": tool_results})
        else:
            break

    return "", n_calls


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in ("timeout", "timed out", "connection reset",
                                     "read operation timed out"))


def run_agent_turn_retrying(
    client: anthropic.Anthropic,
    history: list,
    system: str,
    tools: list,
    model: str,
    label: str,
    max_attempts: int = 3,
    base_delay: float = 15.0,
) -> tuple[str, int]:
    """Like run_agent_turn but retries transient API errors."""
    last_exc: Exception | None = None
    total_calls = 0

    for attempt in range(max_attempts):
        if attempt > 0:
            delay = base_delay * (2 ** (attempt - 1))
            print(
                f"  [{label}] Retrying (attempt {attempt + 1}/{max_attempts}) "
                f"in {delay:.0f}s...",
                flush=True,
            )
            time.sleep(delay)

        history_snapshot = copy.deepcopy(history)
        try:
            text, n = run_agent_turn(client, history, system, tools, model, label)
            return text, total_calls + n
        except Exception as e:
            total_calls += 1
            history.clear()
            history.extend(history_snapshot)
            if _is_transient_error(e) and attempt < max_attempts - 1:
                print(
                    f"  [{label}] Transient error on attempt {attempt + 1}: "
                    f"{type(e).__name__}: {str(e)[:150]}",
                    flush=True,
                )
                last_exc = e
            else:
                raise

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_submission() -> tuple[float, str]:
    """Run run_eval.py and return (geomean_us, error_msg). geomean_us=0 on failure."""
    print("[evaluator] running run_eval.py...", flush=True)
    try:
        ret = subprocess.run(
            [sys.executable, "run_eval.py", "submission.py", "-o", "results.json"],
            cwd=PROJECT_DIR,
            timeout=660,
        )
    except subprocess.TimeoutExpired:
        return 0.0, "run_eval.py timed out after 660s"
    except Exception as e:
        return 0.0, f"run_eval.py failed to start: {e}"

    try:
        with open(RESULTS_FILE) as f:
            md = json.load(f)
        text = md if isinstance(md, str) else ""

        if "> ❌ Testing failed" in text or "> ❌ Benchmarking failed" in text:
            err_m = re.search(r"## Error:\s*```\s*(.*?)\s*```", text, re.DOTALL)
            detail = err_m.group(1).strip()[:400] if err_m else ""
            label = ("Correctness check failed" if "> ❌ Testing failed" in text
                     else "Benchmark correctness failed")
            return 0.0, f"{label}. {detail}".strip()

        m = re.search(r"Geometric mean: ⏱ ([\d.]+)", text)
        if m and ret.returncode == 0:
            return float(m.group(1)), ""

        err_m = re.search(r"## Error:\s*```\s*(.*?)\s*```", text, re.DOTALL)
        error = (err_m.group(1).strip()[:500] if err_m
                 else f"run_eval exited {ret.returncode}")
        return 0.0, error
    except Exception as e:
        return 0.0, f"could not parse results.json: {e}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_hypothesis(worker_text: str) -> str:
    m = re.search(r"Implemented:\s*(.+?)(?:\n|$)", worker_text)
    if m:
        return m.group(1).strip()[:200]
    for line in worker_text.splitlines():
        if line.strip():
            return line.strip()[:200]
    return "Worker implementation"


def read_results_summary() -> str:
    if not os.path.exists(_tools.TSV_FILE):
        return "No experiments run yet."
    with open(_tools.TSV_FILE) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return "No experiments run yet."

    total = len(lines) - 1
    keeps, discards, crashes = [], 0, 0
    best_time = float("inf")
    best_desc = ""

    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        it, time_str, status = parts[0], parts[3], parts[4]
        desc = parts[5] if len(parts) > 5 else ""
        try:
            t = float(time_str)
        except ValueError:
            t = 0.0
        if status == "keep" and t > 0:
            keeps.append((int(it) if it.isdigit() else 0, t, desc))
            if t < best_time:
                best_time, best_desc = t, desc
        elif status == "discard":
            discards += 1
        elif status == "crash":
            crashes += 1

    summary = f"=== EXPERIMENT SUMMARY ({total} total) ===\n"
    if best_time < float("inf"):
        summary += f"Best time: {best_time:.2f} μs — {best_desc[:80]}\n"
    else:
        summary += "Best time: none yet\n"
    summary += f"Keeps: {len(keeps)} | Discards: {discards} | Crashes: {crashes}\n"
    if keeps:
        summary += "Keep history:\n"
        for it, t, d in keeps[-10:]:
            summary += f"  #{it}: {t:.2f}μs — {d[:60]}\n"
    return summary


def save_proposals(run_dir: str, proposals: list) -> None:
    with open(os.path.join(run_dir, "proposals.md"), "w") as f:
        f.write("# Advisor Proposals\n\n")
        for iteration, proposal in proposals:
            f.write(f"---\n\n## Iteration {iteration}\n\n{proposal}\n\n")


def print_checkpoint(
    iteration: int, total: int, start_time: float, llm_call_count: int = 0
) -> None:
    elapsed_min = (time.time() - start_time) / 60
    rate = iteration / elapsed_min if elapsed_min > 0 else 0
    summary = read_results_summary()
    print(f"\n{'#'*60}")
    print(f"  CHECKPOINT — Iteration {iteration}/{total}")
    print(f"  Elapsed: {elapsed_min:.1f} min | Rate: {rate:.1f} iter/min")
    print(f"  LLM calls (total): {llm_call_count}")
    print(f"{'#'*60}")
    print(summary)
    try:
        _update_plot()
    except Exception as e:
        print(f"  Plot update failed: {e}")
    print(f"{'#'*60}\n")


def print_final_report(
    total_iterations: int, actual_iterations: int, start_time: float,
    llm_call_count: int = 0,
) -> None:
    elapsed_min = (time.time() - start_time) / 60
    print(f"\n{'='*60}\n  FINAL REPORT\n{'='*60}")
    print(f"  Iterations: {actual_iterations}/{total_iterations} | Time: {elapsed_min:.1f} min")
    print(f"  LLM calls (total): {llm_call_count}")
    print(read_results_summary())
    try:
        _update_plot()
    except Exception:
        pass
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Advisor-Worker attn_bwd Optimization Agent"
    )
    parser.add_argument("--iterations", "-n", type=int, default=20)
    parser.add_argument("--checkpoint-every", "-c", type=int, default=5)
    parser.add_argument("--baseline", "-b", default=None, help="Path to baseline file")
    parser.add_argument("--advisor-model", default=None)
    parser.add_argument("--worker-model", default=None)
    parser.add_argument("--max-tokens", "-t", type=int, default=None,
                        help="Stop after this many total tokens (input + output)")
    args = parser.parse_args()

    load_dotenv(os.path.join(REPO_ROOT, ".env"))

    default_model  = os.environ.get("AUTORESEARCH_MODEL", "claude-sonnet-4-6")
    advisor_model  = args.advisor_model or default_model
    worker_model   = args.worker_model or default_model

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    baseline_path, baseline_name = None, "scratch"
    if args.baseline:
        baseline_path = os.path.abspath(args.baseline)
        if not os.path.isfile(baseline_path):
            print(f"Error: baseline not found: {baseline_path}")
            sys.exit(1)
        baseline_name = os.path.splitext(os.path.basename(baseline_path))[0]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(PROJECT_DIR, "runs",
                           f"{timestamp}_attn_bwd_{baseline_name}")
    os.makedirs(run_dir, exist_ok=True)
    set_run_directory(run_dir)

    global _token_file
    _token_file = Path(run_dir) / "token_usage.json"
    _patch_anthropic()

    if baseline_path:
        shutil.copy2(baseline_path, SUBMISSION_FILE)
        print(f"Copied baseline '{baseline_name}' -> submission.py", flush=True)
    else:
        print("No baseline — using current submission.py.", flush=True)

    client         = anthropic.Anthropic()
    advisor_system = load_prompt("advisor_prompt.md")
    worker_system  = load_prompt("worker_prompt.md")

    advisor_history: list = []
    worker_history: list  = []

    print(f"Starting advisor-worker attn_bwd optimization loop")
    print(f"  Advisor model:  {advisor_model}")
    print(f"  Worker model:   {worker_model}")
    print(f"  Baseline:       {baseline_name}")
    print(f"  Run dir:        {run_dir}")
    print(f"  Iterations:     {args.iterations}")
    print()

    def _sigterm_handler(signum, frame):
        print("\n--- SIGTERM ---", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    start_time   = time.time()
    current_best = float("inf")

    kickoff_note = ""
    if baseline_path:
        print(f"Benchmarking baseline '{baseline_name}'...", flush=True)
        time_us, error_msg = evaluate_submission()
        try:
            with open(SUBMISSION_FILE) as f:
                baseline_code = f.read()
            status = "keep" if time_us > 0 else "crash"
            if status == "keep":
                current_best = time_us
            _log_experiment_direct(
                kernel_code=baseline_code,
                hypothesis=f"Baseline '{baseline_name}' — initial benchmark",
                time_us=time_us,
                status=status,
                error_message=error_msg,
            )
            print(f"Baseline logged: {time_us:.1f} µs ({status})", flush=True)
            kickoff_note = (
                f"The '{baseline_name}' baseline is already benchmarked and logged as experiment #1 "
                f"({time_us:.1f} µs). Your job is to beat it. "
                if status == "keep"
                else (
                    f"The '{baseline_name}' baseline CRASHED (logged as experiment #1). "
                    "Read the crash error via get_experiment_history and fix the kernel. "
                )
            )
        except Exception as e:
            print(f"Warning: could not log baseline: {e}", flush=True)
            kickoff_note = (
                f"submission.py has been pre-loaded with '{baseline_name}'. "
                "Benchmark it first, then improve. "
            )
    else:
        kickoff_note = "submission.py is the current kernel. Improve it. "

    all_proposals: list = []
    all_advisor_history: list = []
    all_worker_history: list = []
    total_llm_calls = 0
    iteration = 0

    try:
        while iteration < args.iterations:
            if args.max_tokens is not None:
                used = _token_totals["input_tokens"] + _token_totals["output_tokens"]
                if used >= args.max_tokens:
                    print(f"\n--- Token budget exhausted: {used:,} / {args.max_tokens:,} tokens ---")
                    break

            iteration += 1
            set_agent_iteration(iteration)
            print(f"\n{'='*60}")
            print(f"  ITERATION {iteration}/{args.iterations}")
            print(f"{'='*60}\n", flush=True)

            summary = read_results_summary()

            # ── ADVISOR ──────────────────────────────────────────────────
            print("[advisor] Proposing...", flush=True)
            experiment_history = _tools.get_experiment_history()
            advisor_history = [{
                "role": "user",
                "content": (
                    f"Iteration {iteration}/{args.iterations}.\n\n"
                    f"{summary}\n\n"
                    f"## Experiment History\n\n{experiment_history}\n\n"
                    "Output your structured proposal."
                ),
            }]
            proposal, advisor_calls = run_agent_turn_retrying(
                client, advisor_history, advisor_system, [],
                advisor_model, label="advisor",
            )
            total_llm_calls += advisor_calls
            set_llm_call_count(total_llm_calls)
            all_advisor_history.append({"iteration": iteration, "turns": list(advisor_history)})
            all_proposals.append((iteration, proposal))
            print(f"\n[advisor proposal]\n{'-'*40}\n{proposal[:1000]}\n{'-'*40}\n",
                  flush=True)
            save_proposals(run_dir, all_proposals)

            # ── WORKER ───────────────────────────────────────────────────
            print("[worker] Implementing...", flush=True)
            snapshot_path = os.path.join(run_dir, f"snapshot_iter{iteration}.py")
            if os.path.exists(SUBMISSION_FILE):
                shutil.copy2(SUBMISSION_FILE, snapshot_path)

            worker_history = []
            all_worker_history.append({"iteration": iteration, "turns": worker_history})
            worker_history.append({
                "role": "user",
                "content": (
                    f"Iteration {iteration}/{args.iterations}.\n\n"
                    f"## Advisor Proposal\n\n{proposal}\n\n"
                    f"## Your Task\n\n"
                    f"{kickoff_note}"
                    "Implement the advisor's proposal: use read_file to read submission.py, "
                    "make ONE targeted change, write the complete new version with write_file, "
                    "then output your implementation report and stop. "
                    "Do NOT run evaluation — the orchestrator does that after you finish.\n\n"
                    f"{summary}"
                ),
            })
            kickoff_note = ""

            worker_text, worker_calls = run_agent_turn_retrying(
                client, worker_history, worker_system, WORKER_TOOLS,
                worker_model, label="worker",
            )
            total_llm_calls += worker_calls
            set_llm_call_count(total_llm_calls)

            # ── EVALUATE ─────────────────────────────────────────────────
            time_us, error_msg = evaluate_submission()
            try:
                with open(SUBMISSION_FILE) as f:
                    kernel_code = f.read()
            except Exception:
                kernel_code = "(could not read submission.py)"

            if time_us == 0.0:
                status = "crash"
            elif time_us < current_best:
                status = "keep"
                current_best = time_us
            else:
                status = "discard"

            hypothesis = extract_hypothesis(worker_text) or "Worker implementation"
            _log_experiment_direct(
                kernel_code=kernel_code,
                hypothesis=hypothesis,
                time_us=time_us,
                status=status,
                error_message=error_msg,
            )
            print(f"[evaluator] {time_us:.2f} µs — {status}", flush=True)

            if status == "crash":
                best_path   = os.path.join(run_dir, "best_submission.py")
                restore_src = best_path if os.path.exists(best_path) else snapshot_path
                if os.path.exists(restore_src):
                    shutil.copy2(restore_src, SUBMISSION_FILE)
                    print(
                        f"  [crash restore] submission.py restored from "
                        f"{os.path.basename(restore_src)}",
                        flush=True,
                    )

            with open(os.path.join(run_dir, "advisor_history.json"), "w") as f:
                json.dump(all_advisor_history, f, indent=2)
            with open(os.path.join(run_dir, "worker_history.json"), "w") as f:
                json.dump(all_worker_history, f, indent=2)

            if iteration % args.checkpoint_every == 0:
                print_checkpoint(iteration, args.iterations, start_time, total_llm_calls)

    except KeyboardInterrupt:
        print(f"\n--- Interrupted at iteration {iteration} ---")
    except Exception as e:
        print(f"\n--- Error at iteration {iteration}: {e} ---")
        import traceback
        traceback.print_exc()
    finally:
        save_proposals(run_dir, all_proposals)
        print_final_report(args.iterations, iteration, start_time, total_llm_calls)


if __name__ == "__main__":
    main()
