"""Custom tools for the TriMul CUDA kernel optimization deep agent."""

import os
from datetime import datetime, timezone
from langchain.tools import tool

HISTORY_FILE = "experiment_history.md"
TSV_FILE = "results.tsv"
PLOT_FILE = "progress.png"

_run_directory = None
_current_agent_iteration = 0
_llm_call_count = 0


def set_run_directory(run_dir):
    global _run_directory, HISTORY_FILE, TSV_FILE, PLOT_FILE
    _run_directory = run_dir
    HISTORY_FILE = os.path.join(run_dir, "experiment_history.md")
    TSV_FILE = os.path.join(run_dir, "results.tsv")
    PLOT_FILE = os.path.join(run_dir, "progress.png")


def set_agent_iteration(n: int):
    global _current_agent_iteration
    _current_agent_iteration = n


def set_llm_call_count(n: int):
    global _llm_call_count
    _llm_call_count = n


def _ensure_history_file():
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            f.write("# Experiment History\n\n")
            f.write("Tracks every kernel attempt, its code, hypothesis, and result.\n\n")


def _ensure_tsv_file():
    if not os.path.exists(TSV_FILE):
        with open(TSV_FILE, "w") as f:
            f.write("experiment\tagent_iteration\tcommit\ttime_us\tstatus\tdescription\n")


def _get_next_iteration() -> int:
    if not os.path.exists(TSV_FILE):
        return 1
    with open(TSV_FILE) as f:
        lines = f.readlines()
    return len([l for l in lines[1:] if l.strip()]) + 1


def _parse_tsv():
    if not os.path.exists(TSV_FILE):
        return []
    with open(TSV_FILE) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return []

    rows = []
    for i, line in enumerate(lines[1:], start=1):
        parts = line.strip().split("\t")
        if not parts or len(parts) < 5:
            continue
        try:
            rows.append({
                "experiment": int(parts[0]),
                "agent_iteration": int(parts[1]) if parts[1].isdigit() else 0,
                "time_us": float(parts[3]),
                "status": parts[4],
            })
        except (ValueError, IndexError):
            continue
    return rows


def _update_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    rows = _parse_tsv()
    if not rows:
        return

    iterations = [r["agent_iteration"] for r in rows]
    times = [r["time_us"] if r["time_us"] > 0 else None for r in rows]
    statuses = [r["status"] for r in rows]

    best_so_far = float("inf")
    best_times = []
    for r in rows:
        if r["time_us"] > 0 and r["time_us"] < best_so_far:
            best_so_far = r["time_us"]
        best_times.append(best_so_far if best_so_far < float("inf") else None)

    fig, ax = plt.subplots(figsize=(14, 6))

    keep_x = [i for i, s, t in zip(iterations, statuses, times) if s == "keep" and t]
    keep_y = [-t for s, t in zip(statuses, times) if s == "keep" and t]
    discard_x = [i for i, s, t in zip(iterations, statuses, times) if s == "discard" and t]
    discard_y = [-t for s, t in zip(statuses, times) if s == "discard" and t]
    crash_x = [i for i, s in zip(iterations, statuses) if s == "crash"]

    all_valid = [-t for t in times if t and t > 0]
    y_lo = min(all_valid) * 1.15 if all_valid else -100
    y_hi = max(all_valid) * 0.85 if all_valid else 0

    if keep_x:
        ax.scatter(keep_x, keep_y, c="#22c55e", s=60, zorder=5, label="keep", edgecolors="white", linewidths=0.5)
    if discard_x:
        ax.scatter(discard_x, discard_y, c="#ef4444", s=40, zorder=4, label="discard", edgecolors="white", linewidths=0.5, alpha=0.7)
    if crash_x:
        ax.scatter(crash_x, [y_lo] * len(crash_x), c="#fbbf24", s=25, zorder=3, label=f"crash ({len(crash_x)})", marker="x", alpha=0.6)

    valid_best = [(i, -bt) for i, bt in zip(iterations, best_times) if bt is not None]
    if valid_best:
        bx, by = zip(*valid_best)
        ax.step(bx, by, where="post", color="#3b82f6", linewidth=2, label="best time", zorder=6)

    ax.set_ylim(y_lo * 1.05, y_hi)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.set_xlabel("Iteration #", fontsize=12)
    ax.set_ylabel("Negative Latency (-μs)", fontsize=12)
    ax.set_title("GPU MODE trimul — Autoresearch Progress", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    if all_valid and best_so_far < float("inf"):
        ax.annotate(
            f"Best: {best_so_far:.2f} μs",
            xy=(0.02, 0.98), xycoords="axes fraction",
            fontsize=11, fontweight="bold", color="#3b82f6",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#3b82f6", alpha=0.9),
        )

    ax.annotate(
        f"LLM calls: {_llm_call_count}",
        xy=(0.98, 0.02), xycoords="axes fraction",
        ha="right", va="bottom",
        fontsize=10, color="#6b7280",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d1d5db", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150)
    plt.close(fig)


def _update_iteration_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    rows = _parse_tsv()
    if not rows or all(r["agent_iteration"] == 0 for r in rows):
        return

    iter_best: dict[int, float] = {}
    running_best = float("inf")
    for r in rows:
        if r["time_us"] > 0 and r["time_us"] < running_best:
            running_best = r["time_us"]
        it = r["agent_iteration"]
        iter_best[it] = running_best

    if not iter_best:
        return

    iters = sorted(iter_best)
    bests = [-iter_best[i] for i in iters]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.step(iters, bests, where="post", color="#3b82f6", linewidth=2)
    ax.scatter(iters, bests, c="#3b82f6", s=60, zorder=5, edgecolors="white", linewidths=0.5)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.set_xlabel("Iteration #", fontsize=12)
    ax.set_ylabel("Negative Latency (-μs)", fontsize=12)
    ax.set_title("GPU MODE trimul — Best per Agent Iteration", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    best_overall = min(iter_best.values())
    ax.annotate(
        f"Best: {best_overall:.2f} μs",
        xy=(0.02, 0.98), xycoords="axes fraction",
        fontsize=11, fontweight="bold", color="#3b82f6",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#3b82f6", alpha=0.9),
    )

    fig.tight_layout()
    if _run_directory:
        fig.savefig(os.path.join(_run_directory, "iterations.png"), dpi=150)
    plt.close(fig)


def _log_experiment_direct(
    kernel_code: str,
    hypothesis: str,
    time_us: float,
    status: str,
    error_message: str = "",
    commit: str = "HEAD",
) -> str:
    _ensure_history_file()
    _ensure_tsv_file()

    iteration = _get_next_iteration()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    emoji = {"keep": "✅", "discard": "❌", "crash": "💥"}.get(status, "❓")

    with open(HISTORY_FILE, "a") as f:
        f.write(f"---\n\n")
        f.write(f"## Experiment #{iteration} — {timestamp} {emoji} {status.upper()}\n\n")
        f.write(f"**Hypothesis:** {hypothesis}\n\n")
        if status == "crash":
            f.write(f"**Result:** CRASH\n\n")
            if error_message:
                f.write(f"**Error:**\n```\n{error_message[:2000]}\n```\n\n")
        else:
            f.write(f"**Result:** {time_us:.2f} μs\n\n")
        f.write(f"**Kernel code:**\n```python\n{kernel_code}\n```\n\n")

    desc = hypothesis[:100]
    with open(TSV_FILE, "a") as f:
        f.write(f"{iteration}\t{_current_agent_iteration}\t{commit}\t{time_us:.2f}\t{status}\t{desc}\n")

    try:
        _update_plot()
        _update_iteration_plot()
    except Exception:
        pass

    if status == "keep" and _run_directory and kernel_code.strip():
        with open(os.path.join(_run_directory, "best_submission.py"), "w") as f:
            f.write(kernel_code)

    if status == "crash":
        return f"Logged #{iteration} CRASH: {hypothesis}"
    return f"Logged #{iteration} {status}: {time_us:.2f} μs — {hypothesis}"


@tool
def log_experiment(
    hypothesis: str,
    time_us: float,
    status: str,
    error_message: str = "",
    commit: str = "HEAD",
) -> str:
    """Log a kernel experiment to experiment_history.md and results.tsv.

    Call this after every submission attempt. Reads submission.py automatically.

    Args:
        hypothesis: What this experiment was trying and why.
        time_us: Geometric mean latency in microseconds. Use 0.0 for crashes.
        status: "keep" (new best), "discard" (worse than best), or "crash".
        error_message: Error output if status is "crash", else empty string.
        commit: Short git commit hash (optional, defaults to "HEAD").

    Returns:
        Confirmation message with experiment number.
    """
    submission_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission.py")
    try:
        with open(submission_path) as f:
            kernel_code = f.read()
    except Exception as e:
        kernel_code = f"(could not read submission.py: {e})"
    return _log_experiment_direct(kernel_code, hypothesis, time_us, status, error_message, commit)


@tool
def get_experiment_history() -> str:
    """Read the full experiment history markdown.

    Returns every prior kernel attempt, its code, hypothesis, and result.
    Call this before proposing a new approach to avoid repeating failures.
    """
    if not os.path.exists(HISTORY_FILE):
        return "No experiment history yet. This will be the first run."
    with open(HISTORY_FILE) as f:
        content = f.read()
    if len(content) > 50000:
        return content[-50000:]
    return content
