"""
Compare 1-advisor/2-worker vs 1-advisor/1-worker — SOLExecBench Problem 1 (attn_bwd).

Data sources:
  1 worker:  this repo, runs/20260630_210612_attn_bwd_starting_point_2_mil_tokens_workers_no_experiment_history/results.tsv
  2 workers: isaac9000/SOL-1-advisor-2-worker, attn_bwd/runs/20260630_205102_attn_bwd_starting_point_2worker_83min/results.tsv

Wall-clock timestamps are not available for the 2-worker run (no per-experiment
timestamps in its results.tsv/proposals.md/SKILLS.md — only an opaque global
trial counter "t=" that isn't real time), so the x-axis uses iteration #
(agent_iteration) instead.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── 1 advisor / 1 worker (this repo, run 20260630_210612) ───────────────────
# Token usage: 1,750,781 input + 264,846 output = 2,015,627 total, 149 LLM calls
ONE_TOKENS = {"input": 1_750_781, "output": 264_846, "api_calls": 149}

ONE_RAW = [
    (0, 3430.17, "keep"),
    (1, 2064.97, "keep"),
    (2, 2007.59, "keep"),
    (3, 615.79, "keep"),
    (4, 487.13, "keep"),
    (5, 1833.86, "discard"),
    (6, 0.0, "crash"),
    (7, 601.48, "discard"),
    (8, 464.59, "keep"),
    (9, 0.0, "crash"),
    (10, 518.09, "discard"),
    (11, 413.4, "keep"),
    (12, 0.0, "crash"),
    (13, 713.14, "discard"),
    (14, 680.62, "discard"),
    (15, 764.63, "discard"),
    (16, 0.0, "crash"),
    (17, 4880.7, "discard"),
    (18, 3453.73, "discard"),
    (19, 0.0, "crash"),
    (20, 408.02, "keep"),
    (21, 421.82, "discard"),
    (22, 436.34, "discard"),
    (23, 770.47, "discard"),
    (24, 0.0, "crash"),
    (25, 421.75, "discard"),
    (26, 498.79, "discard"),
    (27, 5730.47, "discard"),
    (28, 690.75, "discard"),
    (29, 852.35, "discard"),
    (30, 744.5, "discard"),
    (31, 0.0, "crash"),
    (32, 0.0, "crash"),
    (33, 436.07, "discard"),
    (34, 0.0, "crash"),
    (35, 910.23, "discard"),
    (36, 882.62, "discard"),
    (37, 884.03, "discard"),
]
one_iters = [r[0] for r in ONE_RAW]
one_times = [r[1] for r in ONE_RAW]
one_kinds = [r[2] for r in ONE_RAW]

# ── 1 advisor / 2 workers (SOL-1-advisor-2-worker, run 20260630_205102) ─────
# Token usage: 991,859 input + 182,757 output = 1,174,616 total, 154 LLM calls
TWO_TOKENS = {"input": 991_859, "output": 182_757, "api_calls": 154}

TWO_RAW = [
    (0, 3454.83, "keep"),
    (0, 3454.83, "keep"),
    (1, 2072.2, "keep"),
    (1, 2001.95, "keep"),
    (2, 882.05, "keep"),
    (2, 892.84, "keep"),
    (3, 899.74, "discard"),
    (3, 0.0, "crash"),
    (4, 907.9, "discard"),
    (4, 2710.11, "keep"),
    (5, 928.89, "discard"),
    (5, 1039.0, "keep"),
    (6, 0.0, "crash"),
    (6, 853.93, "keep"),
    (7, 4640.14, "keep"),
    (7, 929.28, "discard"),
    (8, 1875.41, "keep"),
    (8, 920.17, "keep"),
    (9, 966.97, "keep"),
    (9, 896.97, "discard"),
    (10, 1024.93, "discard"),
    (10, 1517.5, "discard"),
    (11, 1025.91, "discard"),
    (11, 1081.28, "discard"),
    (12, 1044.3, "discard"),
    (12, 1084.89, "discard"),
    (13, 1326.46, "keep"),
    (13, 0.0, "crash"),
    (14, 4229.99, "discard"),
    (14, 857.7, "discard"),
    (15, 0.0, "crash"),
    (15, 899.07, "discard"),
    (16, 1769.64, "discard"),
    (16, 866.33, "discard"),
    (17, 1604.5, "discard"),
    (17, 862.97, "discard"),
    (18, 3680.36, "discard"),
    (18, 1065.79, "discard"),
    (19, 0.0, "crash"),
    (19, 928.01, "discard"),
    (20, 924.86, "keep"),
    (20, 0.0, "crash"),
    (21, 0.0, "crash"),
    (21, 981.0, "discard"),
    (22, 0.0, "crash"),
]
two_iters = [r[0] for r in TWO_RAW]
two_times = [r[1] for r in TWO_RAW]
two_kinds = [r[2] for r in TWO_RAW]


def best_step(iters, times, kinds):
    bx, by = [], []
    best = float("inf")
    for it, t, k in sorted(zip(iters, times, kinds)):
        if k == "keep" and t > 0:
            best = t
        if best < float("inf"):
            bx.append(it)
            by.append(best)
    return bx, by


one_bx, one_by = best_step(one_iters, one_times, one_kinds)
two_bx, two_by = best_step(two_iters, two_times, two_kinds)

one_best = min(t for t, k in zip(one_times, one_kinds) if k == "keep" and t > 0)
two_best = min(t for t, k in zip(two_times, two_kinds) if k == "keep" and t > 0)

# ── Y-axis (negative latency, clip outliers) ─────────────────────────────────
CLIP_US = 5000.0
all_valid = [t for t in one_times + two_times if 0 < t <= CLIP_US]
y_hi = -(min(all_valid) * 0.82)
y_lo = -(CLIP_US * 1.08)


def ny(t):
    return max(-t, y_lo) if t > 0 else y_lo


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
fig.subplots_adjust(top=0.75)

# ── 1 advisor / 1 worker — green ──────────────────────────────────────────────
one_kx = [it for it, k in zip(one_iters, one_kinds) if k == "keep"]
one_ky = [ny(t) for t, k in zip(one_times, one_kinds) if k == "keep"]
one_dx = [it for it, k in zip(one_iters, one_kinds) if k == "discard"]
one_dy = [ny(t) for t, k in zip(one_times, one_kinds) if k == "discard"]
one_cx = [it for it, k in zip(one_iters, one_kinds) if k == "crash"]

if one_kx:
    ax.scatter(one_kx, one_ky, c="#22c55e", s=70, zorder=5,
               edgecolors="white", linewidths=0.5, label="1 worker keep")
if one_dx:
    ax.scatter(one_dx, one_dy, c="#86efac", s=40, zorder=4,
               edgecolors="white", linewidths=0.3, alpha=0.8, label="1 worker discard")
if one_bx:
    ax.step(one_bx, [-t for t in one_by], where="post", color="#22c55e",
            linewidth=2, label="1 worker best", zorder=6)

# ── 1 advisor / 2 workers — blue ──────────────────────────────────────────────
two_kx = [it for it, k in zip(two_iters, two_kinds) if k == "keep"]
two_ky = [ny(t) for t, k in zip(two_times, two_kinds) if k == "keep"]
two_dx = [it for it, k in zip(two_iters, two_kinds) if k == "discard"]
two_dy = [ny(t) for t, k in zip(two_times, two_kinds) if k == "discard"]
two_cx = [it for it, k in zip(two_iters, two_kinds) if k == "crash"]

if two_kx:
    ax.scatter(two_kx, two_ky, c="#3b82f6", s=70, zorder=5,
               edgecolors="white", linewidths=0.5, label="2 workers keep")
if two_dx:
    ax.scatter(two_dx, two_dy, c="#93c5fd", s=40, zorder=4,
               edgecolors="white", linewidths=0.3, alpha=0.8, label="2 workers discard")
if two_bx:
    ax.step(two_bx, [-t for t in two_by], where="post", color="#3b82f6",
            linewidth=2, label="2 workers best", zorder=6)

# ── Crashes ───────────────────────────────────────────────────────────────────
all_cx = one_cx + two_cx
if all_cx:
    ax.scatter(all_cx, [y_lo] * len(all_cx), c="#fbbf24", s=40, zorder=3,
               marker="x", linewidths=1.5,
               label=f"crash ({len(all_cx)})", alpha=0.8)

ax.set_ylim(y_lo * 1.05, y_hi)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
ax.set_xlabel("Advisor Iteration #", fontsize=12)
ax.set_ylabel("Negative Latency (−μs)", fontsize=12)
ax.grid(True, alpha=0.3)

# ── Legend above the plot ─────────────────────────────────────────────────────
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4,
          framealpha=0.9, fontsize=10, borderaxespad=0)

# ── Best-time band ────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.92,
    f"1 worker best: {one_best:.2f} μs    |    "
    f"2 workers best: {two_best:.2f} μs",
    ha="center", va="top", fontsize=11, fontweight="bold", color="#1e3a5f",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor="#6b7280", alpha=0.9),
)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.995,
    "1 advisor 2 workers vs 1 advisor 1 worker - SOLExecBench Problem 1 (all llms)",
    ha="center", va="top", fontsize=14, fontweight="bold",
)

# ── Token usage annotations ───────────────────────────────────────────────────
token_lines = [
    ("1 worker", "#22c55e", ONE_TOKENS, 0.26),
    ("2 workers", "#3b82f6", TWO_TOKENS, 0.51),
]
for label, color, tok, xfrac in token_lines:
    total = tok["input"] + tok["output"]
    total_disp = f"{total/1_000_000:.1f}M" if total >= 1_000_000 else f"{total/1_000:.0f}K"
    text = (
        f"{label}\n"
        f"{tok['api_calls']} LLM calls\n"
        f"~{total_disp} tokens"
    )
    ax.annotate(
        text,
        xy=(xfrac, 0.02), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=8.5, color=color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=color, alpha=0.85),
    )

# ── Outlier note ──────────────────────────────────────────────────────────────
ax.annotate(
    f"(outliers > {CLIP_US:.0f} μs shown at floor)",
    xy=(0.5, 0.10), xycoords="axes fraction",
    ha="center", va="bottom", fontsize=9, color="#6b7280",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor="#d1d5db", alpha=0.8),
)

# ── Baseline & SOL reference lines ────────────────────────────────────────────
ax.axhline(-756, color="#9ca3af", linewidth=1.0, linestyle="--", alpha=0.5,
           label="baseline ≈756 μs")
ax.axhline(-82, color="#10b981", linewidth=1.0, linestyle="--", alpha=0.5,
           label="SOL ≈82 μs")

out = "/workspace/SOL-1-advisor-token-fix/attn_bwd/comparison_2worker_vs_1worker.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
