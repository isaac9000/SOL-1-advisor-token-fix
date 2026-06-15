"""
Compare openevolve vs advisor (no refresh) vs advisor-refresh vs evox runs.
Marks epoch refresh boundary with a vertical line.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv

# ── Advisor (no refresh) data ─────────────────────────────────────────────────
ADV_TSV = "/workspace/trimul-advisor/trimul/runs/20260609_012417_trimul_starting_point/results.tsv"
adv_iters, adv_times, adv_kinds = [], [], []
with open(ADV_TSV) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        adv_iters.append(int(row["agent_iteration"]))
        adv_times.append(float(row["time_us"]))
        adv_kinds.append(row["status"])

# ── Advisor-refresh data (epoch 1 + epoch 2, stitched by agent_iteration) ─────
# Epoch 1: agent_iterations 0–15, Epoch 2: agent_iterations 15–25
# Epoch refresh happens after agent_iteration 15.
REFRESH_ITER = 15

refresh_iters, refresh_times, refresh_kinds = [], [], []

epoch1_rows = [
    (0,  11010.26, "keep"),
    (1,  10349.94, "keep"),
    (2,  7070.93,  "keep"),
    (3,  0.00,     "crash"),
    (4,  0.00,     "crash"),
    (5,  6095.95,  "keep"),
    (6,  13490.71, "discard"),
    (7,  7693.57,  "discard"),
    (8,  7473.20,  "discard"),
    (9,  35733.22, "discard"),
    (10, 6234.40,  "discard"),
    (11, 22188.10, "discard"),
    (12, 0.00,     "crash"),
    (13, 6814.14,  "discard"),
    (14, 7261.37,  "discard"),
    (15, 0.00,     "crash"),
]

epoch2_rows = [
    (15, 6555.38,  "keep"),
    (16, 12154.78, "discard"),
    (17, 6250.77,  "keep"),
    (18, 6493.02,  "discard"),
    (19, 6667.73,  "discard"),
    (20, 36043.76, "discard"),
    (21, 6187.29,  "keep"),
    (22, 7495.09,  "discard"),
    (23, 6296.05,  "discard"),
    (24, 5754.16,  "keep"),
    (25, 13144.36, "discard"),
]

for it, t, k in epoch1_rows + epoch2_rows:
    refresh_iters.append(it)
    refresh_times.append(t)
    refresh_kinds.append(k)

# ── EvoX data ────────────────────────────────────────────────────────────────
# From trimul-evox/plot_run.py — skydiscover run1, 25 iterations
evox_raw = [
    (1,  9344.256,  "keep"),
    (3,  9356.484,  "discard"),
    (4,  9291.110,  "keep"),
    (5,  9329.417,  "discard"),
    (6,  9266.186,  "keep"),
    (7,  9302.989,  "discard"),
    (8,  9281.608,  "discard"),
    (9,  6572.176,  "keep"),
    (10, 0.0,       "fail"),
    (13, 6562.477,  "keep"),
    (14, 0.0,       "fail"),
    (17, 6352.167,  "keep"),
    (18, 6480.952,  "discard"),
    (21, 6428.371,  "discard"),
    (23, 6377.658,  "discard"),
]
evox_iters = [r[0] for r in evox_raw]
evox_times = [r[1] for r in evox_raw]
evox_kinds = ["crash" if r[2] == "fail" else r[2] for r in evox_raw]

# ── OpenEvolve data ───────────────────────────────────────────────────────────
# Extracted from trimul_kernel/openevolve_runs/run1/logs/*.log
# Iterations absent from this list had no valid geomean score (correctness failures)
oe_raw = [
    (0,  11042.03),
    (1,  9360.47),
    (2,  9492.79),
    (3,  9348.14),
    (4,  None),
    (5,  9348.82),
    (6,  7364.38),
    (8,  7402.43),
    (11, 8130.58),
    (12, 7381.60),
    (13, 8120.18),
    (15, 8134.02),
    (19, 8091.53),
    (20, None),
    (21, 8082.31),
    (22, 7420.66),
    (25, 8074.29),
]
oe_iters, oe_times, oe_kinds = [], [], []
best_so_far = float("inf")
for it, t in oe_raw:
    oe_iters.append(it)
    oe_times.append(t if t is not None else 0.0)
    if t is None:
        oe_kinds.append("crash")
    elif t < best_so_far:
        best_so_far = t
        oe_kinds.append("keep")
    else:
        oe_kinds.append("discard")

# ── Best-over-time step lines ─────────────────────────────────────────────────
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

adv_bx,  adv_by  = best_step(adv_iters,     adv_times,     adv_kinds)
ref_bx,  ref_by  = best_step(refresh_iters, refresh_times, refresh_kinds)
oe_bx,   oe_by   = best_step(oe_iters,      oe_times,      oe_kinds)
evox_bx, evox_by = best_step(evox_iters,    evox_times,    evox_kinds)

adv_best  = min(t for t, k in zip(adv_times, adv_kinds) if k == "keep")
ref_best  = min(t for t, k in zip(refresh_times, refresh_kinds) if k == "keep" and t > 0)
oe_best   = min(oe_by) if oe_by else float("inf")
evox_best = min(evox_by) if evox_by else float("inf")

# ── Y-axis (negative latency, clip outliers) ──────────────────────────────────
CLIP_US = 20000.0
all_valid = [t for t in adv_times + refresh_times + oe_times + evox_times if 0 < t <= CLIP_US]
y_hi = -(min(all_valid) * 0.82)
y_lo = -(CLIP_US * 1.08)

def ny(t):
    return max(-t, y_lo) if t > 0 else y_lo

LLM_CALLS = 160  # advisor-refresh approximate total

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
fig.subplots_adjust(top=0.75)

# OpenEvolve — blue
oe_kx = [it for it, k in zip(oe_iters, oe_kinds) if k == "keep"]
oe_ky = [ny(oe_times[i]) for i, k in enumerate(oe_kinds) if k == "keep"]
oe_dx = [it for it, k in zip(oe_iters, oe_kinds) if k == "discard"]
oe_dy = [ny(oe_times[i]) for i, k in enumerate(oe_kinds) if k == "discard"]
oe_cx = [it for it, k in zip(oe_iters, oe_kinds) if k == "crash"]
if oe_kx:
    ax.scatter(oe_kx, oe_ky, c="#3b82f6", s=70, zorder=5, edgecolors="white", linewidths=0.5, label="openevolve keep")
if oe_dx:
    ax.scatter(oe_dx, oe_dy, c="#93c5fd", s=40, zorder=4, edgecolors="white", linewidths=0.3, alpha=0.8, label="openevolve discard")
if oe_bx:
    ax.step(oe_bx, [-t for t in oe_by], where="post", color="#3b82f6", linewidth=2, label="openevolve best", zorder=6)

# Advisor (no refresh) — green
adv_kx = [it for it, k in zip(adv_iters, adv_kinds) if k == "keep"]
adv_ky = [ny(adv_times[i]) for i, k in enumerate(adv_kinds) if k == "keep"]
adv_dx = [it for it, k in zip(adv_iters, adv_kinds) if k == "discard"]
adv_dy = [ny(adv_times[i]) for i, k in enumerate(adv_kinds) if k == "discard"]
adv_cx = [it for it, k in zip(adv_iters, adv_kinds) if k == "crash"]
if adv_kx:
    ax.scatter(adv_kx, adv_ky, c="#22c55e", s=70, zorder=5, edgecolors="white", linewidths=0.5, label="advisor keep")
if adv_dx:
    ax.scatter(adv_dx, adv_dy, c="#ef4444", s=40, zorder=4, edgecolors="white", linewidths=0.3, alpha=0.7, label="advisor discard")
if adv_bx:
    ax.step(adv_bx, [-t for t in adv_by], where="post", color="#22c55e", linewidth=2, label="advisor best", zorder=6)

# Advisor-refresh — purple
ref_kx = [it for i, (it, k) in enumerate(zip(refresh_iters, refresh_kinds))
          if k == "keep" and refresh_times[i] > 0]
ref_ky = [ny(refresh_times[i]) for i, k in enumerate(refresh_kinds)
          if k == "keep" and refresh_times[i] > 0]
ref_dx = [it for it, k in zip(refresh_iters, refresh_kinds) if k == "discard"]
ref_dy = [ny(refresh_times[i]) for i, k in enumerate(refresh_kinds) if k == "discard"]
ref_cx = [it for it, k in zip(refresh_iters, refresh_kinds) if k == "crash"]
if ref_kx:
    ax.scatter(ref_kx, ref_ky, c="#a855f7", s=70, zorder=5, edgecolors="white", linewidths=0.5, label="advisor-refresh keep")
if ref_dx:
    ax.scatter(ref_dx, ref_dy, c="#d8b4fe", s=40, zorder=4, edgecolors="white", linewidths=0.3, alpha=0.7, label="advisor-refresh discard")
if ref_bx:
    ax.step(ref_bx, [-t for t in ref_by], where="post", color="#a855f7", linewidth=2, label="advisor-refresh best", zorder=6)

# EvoX — orange
evox_kx = [it for it, k in zip(evox_iters, evox_kinds) if k == "keep"]
evox_ky = [ny(evox_times[i]) for i, k in enumerate(evox_kinds) if k == "keep"]
evox_dx = [it for it, k in zip(evox_iters, evox_kinds) if k == "discard"]
evox_dy = [ny(evox_times[i]) for i, k in enumerate(evox_kinds) if k == "discard"]
evox_cx = [it for it, k in zip(evox_iters, evox_kinds) if k == "crash"]
if evox_kx:
    ax.scatter(evox_kx, evox_ky, c="#f97316", s=70, zorder=5, edgecolors="white", linewidths=0.5, label="evox keep")
if evox_dx:
    ax.scatter(evox_dx, evox_dy, c="#fed7aa", s=40, zorder=4, edgecolors="white", linewidths=0.3, alpha=0.8, label="evox discard")
if evox_bx:
    ax.step(evox_bx, [-t for t in evox_by], where="post", color="#f97316", linewidth=2, label="evox best", zorder=6)

# Crashes (all series)
all_cx = oe_cx + adv_cx + ref_cx + evox_cx
if all_cx:
    ax.scatter(all_cx, [y_lo] * len(all_cx), c="#fbbf24", s=40, zorder=3,
               marker="x", linewidths=1.5, label=f"crash ({len(all_cx)})", alpha=0.8)

# Epoch refresh marker
ax.axvline(x=REFRESH_ITER, color="#a855f7", linewidth=1.5, linestyle="--", alpha=0.7, zorder=2)
ax.annotate("← epoch refresh", xy=(REFRESH_ITER + 0.2, y_hi * 0.97),
            fontsize=9, color="#7c3aed", va="top")

ax.set_ylim(y_lo * 1.05, y_hi)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
ax.set_xlabel("Iteration #", fontsize=12)
ax.set_ylabel("Negative Latency (-μs)", fontsize=12)
ax.grid(True, alpha=0.3)

# Legend above the plot
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=5,
          framealpha=0.9, fontsize=10, borderaxespad=0)

# Best-time records above the plot (figure-level text)
fig.text(0.5, 0.92,
         f"EvoX best: {evox_best:.2f} μs    |    "
         f"OpenEvolve best: {oe_best:.2f} μs    |    "
         f"Advisor best: {adv_best:.2f} μs    |    "
         f"Advisor-refresh best: {ref_best:.2f} μs",
         ha="center", va="top", fontsize=11, fontweight="bold", color="#1e3a5f",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#a855f7", alpha=0.9))

# Title
fig.text(0.5, 0.995, "evox vs openevolve vs advisor vs advisor-refresh — trimul",
         ha="center", va="top", fontsize=14, fontweight="bold")

# LLM call counter — bottom right (advisor-refresh only)
ax.annotate(
    f"advisor-refresh LLM calls: ~{LLM_CALLS}",
    xy=(0.99, 0.02), xycoords="axes fraction",
    ha="right", va="bottom", fontsize=10, color="#6b7280",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d1d5db", alpha=0.9),
)

# Outlier note — bottom left
ax.annotate(
    f"(outliers > {CLIP_US:.0f} μs shown at floor)",
    xy=(0.01, 0.02), xycoords="axes fraction",
    ha="left", va="bottom", fontsize=9, color="#6b7280",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d1d5db", alpha=0.8),
)

out = "/workspace/trimul-advisor/comparison.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved {out}")
