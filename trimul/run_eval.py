#!/usr/bin/env python3
"""
CLI wrapper that submits a TriMul kernel to the deployed Modal H100 evaluator
and writes results.json in markdown format the agent can parse.

Deploy the evaluator once before running:
    uv run modal deploy eval_modal_trimul.py

Usage:
    python run_eval.py submission.py -o results.json
    python run_eval.py submission.py -o results.json --mode test   # correctness only
"""

import argparse
import json
import sys
import threading

import modal

TEST_CASES = [
    {"seqlen": 32,   "bs": 1, "dim": 128, "hiddendim": 128, "seed": 9371,   "nomask": True,  "distribution": "normal"},
    {"seqlen": 32,   "bs": 1, "dim": 128, "hiddendim": 128, "seed": 1092,   "nomask": False, "distribution": "normal"},
    {"seqlen": 64,   "bs": 2, "dim": 256, "hiddendim": 128, "seed": 2291,   "nomask": True,  "distribution": "normal"},
    {"seqlen": 64,   "bs": 2, "dim": 256, "hiddendim": 128, "seed": 210284, "nomask": False, "distribution": "normal"},
    {"seqlen": 128,  "bs": 1, "dim": 768, "hiddendim": 128, "seed": 81934,  "nomask": True,  "distribution": "normal"},
    {"seqlen": 256,  "bs": 1, "dim": 128, "hiddendim": 128, "seed": 1932,   "nomask": True,  "distribution": "normal"},
    {"seqlen": 256,  "bs": 1, "dim": 128, "hiddendim": 128, "seed": 10432,  "nomask": False, "distribution": "normal"},
    {"seqlen": 768,  "bs": 2, "dim": 128, "hiddendim": 128, "seed": 731,    "nomask": True,  "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 53121,  "nomask": False, "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 31,     "nomask": True,  "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 4921,   "nomask": False, "distribution": "normal"},
    {"seqlen": 32,   "bs": 1, "dim": 128, "hiddendim": 128, "seed": 937321, "nomask": True,  "distribution": "cauchy"},
    {"seqlen": 64,   "bs": 2, "dim": 256, "hiddendim": 128, "seed": 2291,   "nomask": True,  "distribution": "cauchy"},
    {"seqlen": 128,  "bs": 1, "dim": 768, "hiddendim": 128, "seed": 8134,   "nomask": True,  "distribution": "cauchy"},
    {"seqlen": 256,  "bs": 1, "dim": 128, "hiddendim": 128, "seed": 932,    "nomask": True,  "distribution": "cauchy"},
    {"seqlen": 768,  "bs": 2, "dim": 128, "hiddendim": 128, "seed": 31,     "nomask": True,  "distribution": "cauchy"},
    {"seqlen": 1024, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 5321,   "nomask": False, "distribution": "cauchy"},
    {"seqlen": 1024, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 491,    "nomask": False, "distribution": "cauchy"},
]

BENCHMARK_CASES = [
    {"seqlen": 256,  "bs": 2, "dim": 128, "hiddendim": 128, "seed": 9371,  "nomask": True,  "distribution": "normal"},
    {"seqlen": 768,  "bs": 1, "dim": 128, "hiddendim": 128, "seed": 381,   "nomask": True,  "distribution": "cauchy"},
    {"seqlen": 256,  "bs": 2, "dim": 384, "hiddendim": 128, "seed": 2301,  "nomask": False, "distribution": "normal"},
    {"seqlen": 512,  "bs": 1, "dim": 128, "hiddendim": 128, "seed": 12819, "nomask": True,  "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 381,   "nomask": True,  "distribution": "cauchy"},
    {"seqlen": 768,  "bs": 1, "dim": 384, "hiddendim": 128, "seed": 481,   "nomask": False, "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 23291, "nomask": True,  "distribution": "normal"},
]


def _case_label(tc: dict) -> str:
    mask_str = "nomask" if tc["nomask"] else "mask"
    return f"seqlen={tc['seqlen']} bs={tc['bs']} dim={tc['dim']} {mask_str} {tc['distribution']}"


def format_results_markdown(res: dict, mode: str = "leaderboard") -> str:
    gpu = res.get("gpu_name", "NVIDIA H100")
    torch_ver = res.get("torch_version", "unknown")
    plat = res.get("platform", "unknown")

    if res["success"]:
        status_line = "**H100 on Modal ✅ success**"
    else:
        status_line = "**H100 on Modal ❌ failure**"

    lines = [status_line]

    if res["success"]:
        lines.append("> ✅ Testing successful")
        if mode == "leaderboard":
            lines.append("> ✅ Benchmarking successful")
    elif res.get("tests_passed", 0) == res.get("tests_total", 1):
        lines.append("> ✅ Testing successful")
        lines.append("> ❌ Benchmarking failed")
    else:
        lines.append("> ❌ Testing failed")

    lines += [
        "",
        "Running on:",
        f"* GPU: `{gpu}`",
        f"* Runtime: `CUDA`",
        f"* Platform: `{plat}`",
        f"* Torch: `{torch_ver}`",
        "",
    ]

    passed = res.get("tests_passed", 0)
    total = res.get("tests_total", 0)
    lines.append(f"## {'✅' if passed == total else '❌'} Passed {passed}/{total} tests:")
    lines.append("```")
    for td in res.get("test_details", []):
        icon = "✅" if td["passed"] else "❌"
        label = _case_label(td)
        lines.append(f"{icon} {label}")
        if td.get("error"):
            lines.append(f"   ERROR: {td['error']}")
    lines.append("```")

    if res.get("error") and not res["success"]:
        lines += ["", "## Error:", "```", res["error"], "```"]

    bm = res.get("benchmark")
    if bm and mode == "leaderboard":
        geomean = bm["geomean_us"]
        score = bm.get("score", "")
        lines += ["", "## Benchmarks:", "```", f"Geometric mean: ⏱ {geomean} µs", ""]
        if score:
            lines.append(f"Score: {score}")
            lines.append("")
        for bd in res.get("benchmark_details", []):
            label = _case_label(bd)
            lines.append(
                f"  {label}: ⏱ {bd['mean_us']} ± {bd['err_us']} µs"
                f"  (runs={bd.get('runs', '?')})"
            )
        lines.append("```")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a TriMul kernel on Modal H100")
    parser.add_argument("submission", help="Path to submission.py")
    parser.add_argument("-o", "--output", default="results.json")
    parser.add_argument(
        "--mode",
        choices=["test", "leaderboard"],
        default="leaderboard",
        help="'test' for correctness only, 'leaderboard' for correctness + benchmark",
    )
    args = parser.parse_args()

    try:
        with open(args.submission) as f:
            kernel_code = f.read()
    except FileNotFoundError:
        print(f"Error: {args.submission} not found")
        sys.exit(1)

    print(f"Submitting {args.submission} to Modal H100 ({args.mode} mode)...")

    evaluate_kernel = modal.Function.from_name("trimul-kernel-eval", "evaluate_kernel")

    MODAL_TIMEOUT = 600
    result_holder = [None]
    error_holder = [None]

    def _call():
        try:
            result_holder[0] = evaluate_kernel.remote(kernel_code, mode=args.mode)
        except Exception as e:
            error_holder[0] = e

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=MODAL_TIMEOUT)

    if t.is_alive():
        print(f"Error: Modal call timed out after {MODAL_TIMEOUT}s", file=sys.stderr)
        sys.exit(2)
    if error_holder[0] is not None:
        print(f"Error: Modal call failed: {error_holder[0]}", file=sys.stderr)
        sys.exit(1)

    raw = result_holder[0]
    res = json.loads(raw)
    md = format_results_markdown(res, mode=args.mode)

    with open(args.output, "w") as f:
        json.dump(md, f)

    print(md)
    sys.exit(0 if res["success"] else 1)


if __name__ == "__main__":
    main()
