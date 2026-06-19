"""
Deployable Modal B200 evaluator for the attention-backward kernel task.

Mirrors the NVIDIA SOL-ExecBench kernel #1 (attn_bwd) benchmark exactly.

Deploy once:
    uv run modal deploy eval_modal_attn_bwd.py

Then the agent's run_eval.py calls evaluate_kernel.remote(kernel_code).
"""

import modal

# ── 16 benchmark workloads from NVIDIA SOL-ExecBench kernel #1 ───────────────
# All cases: num_attention_heads=80, num_key_value_heads=8, head_dim=128,
#            num_key_value_groups=10, attention_dropout=0.1

CASES = [
    {"batch_size": 4,  "seq_len_q": 256,  "seq_len_kv": 256,  "seed": 1001},
    {"batch_size": 8,  "seq_len_q": 373,  "seq_len_kv": 449,  "seed": 1002},
    {"batch_size": 4,  "seq_len_q": 1024, "seq_len_kv": 2048, "seed": 1003},
    {"batch_size": 64, "seq_len_q": 128,  "seq_len_kv": 128,  "seed": 1004},
    {"batch_size": 2,  "seq_len_q": 256,  "seq_len_kv": 512,  "seed": 1005},
    {"batch_size": 32, "seq_len_q": 691,  "seq_len_kv": 773,  "seed": 1006},
    {"batch_size": 8,  "seq_len_q": 128,  "seq_len_kv": 128,  "seed": 1007},
    {"batch_size": 32, "seq_len_q": 512,  "seq_len_kv": 512,  "seed": 1008},
    {"batch_size": 4,  "seq_len_q": 211,  "seq_len_kv": 293,  "seed": 1009},
    {"batch_size": 8,  "seq_len_q": 256,  "seq_len_kv": 256,  "seed": 1010},
    {"batch_size": 16, "seq_len_q": 128,  "seq_len_kv": 256,  "seed": 1011},
    {"batch_size": 1,  "seq_len_q": 1024, "seq_len_kv": 1024, "seed": 1012},
    {"batch_size": 16, "seq_len_q": 256,  "seq_len_kv": 512,  "seed": 1013},
    {"batch_size": 32, "seq_len_q": 128,  "seq_len_kv": 128,  "seed": 1014},
    {"batch_size": 1,  "seq_len_q": 512,  "seq_len_kv": 512,  "seed": 1015},
    {"batch_size": 1,  "seq_len_q": 4096, "seq_len_kv": 4096, "seed": 1016},
]

TEST_CASES = CASES
BENCHMARK_CASES = CASES

# Scoring: score = SCORE_SCALE / geomean_us (higher is better)
# Baseline geomean ≈ 756 μs → score ≈ 1.0 at baseline; SOL ≈ 82 μs → score ≈ 9.3
SCORE_SCALE = 756.0

BENCH_USE_CUDA_EVENTS = True
BENCH_REL_ERROR = 0.001      # stop when stderr/mean < 0.1%
BENCH_WALL_TIMEOUT_NS = 120e9
BENCH_NO_GRAD = True
BENCH_MAX_REPEATS = 100
BENCH_MAX_TIME_NS = 10e9

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
ATTENTION_DROPOUT = 0.1

# ── Modal image ───────────────────────────────────────────────────────────────
# B200 (Blackwell sm_100) requires CUDA 12.8+ and PyTorch 2.6+.
# Update the tag if a newer image is available.
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .pip_install("triton")
)

app = modal.App("attn-bwd-kernel-eval")


@app.function(gpu="B200", image=image, timeout=600)
def evaluate_kernel(kernel_code: str, mode: str = "leaderboard") -> str:
    import contextlib
    import copy
    import dataclasses
    import gc
    import importlib.util
    import json as _json
    import math
    import os as _os
    import tempfile
    import time
    import traceback

    import torch

    # ── Architecture constants ───────────────────────────────────────────────

    n_heads     = NUM_ATTENTION_HEADS
    n_kv_heads  = NUM_KEY_VALUE_HEADS
    n_groups    = n_heads // n_kv_heads    # 10
    head_dim    = HEAD_DIM
    dropout_p   = ATTENTION_DROPOUT

    # ── Reference implementation ─────────────────────────────────────────────

    def ref_kernel(data):
        (grad_attn_output, attn_weights, attn_weights_dropped,
         value_states, dropout_mask, attention_dropout) = data

        bs     = grad_attn_output.shape[0]
        seq_q  = grad_attn_output.shape[1]
        seq_kv = value_states.shape[2]

        # Expand value_states for GQA: [bs, n_kv, skv, d] → [bs, n_heads, skv, d]
        vs_exp = value_states[:, :, None, :, :].expand(
            bs, n_kv_heads, n_groups, seq_kv, head_dim
        ).reshape(bs, n_heads, seq_kv, head_dim)

        # 1. Transpose: [bs, sq, h, d] → [bs, h, sq, d]  (f32)
        dO = grad_attn_output.transpose(1, 2).to(torch.float32)

        # 2. dP̃ = dO @ V^T  →  [bs, h, sq, skv]
        dP_dropped = torch.matmul(dO, vs_exp.to(torch.float32).transpose(-2, -1))

        # 3. Dropout backward
        if attention_dropout > 0.0:
            dP = dP_dropped * dropout_mask / (1.0 - attention_dropout)
        else:
            dP = dP_dropped

        # 4. Softmax backward:  dS = P ⊙ (dP - sum(dP ⊙ P, dim=-1, keepdim=True))
        P = attn_weights.to(torch.float32)
        dS = P * (dP - (dP * P).sum(dim=-1, keepdim=True))
        dS = dS.to(torch.bfloat16)

        # 5. dV_exp = P̃^T @ dO  →  [bs, h, skv, d]
        dV_exp = torch.matmul(
            attn_weights_dropped.to(torch.float32).transpose(-2, -1), dO
        )

        # 6. GQA aggregation: sum over groups  →  [bs, n_kv, skv, d]
        dV = dV_exp.reshape(bs, n_kv_heads, n_groups, seq_kv, head_dim).sum(dim=2)
        dV = dV.to(torch.bfloat16)

        return dS, dV

    # ── Input generation ─────────────────────────────────────────────────────

    def generate_input(batch_size, seq_len_q, seq_len_kv, seed):
        g = torch.Generator(device="cuda")
        g.manual_seed(seed)

        grad_attn_output = torch.randn(
            batch_size, seq_len_q, n_heads, head_dim,
            dtype=torch.bfloat16, device="cuda", generator=g,
        )

        attn_scores_raw = torch.randn(
            batch_size, n_heads, seq_len_q, seq_len_kv,
            dtype=torch.float32, device="cuda", generator=g,
        )
        attn_weights = torch.softmax(attn_scores_raw, dim=-1).to(torch.bfloat16)

        dropout_mask = (
            torch.rand(batch_size, n_heads, seq_len_q, seq_len_kv,
                       device="cuda", generator=g) > dropout_p
        )

        attn_weights_dropped = (
            attn_weights.float() * dropout_mask / (1.0 - dropout_p)
        ).to(torch.bfloat16)

        value_states = torch.randn(
            batch_size, n_kv_heads, seq_len_kv, head_dim,
            dtype=torch.bfloat16, device="cuda", generator=g,
        )

        return (grad_attn_output, attn_weights, attn_weights_dropped,
                value_states, dropout_mask, float(dropout_p))

    # ── Correctness check ────────────────────────────────────────────────────

    def check_implementation(data, submission_output, rtol=1e-2, atol=1e-2):
        try:
            ref_dS, ref_dV = ref_kernel(data)
            sub_dS, sub_dV = submission_output
        except Exception as e:
            return False, f"unpack failed: {e}"

        if ref_dS.shape != sub_dS.shape:
            return False, f"grad_attn_scores shape mismatch: {ref_dS.shape} vs {sub_dS.shape}"
        if ref_dV.shape != sub_dV.shape:
            return False, f"grad_value_states shape mismatch: {ref_dV.shape} vs {sub_dV.shape}"

        ok_dS = torch.allclose(ref_dS.float(), sub_dS.float(), rtol=rtol, atol=atol)
        if not ok_dS:
            d = torch.abs(ref_dS.float() - sub_dS.float())
            return False, (f"grad_attn_scores mismatch: "
                           f"max={d.max().item():.4e} mean={d.mean().item():.4e}")

        ok_dV = torch.allclose(ref_dV.float(), sub_dV.float(), rtol=rtol, atol=atol)
        if not ok_dV:
            d = torch.abs(ref_dV.float() - sub_dV.float())
            return False, (f"grad_value_states mismatch: "
                           f"max={d.max().item():.4e} mean={d.mean().item():.4e}")

        return True, "Match"

    # ── Helpers ───────────────────────────────────────────────────────────────

    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    def _clone(data):
        if isinstance(data, tuple):
            return tuple(_clone(x) for x in data)
        if isinstance(data, list):
            return [_clone(x) for x in data]
        if isinstance(data, dict):
            return {k: _clone(v) for k, v in data.items()}
        if isinstance(data, torch.Tensor):
            return data.clone()
        if dataclasses.is_dataclass(data) and not isinstance(data, type):
            fields = {f.name: _clone(getattr(data, f.name)) for f in dataclasses.fields(data)}
            return type(data)(**fields)
        return data

    def _stats(durations):
        n = len(durations)
        avg = sum(durations) / n
        if n > 1:
            var = sum((x - avg) ** 2 for x in durations) / (n - 1)
            std = math.sqrt(var)
            err = std / math.sqrt(n)
        else:
            std, err = 0.0, 0.0
        return {"runs": n, "mean": avg, "std": std, "err": err}

    def clear_l2_cache():
        dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda")
        dummy.fill_(42)
        del dummy

    # ── Load submission ───────────────────────────────────────────────────────

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    torch_ver = torch.__version__

    tmp_dir = tempfile.mkdtemp(prefix="submission_")
    tmp_path = _os.path.join(tmp_dir, "submission.py")
    with open(tmp_path, "w") as f:
        f.write(kernel_code)

    try:
        spec = importlib.util.spec_from_file_location("submission", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        custom_kernel = mod.custom_kernel
    except Exception:
        return _json.dumps({
            "success": False,
            "error": f"Failed to load submission:\n{traceback.format_exc()}",
            "tests_passed": 0,
            "tests_total": len(TEST_CASES),
            "test_details": [],
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-b200",
            "failure_stage": "import",
        })

    # ── Correctness tests ─────────────────────────────────────────────────────

    test_details = []
    tests_passed = 0

    for tc in TEST_CASES:
        label = f"bs={tc['batch_size']} sq={tc['seq_len_q']} skv={tc['seq_len_kv']}"
        try:
            data = generate_input(**tc)
            data_copy = _clone(data)
            torch.cuda.synchronize()
            with torch.no_grad():
                output = custom_kernel(data)
            torch.cuda.synchronize()
            del data
            gc.collect()
            torch.cuda.empty_cache()

            passed, msg = check_implementation(data_copy, output)
            del data_copy, output
            gc.collect()
            torch.cuda.empty_cache()

            test_details.append({
                "batch_size": tc["batch_size"],
                "seq_len_q": tc["seq_len_q"],
                "seq_len_kv": tc["seq_len_kv"],
                "seed": tc["seed"],
                "passed": passed,
                "error": "" if passed else msg,
            })
            if passed:
                tests_passed += 1
        except Exception:
            test_details.append({
                "batch_size": tc["batch_size"],
                "seq_len_q": tc["seq_len_q"],
                "seq_len_kv": tc["seq_len_kv"],
                "seed": tc["seed"],
                "passed": False,
                "error": traceback.format_exc()[:600],
            })

    if tests_passed < len(TEST_CASES):
        return _json.dumps({
            "success": False,
            "tests_passed": tests_passed,
            "tests_total": len(TEST_CASES),
            "test_details": test_details,
            "error": "Correctness check failed — see test_details",
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-b200",
            "failure_stage": "correctness",
        })

    if mode == "test":
        return _json.dumps({
            "success": True,
            "tests_passed": tests_passed,
            "tests_total": len(TEST_CASES),
            "test_details": test_details,
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-b200",
        })

    # ── Benchmarks ────────────────────────────────────────────────────────────

    ctx = torch.no_grad() if BENCH_NO_GRAD else contextlib.nullcontext()
    benchmark_details = []
    bench_means_ns = []

    for bench_args in BENCHMARK_CASES:
        data = generate_input(**bench_args)
        data_copy = _clone(data)

        with ctx:
            output = custom_kernel(data)
            torch.cuda.synchronize()
            del data
            gc.collect()
            torch.cuda.empty_cache()
            passed, msg = check_implementation(data_copy, output)
            del data_copy, output
            gc.collect()
            torch.cuda.empty_cache()

        if not passed:
            return _json.dumps({
                "success": False,
                "tests_passed": tests_passed,
                "tests_total": len(TEST_CASES),
                "test_details": test_details,
                "error": f"Benchmark correctness: {msg}",
                "gpu_name": gpu_name,
                "torch_version": torch_ver,
                "platform": "modal-b200",
                "failure_stage": "benchmark",
            })

        data = generate_input(**bench_args)

        # Warmup
        for _ in range(3):
            custom_kernel(data)
            torch.cuda.synchronize()

        durations_ns = []
        bm_start = time.perf_counter_ns()

        with ctx:
            for t in range(BENCH_MAX_REPEATS):
                clear_l2_cache()
                torch.cuda.synchronize()

                if BENCH_USE_CUDA_EVENTS:
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    output = custom_kernel(data)
                    e.record()
                    torch.cuda.synchronize()
                    duration_ns = s.elapsed_time(e) * 1e6  # ms → ns
                else:
                    t0 = time.perf_counter_ns()
                    output = custom_kernel(data)
                    torch.cuda.synchronize()
                    duration_ns = time.perf_counter_ns() - t0

                del output
                durations_ns.append(duration_ns)

                if t > 1:
                    st = _stats(durations_ns)
                    if st["mean"] > 0 and st["err"] / st["mean"] < BENCH_REL_ERROR:
                        break
                    if st["mean"] * st["runs"] > BENCH_MAX_TIME_NS:
                        break
                    if (time.perf_counter_ns() - bm_start) > BENCH_WALL_TIMEOUT_NS:
                        break

        st = _stats(durations_ns)
        mean_us = st["mean"] / 1e3
        err_us = st["err"] / 1e3
        benchmark_details.append({
            "batch_size": bench_args["batch_size"],
            "seq_len_q": bench_args["seq_len_q"],
            "seq_len_kv": bench_args["seq_len_kv"],
            "seed": bench_args["seed"],
            "mean_us": round(mean_us, 3),
            "err_us": round(err_us, 3),
            "runs": st["runs"],
        })
        bench_means_ns.append(st["mean"])

    means_s = [ns / 1e9 for ns in bench_means_ns]
    geomean_s = math.pow(math.prod(means_s), 1.0 / len(means_s))
    geomean_us = geomean_s * 1e6
    score = SCORE_SCALE / geomean_us

    return _json.dumps({
        "success": True,
        "tests_passed": tests_passed,
        "tests_total": len(TEST_CASES),
        "test_details": test_details,
        "benchmark": {
            "geomean_us": round(geomean_us, 3),
            "score": round(score, 3),
        },
        "benchmark_details": benchmark_details,
        "gpu_name": gpu_name,
        "torch_version": torch_ver,
        "platform": "modal-b200",
    })
