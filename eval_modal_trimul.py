"""
Deployable Modal H100 evaluator for the TriMul kernel task.
 
Evaluation logic mirrors skydiscover benchmarks/gpu_mode/trimul exactly.
 
Deploy once:
    uv run modal deploy eval_modal_trimul.py
 
Then the agent's run_eval.py calls evaluate_kernel.remote(kernel_code).
"""
 
import modal
 
# ── Reference implementation (mirrors skydiscover trimul/reference.py) ────────
 
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
 
SCORE_SCALE = 3000.0
BENCH_USE_CUDA_EVENTS = True
BENCH_REL_ERROR = 0.001
BENCH_WALL_TIMEOUT_NS = 120e9
BENCH_NO_GRAD = False
BENCH_MAX_REPEATS = 100
BENCH_MAX_TIME_NS = 10e9
 
# ── Modal image ───────────────────────────────────────────────────────────────
 
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .pip_install("triton")
)
 
app = modal.App("trimul-kernel-eval")
 
 
# ── Evaluator function ────────────────────────────────────────────────────────
 
@app.function(gpu="H100", image=image, timeout=600)
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
    from torch import nn, einsum
 
    # ── Reference helpers ────────────────────────────────────────────────────
 
    class _TriMul(nn.Module):
        def __init__(self, dim, hidden_dim, device="cuda"):
            super().__init__()
            self.norm = nn.LayerNorm(dim, device=device)
            self.left_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)
            self.right_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)
            self.left_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
            self.right_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
            self.out_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
            self.to_out_norm = nn.LayerNorm(hidden_dim, device=device)
            self.to_out = nn.Linear(hidden_dim, dim, bias=False, device=device)
 
        def forward(self, x, mask):
            x = self.norm(x)
            left = self.left_proj(x)
            right = self.right_proj(x)
            mask = mask.unsqueeze(-1)
            left = left * mask
            right = right * mask
            left = left * self.left_gate(x).sigmoid()
            right = right * self.right_gate(x).sigmoid()
            out_gate = self.out_gate(x).sigmoid()
            out = einsum("... i k d, ... j k d -> ... i j d", left, right)
            out = self.to_out_norm(out)
            out = out * out_gate
            return self.to_out(out)
 
    def ref_kernel(data):
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            input_tensor, mask, weights, config = data
            trimul = _TriMul(dim=config["dim"], hidden_dim=config["hidden_dim"],
                             device=input_tensor.device)
            trimul.norm.weight = nn.Parameter(weights["norm.weight"])
            trimul.norm.bias = nn.Parameter(weights["norm.bias"])
            trimul.left_proj.weight = nn.Parameter(weights["left_proj.weight"])
            trimul.right_proj.weight = nn.Parameter(weights["right_proj.weight"])
            trimul.left_gate.weight = nn.Parameter(weights["left_gate.weight"])
            trimul.right_gate.weight = nn.Parameter(weights["right_gate.weight"])
            trimul.out_gate.weight = nn.Parameter(weights["out_gate.weight"])
            trimul.to_out_norm.weight = nn.Parameter(weights["to_out_norm.weight"])
            trimul.to_out_norm.bias = nn.Parameter(weights["to_out_norm.bias"])
            trimul.to_out.weight = nn.Parameter(weights["to_out.weight"])
            return trimul(input_tensor, mask)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn
 
    def generate_input(seqlen, bs, dim, hiddendim, seed, nomask, distribution="normal"):
        hidden_dim = hiddendim
        config = {"hidden_dim": hidden_dim, "dim": dim}
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
 
        if distribution == "cauchy":
            u = torch.empty((bs, seqlen, seqlen, dim), device="cuda", dtype=torch.float32)
            u.uniform_(0.0, 1.0, generator=gen)
            input_tensor = 2.0 * torch.tan(math.pi * (u - 0.5))
        else:
            input_tensor = torch.randn(
                (bs, seqlen, seqlen, dim), device="cuda", dtype=torch.float32, generator=gen
            ).contiguous()
 
        if nomask:
            mask = torch.ones(bs, seqlen, seqlen, device="cuda")
        else:
            mask = torch.randint(0, 2, (bs, seqlen, seqlen), device="cuda", generator=gen).float()
 
        weights = {
            "norm.weight": torch.randn(dim, device="cuda"),
            "norm.bias": torch.randn(dim, device="cuda"),
            "left_proj.weight": torch.randn(hidden_dim, dim, device="cuda") / math.sqrt(hidden_dim),
            "right_proj.weight": torch.randn(hidden_dim, dim, device="cuda") / math.sqrt(hidden_dim),
            "left_gate.weight": torch.randn(hidden_dim, dim, device="cuda") / math.sqrt(hidden_dim),
            "right_gate.weight": torch.randn(hidden_dim, dim, device="cuda") / math.sqrt(hidden_dim),
            "out_gate.weight": torch.randn(hidden_dim, dim, device="cuda") / math.sqrt(hidden_dim),
            "to_out_norm.weight": torch.randn(hidden_dim, device="cuda"),
            "to_out_norm.bias": torch.randn(hidden_dim, device="cuda"),
            "to_out.weight": torch.randn(dim, hidden_dim, device="cuda") / math.sqrt(dim),
        }
        return (input_tensor, mask, weights, config)
 
    def check_implementation(data, submission_output, rtol=2e-2, atol=2e-2):
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            ref_output = ref_kernel(data)
            if ref_output.shape != submission_output.shape:
                return False, f"Shape mismatch: {ref_output.shape} vs {submission_output.shape}"
            if torch.allclose(ref_output.float(), submission_output.float(), rtol=rtol, atol=atol):
                return True, "Match"
            diff = torch.abs(ref_output.float() - submission_output.float())
            return False, f"max_diff={diff.max().item():.6f}, avg_diff={diff.mean().item():.6f}"
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn
 
    # matches skydiscover modal_eval.py _eval_triton_impl
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
        if isinstance(data, torch.nn.Module):
            cloned = copy.deepcopy(data)
            if hasattr(data, 'seq_len'):
                cloned.seq_len = data.seq_len
            return cloned
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
        # Mirrors gpu-mode/reference-kernels utils.clear_l2_cache: scrub L2 between
        # timed calls so a reused `data` tensor can't read a warm cache from the
        # previous identical call and report optimistically fast.
        dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda")
        dummy.fill_(42)
        del dummy
 
    # ── Load submission ──────────────────────────────────────────────────────
 
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
            "platform": "modal-h100",
            "failure_stage": "import",
        })
 
    # ── Correctness tests (no_grad to reduce autograd memory) ───────────────
 
    test_details = []
    tests_passed = 0
    for i, tc in enumerate(TEST_CASES):
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
                "seqlen": tc["seqlen"],
                "bs": tc["bs"],
                "dim": tc["dim"],
                "hiddendim": tc["hiddendim"],
                "nomask": tc["nomask"],
                "distribution": tc["distribution"],
                "seed": tc["seed"],
                "passed": passed,
                "error": "" if passed else msg,
            })
            if passed:
                tests_passed += 1
        except Exception:
            test_details.append({
                "seqlen": tc["seqlen"],
                "bs": tc["bs"],
                "dim": tc["dim"],
                "hiddendim": tc["hiddendim"],
                "nomask": tc["nomask"],
                "distribution": tc["distribution"],
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
            "platform": "modal-h100",
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
            "platform": "modal-h100",
        })
 
    # ── Benchmarks ───────────────────────────────────────────────────────────
 
    ctx = torch.no_grad() if BENCH_NO_GRAD else contextlib.nullcontext()
    benchmark_details = []
    bench_means_ns = []
 
    for bench_args in BENCHMARK_CASES:
        data = generate_input(**bench_args)
        data_copy = _clone(data)
 
        # Correctness check, then free GPU memory before ref kernel runs
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
                "platform": "modal-h100",
                "failure_stage": "benchmark",
            })
 
        # Regenerate fresh data for timed runs
        data = generate_input(**bench_args)
 
        # Per-shape warmup: ensures torch.compile/Triton is fully warm
        # for this specific shape with these exact tensor addresses before timing
        for _ in range(3):
            custom_kernel(data)
            torch.cuda.synchronize()
 
        durations_ns = []
        bm_start = time.perf_counter_ns()
 
        with ctx:
            for t in range(BENCH_MAX_REPEATS):
                clear_l2_cache()  # scrub L2 before each timed call (not measured)
                torch.cuda.synchronize()  # flush any pending work before timing
 
                if BENCH_USE_CUDA_EVENTS:
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    output = custom_kernel(data)
                    e.record()
                    torch.cuda.synchronize()
                    duration_ns = s.elapsed_time(e) * 1e6  # ms -> ns
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
            "seqlen": bench_args["seqlen"],
            "bs": bench_args["bs"],
            "dim": bench_args["dim"],
            "hiddendim": bench_args["hiddendim"],
            "nomask": bench_args["nomask"],
            "distribution": bench_args["distribution"],
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
        "platform": "modal-h100",
    })
 
