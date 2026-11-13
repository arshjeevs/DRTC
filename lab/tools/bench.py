"""Inference benchmark: how fast and how memory-hungry a registry model is on this machine.

Loads the model the way the policy server does (full or LoRA, with the checkpoint's processors), feeds
dummy observations with the dataset's camera and state shapes, and times whole action-chunk predictions
(preprocess + model + postprocess) after a warm-up. Variants: --dtype bf16 (autocast), --compile
(torch.compile), --steps for the number of denoising steps of flow policies (smolvla, pi0, pi05).

Results are printed and saved to results/benchmark/<host>/<model>__<variant>.json, so the same command
on the laptop, the GPU servers and the Pi builds one comparison (lab/bin/analyze reads them).

Usage:
    python -m lab.tools.bench <model> [--device cuda|cpu] [--dtype fp32|bf16] [--compile]
                                      [--steps 10 8 4 2] [--runs 30] [--warmup 5] [--threads N]
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import statistics
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch

from lab.tools.config import RESULTS_DIR, load_models

STEP_FIELDS = ("num_steps", "num_inference_steps")  # smolvla / pi0, pi05


def device_name(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    return f"{platform.processor() or platform.machine()} CPU ({torch.get_num_threads()} threads)"


def load(entry: dict, device: str):
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.lora import load_policy

    path, finetune = entry["pretrained_name_or_path"], entry.get("finetune", "full")
    policy = load_policy(get_policy_class(entry["policy_type"]), path, finetune)
    policy.config.device = device
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def dummy_observation(policy, task: str) -> dict:
    """Shapes from the policy config, which records the training dataset's cameras and state."""
    obs = {key: torch.rand(1, *feature.shape) for key, feature in policy.config.image_features.items()}
    state = policy.config.robot_state_feature
    if state is not None:
        obs["observation.state"] = torch.randn(1, *state.shape)
    obs["task"] = [task]
    return obs


def set_steps(policy, steps: int | None) -> int | None:
    """Set the denoising steps of a flow policy; returns the value in effect (None if not a flow policy)."""
    for field in STEP_FIELDS:
        if hasattr(policy.config, field):
            if steps is not None:
                setattr(policy.config, field, steps)
            return getattr(policy.config, field)
    if steps is not None:
        sys.exit(f"--steps applies to flow policies (smolvla, pi0, pi05), not {policy.config.type}")
    return None


def time_chunks(policy, pre, post, obs: dict, device: str, dtype: str, runs: int, warmup: int) -> dict:
    autocast = torch.autocast(device, dtype=torch.bfloat16) if dtype == "bf16" else nullcontext()

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def once() -> tuple[float, float]:
        sync()
        t0 = time.perf_counter()
        batch = pre(dict(obs))
        sync()
        t1 = time.perf_counter()
        policy.reset()
        chunk = policy.predict_action_chunk(batch)
        post(chunk)
        sync()
        return time.perf_counter() - t0, time.perf_counter() - t1

    with torch.inference_mode(), autocast:
        first = once()[0]
        for _ in range(warmup):
            once()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        times = [once() for _ in range(runs)]
    total = sorted(t for t, _ in times)
    model = sorted(m for _, m in times)
    pct = lambda xs, q: xs[min(len(xs) - 1, int(q * len(xs)))]  # noqa: E731
    return {
        "first_call_ms": first * 1e3,
        "total_ms_median": statistics.median(total) * 1e3,
        "total_ms_p90": pct(total, 0.9) * 1e3,
        "model_ms_median": statistics.median(model) * 1e3,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference benchmark for a lab/models.yaml entry")
    parser.add_argument("model")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model before timing")
    parser.add_argument("--steps", type=int, nargs="+", help="denoising steps to sweep (flow policies)")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, help="CPU threads (torch.set_num_threads)")
    args = parser.parse_args()

    models = load_models()
    if args.model not in models:
        sys.exit(f"Unknown model {args.model!r}. Registered: {', '.join(models)}")
    entry = models[args.model]
    if args.threads:
        torch.set_num_threads(args.threads)

    t0 = time.perf_counter()
    policy, pre, post = load(entry, args.device)
    load_s = time.perf_counter() - t0
    if args.compile:
        policy.model = torch.compile(policy.model)
    params = sum(p.numel() for p in policy.parameters())
    obs = dummy_observation(policy, entry.get("task", ""))
    host = socket.gethostname()
    print(f"{args.model} ({entry['policy_type']}, {entry.get('finetune', 'full')}, {params / 1e6:.0f}M params) "
          f"on {device_name(args.device)} [{host}]; loaded in {load_s:.1f} s")
    print(f"  {'variant':28s} {'steps':>5s} {'total ms':>9s} {'p90 ms':>8s} {'model ms':>9s} {'1st call ms':>11s} {'peak GB':>8s}")

    out_dir = RESULTS_DIR / "benchmark" / host
    out_dir.mkdir(parents=True, exist_ok=True)
    for steps in args.steps or [None]:
        in_effect = set_steps(policy, steps)
        result = time_chunks(policy, pre, post, obs, args.device, args.dtype, args.runs, args.warmup)
        variant = f"{args.device}_{args.dtype}" + ("_compile" if args.compile else "") + (
            f"_steps{in_effect}" if in_effect is not None else "")
        record = {
            "model": args.model, "policy_type": entry["policy_type"], "finetune": entry.get("finetune", "full"),
            "variant": variant, "device": args.device, "device_name": device_name(args.device), "host": host,
            "dtype": args.dtype, "compile": args.compile, "denoising_steps": in_effect,
            "chunk_size": getattr(policy.config, "chunk_size", None), "params_m": params / 1e6,
            "load_s": load_s, "runs": args.runs, "torch": torch.__version__,
            "date": datetime.now().isoformat(timespec="seconds"), **result,
        }
        (out_dir / f"{args.model}__{variant}.json").write_text(json.dumps(record, indent=2))
        peak = f"{result['peak_memory_gb']:.2f}" if result["peak_memory_gb"] is not None else "-"
        print(f"  {variant:28s} {str(in_effect or '-'):>5s} {result['total_ms_median']:9.1f} "
              f"{result['total_ms_p90']:8.1f} {result['model_ms_median']:9.1f} {result['first_call_ms']:11.1f} {peak:>8s}")
    print(f"saved to {out_dir}/")


if __name__ == "__main__":
    main()
