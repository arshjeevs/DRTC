"""Inference time and GPU memory of a kept model in the OLD setup on this machine (the server's per-request path:
preprocess -> predict_action_chunk -> postprocess), on real dataset frames.

    ~/Projects/drtc/.venv-groot/bin/python tests/parity/bench_old.py groot_cubebox_v1_lora_raw [--n 30]
Writes ~/Projects/drtc-snapshots/parity/bench_<model>_<host>.json
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import time
from pathlib import Path

import torch

from make_reference import LEROBOT, MODELS, OUT, U  # same model table and paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()
    policy_type, finetune, ds_dir, _ = MODELS[a.model]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.lora import load_policy

    hub = f"{U}/{a.model}"
    t0 = time.perf_counter()
    policy = load_policy(get_policy_class(policy_type), hub, finetune)
    policy.to("cuda").eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=hub,
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    load_s = time.perf_counter() - t0
    ds = LeRobotDataset(f"{U}/{ds_dir}", root=LEROBOT / ds_dir, video_backend="pyav")
    frames = [ds[i] for i in range(0, len(ds), max(1, len(ds) // (a.n + a.warmup)))][: a.n + a.warmup]

    def batch_of(item):
        b = {k: v.unsqueeze(0) for k, v in item.items() if k.startswith("observation.")}
        b["task"] = [item["task"]]
        return b

    torch.cuda.reset_peak_memory_stats()
    times = []
    with torch.no_grad():
        for i, item in enumerate(frames):
            b = batch_of(item)
            torch.cuda.synchronize()
            t = time.perf_counter()
            chunk = policy.predict_action_chunk(pre(b))
            post(chunk)
            torch.cuda.synchronize()
            if i >= a.warmup:
                times.append((time.perf_counter() - t) * 1000)
    times.sort()
    res = {
        "model": a.model, "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "load_s": round(load_s, 1), "n": len(times),
        "ms_mean": round(statistics.mean(times), 1), "ms_p50": round(times[len(times) // 2], 1),
        "ms_p90": round(times[int(len(times) * 0.9)], 1), "ms_max": round(times[-1], 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "chunk_actions": int(chunk.shape[1]),
    }
    print(json.dumps(res, indent=2))
    (OUT / f"bench_{a.model}_{res['host']}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
