"""Training speed from a lerobot-train log (Phase 3 baseline): tests/bench/train_speed.sh runs it per model.

    .venv/bin/python tests/bench/speed_report.py <model> <train log> [--gpu-csv <nvidia-smi log>] [--warmup 50]

LeRobot 0.7 logs, every log_freq steps, the window's averages: data_s (waiting for the next batch), prep_s (moving
it to the GPU, augmentation, normalization), updt_s (forward + backward + optimizer), step_s (all of it) and smp/s.
Windows ending at or before `--warmup` steps are left out (compilation, cache warmup). The GPU log, if given, is
nvidia-smi's utilization sampled during the run; only samples from after the warmup are averaged.

The projection is the recipe's full run length (`lab train` without --steps) at the measured step time. It does
not include held-out evaluation or checkpoint saves.

Prints one summary line and writes logs/bench/speed_<model>.json.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

LINE = re.compile(r"^INFO (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) .*?\bstep:(\d+(?:\.\d+)?[KM]?) .*\bstep_s:")
FIELD = re.compile(r"\b(data_s|prep_s|updt_s|step_s|smp/s|mem_gb):([0-9.]+)")


def big(s: str) -> int:
    mult = {"K": 1_000, "M": 1_000_000}.get(s[-1], 1)
    return int(float(s.rstrip("KM")) * mult)


def parse_log(text: str) -> list[dict]:
    """One record per logged training window: step, time, and the timing fields."""
    out = []
    for line in text.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        rec = {"step": big(m.group(2)), "time": datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")}
        rec.update({k: float(v) for k, v in FIELD.findall(line)})
        out.append(rec)
    return out


def parse_gpu(text: str) -> list[tuple[datetime, float]]:
    """nvidia-smi --query-gpu=timestamp,utilization.gpu --format=csv,noheader,nounits (first GPU's rows)."""
    out = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            t = datetime.strptime(parts[0].split(".")[0], "%Y/%m/%d %H:%M:%S")
            out.append((t, float(parts[1])))
        except ValueError:
            continue
    return out


def summarize(windows: list[dict], warmup: int, gpu: list[tuple[datetime, float]] | None = None) -> dict:
    kept = [w for w in windows if w["step"] > warmup]
    if not kept:
        raise ValueError(f"no logged window after step {warmup} (logged steps: {[w['step'] for w in windows]})")
    res = {"measured_steps": f"{warmup}-{kept[-1]['step']}", "windows": len(kept)}
    for k in ("data_s", "prep_s", "updt_s", "step_s", "smp/s"):
        vals = [w[k] for w in kept if k in w]
        if vals:
            res[k] = round(statistics.mean(vals), 4)
    mems = [w["mem_gb"] for w in windows if "mem_gb" in w]
    if mems:
        res["peak_mem_gb"] = max(mems)
    if "step_s" in res:
        res["data_share"] = round(res.get("data_s", 0.0) / res["step_s"], 3)
    if gpu:
        start = max((w["time"] for w in windows if w["step"] <= warmup), default=kept[0]["time"])
        util = [u for t, u in gpu if start <= t <= kept[-1]["time"]]
        if util:
            res["gpu_util_pct"] = round(statistics.mean(util), 1)
    return res


def projection(model: str) -> dict:
    """The recipe's full run length and batch size (no download: the registry and the dataset's metadata only)."""
    from drtc.registry import Registry
    from drtc.train import build

    flags, _info = build(Registry(), model, download=False)
    return {"recipe_steps": int(flags["steps"]), "batch_size": int(flags["batch_size"])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("log", type=Path)
    ap.add_argument("--gpu-csv", type=Path)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path("logs/bench"))
    a = ap.parse_args(argv)
    gpu = parse_gpu(a.gpu_csv.read_text()) if a.gpu_csv and a.gpu_csv.is_file() else None
    try:
        res = summarize(parse_log(a.log.read_text(errors="replace")), a.warmup, gpu)
    except ValueError as e:
        print(f"{a.model}: no measurement ({e})", flush=True)
        return 1
    res = {"model": a.model, **res, **projection(a.model)}
    res["projected_hours"] = round(res["recipe_steps"] * res["step_s"] / 3600, 1)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"speed_{a.model}.json").write_text(json.dumps(res, indent=1))
    util = f"  GPU {res['gpu_util_pct']}%" if "gpu_util_pct" in res else ""
    print(f"{a.model}: step {res['step_s']:.3f} s (data {res.get('data_s', 0):.3f}, prep {res.get('prep_s', 0):.3f}, "
          f"update {res.get('updt_s', 0):.3f}; data {res['data_share']:.0%} of the step){util}  |  "
          f"{res.get('smp/s', 0):.0f} samples/s at batch {res['batch_size']}  |  peak {res.get('peak_mem_gb')} GB  |  "
          f"full recipe {res['recipe_steps']} steps ≈ {res['projected_hours']} h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
