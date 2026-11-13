"""Summarise a training log: where the time goes, the bottleneck, how long the run takes, loss trends.

LeRobot logs one line per `log_freq` steps:
    step:200 ... loss:0.093 ... updt_s:0.341 data_s:0.012
updt_s = forward + backward + optimizer (GPU work), data_s = waiting for the next batch (video decoding,
augmentation, loading). Both are averages over the steps since the previous line.

Usage:
    python -m lab.tools.logstats <model name | log file name | path>   (model: its latest log in logs/)
"""

from __future__ import annotations

import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

from lab.tools.config import REPO_ROOT

STEP = re.compile(r"^INFO (\S+ \S+) .*?step:(\S+) .*?epch:(\S+) loss:(\S+) .*?updt_s:(\S+) data_s:(\S+)")
VAL = re.compile(r"step:(\S+) val_loss:(\S+)")
CONFIG = {key: re.compile(rf"'{key}': ([^,}}]+)") for key in ("batch_size", "num_workers", "steps")}
POLICY_TYPE = re.compile(r"'type': '([^']+)'")


def number(text: str) -> float:
    """'2K' / '1.5M' / '200' -> float (LeRobot abbreviates step counts)."""
    scale = {"K": 1e3, "M": 1e6, "B": 1e9}.get(text[-1], 1)
    return float(text[:-1] if scale != 1 else text) * scale


def find_log(arg: str) -> Path:
    """A log path, a file name in logs/, or a model name (its latest log; train_ prefix optional)."""
    logs_dir = REPO_ROOT / "logs"
    for path in (Path(arg), logs_dir / arg):
        if path.is_file():
            return path
    model = arg.removesuffix(".log").removeprefix("train_")
    # The date-time suffix keeps smolvla_x_full from also matching smolvla_x_full_short.
    exact = re.compile(rf"^train_{re.escape(model)}_\d{{8}}-\d{{6}}\.log$")
    logs = sorted((p for p in logs_dir.glob("train_*.log") if exact.match(p.name)), key=lambda p: p.name[-19:])
    if not logs:
        sys.exit(f"No log {arg!r}: not a file, not in logs/, and no logs/train_{model}_<date-time>.log")
    # Skip restarts that crashed before training: use the latest log with at least two step lines.
    trained = [p for p in logs if sum(1 for line in p.read_text(errors="ignore").splitlines() if STEP.match(line)) >= 2]
    if trained and trained[-1] != logs[-1]:
        print(f"({len(logs) - 1 - logs.index(trained[-1])} newer log(s) of {model} have no training steps; "
              f"using the latest one that does)")
    return (trained or logs)[-1]


def fmt_hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python -m lab.tools.logstats <model name | log file>")
    log = find_log(sys.argv[1])
    text = log.read_text(errors="ignore")
    config = {key: (m.group(1).strip("' ") if (m := rx.search(text)) else "?") for key, rx in CONFIG.items()}
    policy_start = text.find("'policy': {")
    m = POLICY_TYPE.search(text, policy_start) if policy_start >= 0 else None
    config["type"] = m.group(1) if m else "?"
    rows = [(datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S"), number(m[2]), float(m[3]), float(m[4]), float(m[5]),
             float(m[6])) for line in text.splitlines() if (m := STEP.match(line))]
    vals = [(number(s), float(v)) for s, v in VAL.findall(text)]
    print(f"log: {log}")
    print(f"policy {config['type']}, batch {config['batch_size']}, num_workers {config['num_workers']}, "
          f"target {config['steps']} steps")
    if len(rows) < 2:
        sys.exit("Fewer than two step lines yet; wait for the run to log more.")

    rows = rows[1:] if len(rows) > 3 else rows  # the first interval includes start-up and warm-up
    update = statistics.median(r[4] for r in rows)
    data = statistics.median(r[5] for r in rows)
    (t0, s0, *_), (t1, s1, *_) = rows[0], rows[-1]
    wall = (t1 - t0).total_seconds() / (s1 - s0) if s1 > s0 else update + data
    share = data / (update + data) if update + data else 0
    print(f"\nper step (median): GPU work {update:.3f} s + data loading {data:.3f} s = {update + data:.3f} s "
          f"(data loading {share:.0%})")
    print(f"measured from timestamps: {wall:.3f} s/step"
          + (f"  ({wall - update - data:+.3f} s of other overhead per step)" if wall > 1.15 * (update + data) else ""))
    if config["steps"] != "?":
        total = number(config["steps"])
        print(f"run length at this speed: {fmt_hours(wall * total)} for {total:.0f} steps; "
              f"{fmt_hours(wall * max(0, total - s1))} left (at step {s1:.0f})")

    print("\nverdict:")
    if share > 0.3:
        print(f"  DATA LOADING is the bottleneck ({share:.0%} of each step is waiting for batches).")
        print(f"  -> raise --num_workers (now {config['num_workers']}); try 16 on a server, at most the CPU count")
        print("     (nproc). Watch `htop`: if every CPU core is busy, decoding is CPU-bound and more workers won't help.")
    else:
        print(f"  GPU WORK is the bottleneck (data loading only {share:.0%}).")
        print("  -> check `nvidia-smi`: other jobs on the same GPU slow every step; GPU-Util should be near 100%.")
        print("  -> bf16 speeds up most GPUs: --policy.dtype=bfloat16 where the policy supports it (pi0, pi05, X-VLA),")
        print("     or ACCELERATE_MIXED_PRECISION=bf16 lab/bin/train ... for the others (check the loss stays similar).")
    if wall > 1.15 * (update + data):
        print("  Extra overhead outside update/data: validation, checkpoint saving or wandb uploads at every log.")

    losses = [r[3] for r in rows]
    print(f"\nloss: {losses[0]:.4f} -> {losses[-1]:.4f} (epoch {rows[-1][2]:.1f})")
    if vals:
        best = min(vals, key=lambda v: v[1])
        print(f"val_loss: {vals[0][1]:.4f} (step {vals[0][0]:.0f}) -> {vals[-1][1]:.4f} (step {vals[-1][0]:.0f}); "
              f"best {best[1]:.4f} at step {best[0]:.0f}")
        if vals[-1][1] > best[1] * 1.1 and vals[-1][0] > best[0]:
            print("  val_loss has risen >10% above its best while training continued: possible overfitting.")


if __name__ == "__main__":
    main()
