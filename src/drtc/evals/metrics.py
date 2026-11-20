"""Metrics v1 (lab/METRICS.md), computed from a trial's raw files. Changing a definition here is a new metrics
version: METRICS.md says what changed, and `lab eval report` recomputes every trial."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

VERSION = "v1"
ARM_JOINTS = 5  # the first five commanded joints are the arm (degrees); the sixth is the gripper (%)


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def longest_run(flags) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def runs(flags) -> int:
    n, prev = 0, False
    for f in flags:
        if f and not prev:
            n += 1
        prev = f
    return n


def trial_metrics(tdir: Path, fps: float = 30.0) -> dict:
    """One value per metric for one trial (METRICS.md rule 4)."""
    tdir = Path(tdir)
    info = json.loads((tdir / "trial.json").read_text())
    out: dict = {"M1_success": info.get("success"), "M1t_time_to_success_s": info.get("time_to_success_s")}
    ticks_path = tdir / "ticks.csv"
    if ticks_path.is_file():
        rows = list(csv.DictReader(open(ticks_path)))
        first = next((i for i, r in enumerate(rows) if r["action"] == "1"), None)
        run = rows[first:] if first is not None else []
        stall = [r["action"] != "1" for r in run]
        out["M2_stall_pct"] = 100.0 * sum(stall) / len(stall) if stall else None
        out["M3_stall_events"] = runs(stall) if run else None
        out["M3_longest_stall_s"] = longest_run(stall) / fps if run else None
        cmd_cols = [k for k in (rows[0].keys() if rows else []) if k.startswith("cmd.")][:ARM_JOINTS]
        if run and cmd_cols:
            held, last = [], None
            for r in run:  # the arm holds its last command through a stall
                vals = [_f(r[c]) for c in cmd_cols]
                if all(v is not None for v in vals):
                    last = vals
                if last is not None:
                    held.append(last)
            a = np.asarray(held, dtype=float)
            if len(a) >= 4:
                jerk = np.diff(a, 3, axis=0) * fps ** 3
                out["M7_jerk_k"] = float(np.sqrt((jerk ** 2).mean())) / 1000.0
            if len(a) >= 2:
                out["M8b_largest_step_deg"] = float(np.abs(np.diff(a, axis=0)).max())
    chunks_path = tdir / "chunks.csv"
    if chunks_path.is_file():
        rows = list(csv.DictReader(open(chunks_path)))
        rtt = [_f(r["turnaround_s"]) for r in rows]
        rtt = [x for x in rtt if x is not None]
        if rtt:
            out["M4_turnaround_med_ms"] = 1000 * float(np.median(rtt))
            out["M4_turnaround_p95_ms"] = 1000 * float(np.percentile(rtt, 95))
        err = [(_f(r["estimate_at_send_s"]), _f(r["turnaround_s"])) for r in rows]
        err = [e - t for e, t in err if e is not None and t is not None]
        if err:
            out["M5_estimate_err_ms"] = 1000 * float(np.median(err))
            out["M5_estimate_abs_err_ms"] = 1000 * float(np.median(np.abs(err)))
        out["M6_server_ms"] = (lambda m: None if m is None else 1000 * m)(
            _median(_f(r["server_total_s"]) for r in rows))
        out["M8a_switch_l2_deg"] = _median(_f(r["switch_l2"]) for r in rows if (_f(r["overlap_steps"]) or 0) > 0)
    return out


# --- across trials ----------------------------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def median_iqr(xs) -> tuple[float, float, float] | None:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    q1, med, q3 = np.percentile(xs, [25, 50, 75])
    return float(med), float(q1), float(q3)


def fisher_p(k1: int, n1: int, k2: int, n2: int) -> float | None:
    if min(n1, n2) == 0:
        return None
    from scipy.stats import fisher_exact

    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def mannwhitney_p(a, b) -> float | None:
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    from scipy.stats import mannwhitneyu

    return float(mannwhitneyu(a, b, alternative="two-sided")[1])


CONTINUOUS = [  # (column, label in the report, decimals)
    ("M1t_time_to_success_s", "M1t time to success (s)", 1),
    ("M2_stall_pct", "M2 stall (%)", 1),
    ("M3_stall_events", "M3 stall events", 0),
    ("M3_longest_stall_s", "M3 longest stall (s)", 2),
    ("M4_turnaround_med_ms", "M4 turnaround median (ms)", 0),
    ("M4_turnaround_p95_ms", "M4 turnaround p95 (ms)", 0),
    ("M5_estimate_err_ms", "M5 estimate error (ms)", 0),
    ("M5_estimate_abs_err_ms", "M5 abs estimate error (ms)", 0),
    ("M6_server_ms", "M6 server inference (ms)", 0),
    ("M7_jerk_k", "M7 jerk (10^3 deg/s^3)", 2),
    ("M8a_switch_l2_deg", "M8a chunk switch (deg)", 2),
    ("M8b_largest_step_deg", "M8b largest step (deg)", 2),
]
