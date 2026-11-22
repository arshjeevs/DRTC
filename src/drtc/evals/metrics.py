"""Metrics v1 (lab/METRICS.md), computed from a trial's raw files. Changing a definition here is a new metrics
version: METRICS.md says what changed, and `lab eval report` recomputes every trial."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

VERSION = "v2"
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
    from drtc.evals.store import ended_by_system

    fault = ended_by_system(info)  # from the run's record, whatever key was pressed (2026-11-22)
    out: dict = {"M1_success": 0 if fault and info.get("success") is not None else info.get("success"),
                 "M1t_time_to_success_s": None if fault else info.get("time_to_success_s"),
                 "M1f_fault": 1 if fault else 0}
    ticks_path = tdir / "ticks.csv"
    if ticks_path.is_file():
        rows = list(csv.DictReader(open(ticks_path)))
        dt = 1.0 / fps
        executed = [r for r in rows if r["action"] == "1"]
        if executed:
            t_exec = np.array([float(r["t"]) for r in executed])
            t_end = max(float(rows[-1]["t"]), t_exec[-1] + dt)  # the last row marks the end of the run
            # A stall is a tick without a new action. While the arm holds, the control loop does not ask for actions,
            # so stall time is every gap between executed actions (and after the last one) beyond one tick.
            gaps = np.diff(np.append(t_exec, t_end)) - dt
            gaps = np.where(gaps > 0.5 * dt, gaps, 0.0)  # timing jitter below half a tick is not a stall
            span = t_end - t_exec[0]
            out["M2_stall_pct"] = 100.0 * float(gaps.sum()) / span if span > 0 else None
            out["M3_stall_events"] = int((gaps > 0).sum())
            out["M3_longest_stall_s"] = float(gaps.max()) if len(gaps) else 0.0
            cmd_cols = [k for k in rows[0].keys() if k.startswith("cmd.")][:ARM_JOINTS]
            if cmd_cols:
                vals = np.array([[_f(r[c]) if _f(r[c]) is not None else np.nan for c in cmd_cols] for r in executed])
                ok = ~np.isnan(vals).any(axis=1)
                t_ok, vals = t_exec[ok], vals[ok]
                if len(t_ok):
                    # the commanded position on the tick grid, held through stalls
                    grid = np.arange(t_ok[0], t_end, dt)
                    a = vals[np.clip(np.searchsorted(t_ok, grid + 1e-9, side="right") - 1, 0, len(t_ok) - 1)]
                    if len(a) >= 4:
                        jerk = np.diff(a, 3, axis=0) * fps ** 3
                        # Metrics v3 (2026-11-22, option A): only while the arm moves. Holding still has zero jerk, so
                        # over the whole run a system that stalls more looked smoother (patient upstream: 55% stalled,
                        # jerk 35 against DRTC's 97 in the local check). A jerk sample spans 4 ticks; it counts when at
                        # least one of them executed a new action, so stopping and restarting still count, a long
                        # hold does not. Stalls themselves are M2 and M3.
                        moving = np.zeros(len(grid), dtype=bool)
                        moving[np.clip(np.round((t_ok - grid[0]) / dt).astype(int), 0, len(grid) - 1)] = True
                        window = np.convolve(moving.astype(int), np.ones(4, dtype=int), mode="valid") > 0
                        jerk = jerk[window[: len(jerk)]]
                        if len(jerk):
                            out["M7_jerk_k"] = float(np.sqrt((jerk ** 2).mean())) / 1000.0
                    if len(a) >= 2:
                        out["M8b_largest_step_deg"] = float(np.abs(np.diff(a, axis=0)).max())
            plan_cols = [k for k in rows[0].keys() if k.startswith("plan.")][:ARM_JOINTS]
            if plan_cols and cmd_cols:
                # M12 (metrics v3, 2026-11-22): how far smoothing moves the arm off the model's own plan: the mean over
                # executed ticks and the five arm joints of |commanded - the newest chunk's raw action for that step|
                dev = []
                for r in executed:
                    c = [_f(r[k]) for k in cmd_cols]
                    p_ = [_f(r[k]) for k in plan_cols]
                    if None not in c and None not in p_:
                        dev.append(np.abs(np.array(c) - np.array(p_)).mean())
                if dev:
                    out["M12_plan_deviation_deg"] = float(np.mean(dev))
            # a run the operator ended (s/f) during a stall says nothing about recovery: not counted for that fault
            operator = not fault and info.get("label") in ("s", "f")
            out.update(recovery(tdir, t_exec, t_end, dt, fault, operator))
        else:
            out["M2_stall_pct"], out["M3_stall_events"], out["M3_longest_stall_s"] = None, None, None
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


# --- M11 recovery (metrics v3, 2026-11-22, option B) -------------------------------------------------------------------

RECOVERY_FAULTS = ("spike", "drops", "outage", "rise")
RECOVERY_AFTER_S = 1.0  # a stall that starts up to this long after a fault ends is still that fault's (a reply lost
#                         near the end of a window shows when the queue runs dry; the mix's faults are 1 s apart)


def fault_ends(tdir: Path) -> dict[str, tuple[float, float]]:
    """(start, end) on the run's clock of each timed fault of the trial, from its config.yaml (with the trial's shift)."""
    import yaml

    p = Path(tdir) / "config.yaml"
    if not p.is_file():
        return {}
    c = yaml.safe_load(p.read_text()) or {}
    shift = float(c.get("fault_shift_applied_s") or 0.0)
    out = {}
    for s in c.get("spikes") or []:
        st = float(s["start_s"]) + shift
        out.setdefault("spike", (st, st + float(s["delay_ms"]) / 1000.0))
    for key, name in (("drop_action", "drops"), ("disconnect", "outage"), ("rise", "rise")):
        for w in c.get(key) or []:
            st = float(w["start_s"]) + shift
            out.setdefault(name, (st, st + float(w["duration_s"])))
    return out


def recovery(tdir: Path, t_exec, t_end: float, dt: float, system_ended: bool = False,
             operator_ended: bool = False) -> dict:
    """M11: for each timed fault, the seconds from its end until the arm moves again: the end of the first stall that
    is still going when the fault ends or starts up to RECOVERY_AFTER_S after it (0 when there is none: the arm never
    stopped). A stall that lasts to the end of the run is "not recovered": M11_recovered_<fault> = 0 and no time. A
    run that ended before the fault did is "not recovered" when the system ended it (a terminal fault), and not counted
    for that fault when the operator or the time limit ended it (e.g. the task was done at 6.8 s, before the outage
    ended at 7.5 s). Likewise a stall cut short by the operator's s/f key is not counted (the run ended, not the
    recovery): only the system or the time limit make a run "not recovered"."""
    out = {}
    ends = fault_ends(tdir)
    if not ends:
        return out
    # stalls (as M2): from one tick after an executed action to the next one; after the last action, to the end of the
    # run, where it never ends (inf: not recovered)
    starts = t_exec + dt
    stalls = [(a, b) for a, b in zip(starts[:-1], t_exec[1:]) if b - a > 0.5 * dt]
    if t_end - starts[-1] > 0.5 * dt:
        stalls.append((starts[-1], np.inf))
    for name in RECOVERY_FAULTS:
        if name not in ends:
            continue
        f_start, f_end = ends[name]
        if f_end >= t_end:  # the run ended before the fault did
            if system_ended:
                out[f"M11_recovered_{name}"], out[f"M11_recovery_{name}_s"] = 0, None
            continue
        hit = next(((a, b) for a, b in stalls if b > f_end and a < f_end + RECOVERY_AFTER_S), None)
        if hit is None:
            out[f"M11_recovered_{name}"], out[f"M11_recovery_{name}_s"] = 1, 0.0
        elif hit[1] == np.inf:
            if not operator_ended:
                out[f"M11_recovered_{name}"], out[f"M11_recovery_{name}_s"] = 0, None
        else:
            out[f"M11_recovered_{name}"], out[f"M11_recovery_{name}_s"] = 1, float(hit[1] - f_end)
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


def holm(pvals: list[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjusted p-values (None stays None)."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    order = sorted(idx, key=lambda i: pvals[i])
    out: list[float | None] = [None] * len(pvals)
    m, running = len(order), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * pvals[i]))
        out[i] = running
    return out


NOT_RECOVERED = 1e6  # in a test of M11, a run that never moved again ranks after every recovered one


def recovery_values(rows: list[dict], fault: str) -> list[float]:
    """M11 values of one fault for a rank test: the time, or NOT_RECOVERED; runs not counted for it are left out."""
    out = []
    for r in rows:
        rec = r.get(f"M11_recovered_{fault}")
        if rec in (None, ""):
            continue
        out.append(float(r[f"M11_recovery_{fault}_s"]) if int(float(rec)) == 1 else NOT_RECOVERED)
    return out


def primary_metrics(condition: str, rig: bool) -> list[str]:
    """What a primary comparison tests (metrics v3): success on the robot; on the bench the stall fraction and the
    recovery from the condition's own faults (named in its `<system>__<fault>` name)."""
    if rig:
        return ["M1_success"]
    fault = condition.rsplit("__", 1)[-1]
    faults = {"mix": ["spike", "drops", "outage"]}.get(fault, [fault] if fault in RECOVERY_FAULTS else [])
    return ["M2_stall_pct"] + [f"M11_recovery_{f}_s" for f in faults]


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
    ("M12_plan_deviation_deg", "M12 deviation from the plan (deg)", 2),
]
