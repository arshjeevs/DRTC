"""`lab eval report <eval>`: trials.csv, metrics.csv and report.md, always rebuilt from all sessions present
(lab/EVALUATION.md rule 3; metrics: lab/METRICS.md v1)."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from drtc.evals import metrics as M
from drtc.evals import spec as S
from drtc.evals import store
from drtc.registry import Registry

TRIAL_COLS = ["session", "machine", "server", "server_gpu", "condition", "model", "trial", "round", "valid", "success",
              "label", "time_to_success_s", "void", "outcome", "started_at", "revision", "git_commit", "path"]


def _fmt(x, d):
    return "-" if x is None else f"{x:.{d}f}"


def _pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def build(reg: Registry, name: str) -> Path:
    sp = S.load(reg, name)
    root = S.results_root(sp)
    out_dir = store.eval_root(name, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = store.trials(name, root)
    rows, mrows = [], []
    for t in trials:
        i = t.info
        rows.append({"session": t.session, "machine": i.get("machine"), "server": i.get("server"),
                     "server_gpu": i.get("server_gpu"), "condition": t.condition, "model": t.model, "trial": t.number,
                     "round": i.get("round"), "valid": int(t.valid), "success": i.get("success"),
                     "label": i.get("label"), "time_to_success_s": i.get("time_to_success_s"), "void": i.get("void"),
                     "outcome": i.get("outcome"), "started_at": i.get("started_at"), "revision": i.get("revision"),
                     "git_commit": i.get("git_commit"), "path": str(t.path)})
        if t.valid:
            mrows.append({"condition": t.condition, "model": t.model, "trial": t.number, "session": t.session,
                          "machine": i.get("machine"), "server": i.get("server"), **M.trial_metrics(t.path)})
    with open(out_dir / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_COLS)
        w.writeheader()
        w.writerows(rows)
    mcols = ["condition", "model", "trial", "session", "machine", "server", "M1_success"] + [c for c, _, _ in
                                                                                           M.CONTINUOUS]
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mcols, extrasaction="ignore")
        w.writeheader()
        w.writerows(mrows)
    (out_dir / "report.md").write_text(render(reg, name, sp, trials, mrows))
    return out_dir


def render(reg, name, sp, trials, mrows) -> str:
    rig = sp.robot == "rig"
    L = [f"# {name}: {sp.description}", "",
         f"*Rebuilt {datetime.now():%Y-%m-%d %H:%M} by `lab eval report {name}` from {len({t.session for t in trials})} "
         f"sessions. Metrics {M.VERSION} ([lab/METRICS.md](../../../lab/METRICS.md)). Do not edit: rebuilt every time.*",
         "", f"- **Task:** {S.task_of(reg, sp)}", f"- **Success rule:** {sp.success}",
         f"- **Plan:** {len(sp.models)} models x {len(sp.conditions)} conditions x {sp.trials} trials; time limit "
         f"{sp.time_limit_s:g} s; robot: {sp.robot}"]
    machines = sorted({(r["machine"], r["server"]) for r in mrows})
    if machines:
        L.append("- **Robot computer / server:** " + ", ".join(f"{m} / {s}" for m, s in machines)
                 + ("  \n  *(several: timing metrics M2-M6 are shown per robot computer and server below; METRICS.md "
                    "rule 6)*" if len(machines) > 1 else ""))
    L.append("")
    groups: dict = {}
    for r in mrows:
        groups.setdefault((r["condition"], r["model"]), []).append(r)
    # success
    if rig:
        L += ["## Success (M1)", "", "| Condition | Model | Valid | Successes | Rate | 95% interval | Time to success "
              "(s), median [IQR] |", "|---|---|---|---|---|---|---|"]
        for c in sp.conditions:
            for m in sp.models:
                g = groups.get((c, m), [])
                k, n = sum(1 for r in g if r["M1_success"] == 1), len(g)
                lo, hi = M.wilson(k, n)
                t = M.median_iqr(r["M1t_time_to_success_s"] for r in g)
                tt = "-" if t is None else f"{t[0]:.1f} [{t[1]:.1f}-{t[2]:.1f}]"
                L.append(f"| {c} | {m} | {n}/{sp.trials} | {k} | {_pct(k / n) if n else '-'} | "
                         f"{_pct(lo) if n else '-'} - {_pct(hi) if n else '-'} | {tt} |")
        L.append("")
    # continuous
    cols = [x for x in M.CONTINUOUS if rig or not x[0].startswith("M1")]
    for split in (machines if len(machines) > 1 else [None]):
        title = "## Timing and smoothness (M2-M8), median [IQR] over trials"
        if split:
            title += f": {split[0]} / {split[1]}"
        L += [title, "", "| Condition | Model | n | " + " | ".join(lbl for _, lbl, _ in cols) + " |",
              "|---|---|---|" + "---|" * len(cols)]
        for c in sp.conditions:
            for m in sp.models:
                g = [r for r in groups.get((c, m), []) if split is None or (r["machine"], r["server"]) == split]
                if not g:
                    continue
                cells = []
                for col, _, d in cols:
                    v = M.median_iqr(r.get(col) for r in g)
                    cells.append("-" if v is None else f"{v[0]:.{d}f} [{v[1]:.{d}f}-{v[2]:.{d}f}]")
                L.append(f"| {c} | {m} | {len(g)} | " + " | ".join(cells) + " |")
        L.append("")
    # comparisons against the baseline condition
    conds = list(sp.conditions)
    base = "baseline" if "baseline" in conds else conds[0]
    others = [c for c in conds if c != base]
    if others:
        L += [f"## Against `{base}` (same model; METRICS.md rule 5)", "",
              "p-values: Fisher's exact test (M1), Mann-Whitney U (the others). Δ = condition minus baseline (medians).",
              ""]
        heads = ([("M1_success", "M1")] if rig else []) + [(c, lbl.split(" (")[0]) for c, lbl, _ in cols]
        L += ["| Condition | Model | " + " | ".join(h for _, h in heads) + " |", "|---|---|" + "---|" * len(heads)]
        for c in others:
            for m in sp.models:
                a, b = groups.get((c, m), []), groups.get((base, m), [])
                if not a or not b:
                    continue
                cells = []
                for col, _ in heads:
                    if col == "M1_success":
                        ka, kb = sum(1 for r in a if r[col] == 1), sum(1 for r in b if r[col] == 1)
                        p = M.fisher_p(ka, len(a), kb, len(b))
                        cells.append(f"{_pct(ka / len(a))} vs {_pct(kb / len(b))}, p={_fmt(p, 3)}")
                    else:
                        va = M.median_iqr(r.get(col) for r in a)
                        vb = M.median_iqr(r.get(col) for r in b)
                        p = M.mannwhitney_p([r.get(col) for r in a], [r.get(col) for r in b])
                        cells.append("-" if va is None or vb is None else f"Δ {va[0] - vb[0]:+.2f}, p={_fmt(p, 3)}")
                L.append(f"| {c} | {m} | " + " | ".join(cells) + " |")
        L.append("")
    offline_rows = []
    for m in sp.models:
        f = Path("results") / "offline" / reg.resolve(m).dataset.id / f"{m}.json"
        if f.is_file():
            d = json.loads(f.read_text())
            if d.get("revision") == sp.frozen.get(m):
                offline_rows.append((m, d))
    if offline_rows:
        L += ["## Without the robot (M9, M10)", "",
              "M9: mean absolute error of the first 15 predicted actions on the held-out episodes, in the dataset's "
              "units (`root` is the served checkpoint). M10: inference time and peak GPU memory of the served "
              "checkpoint.", "",
              "| Model | Checkpoint | M9 arm | M9 gripper | M10 median (ms) | M10 p95 (ms) | GPU (GB) | Units | Measured on |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m, d in offline_rows:
            for ck, e in d["checkpoints"].items():
                b = e.get("M10") or {}
                L.append(f"| {m} | {ck} | {_fmt(e.get('M9_arm_mae'), 2)} | {_fmt(e.get('M9_gripper_mae'), 2)} | "
                         f"{_fmt(b.get('server_ms_median'), 0)} | {_fmt(b.get('server_ms_p95'), 0)} | "
                         f"{_fmt(b.get('peak_gpu_gb'), 1)} | {d.get('units')} | {d.get('gpu')} |")
        L.append("")
    voids = [t for t in trials if t.void]
    bad = [t for t in trials if not t.void and not t.valid]
    if voids or bad:
        L += ["## Not counted", "", "| Trial | Why |", "|---|---|"]
        L += [f"| {t.condition}/{t.model}/t{t.number:02d} ({t.session}) | void: {t.info['void']} |" for t in voids]
        L += [f"| {t.condition}/{t.model}/t{t.number:02d} ({t.session}) | {t.info.get('label')}: "
              f"{t.info.get('outcome')} {t.info.get('error') or ''} |" for t in bad]
        L.append("")
    L += ["## Sessions", "", "| Session | Robot computer | Server | Trials |", "|---|---|---|---|"]
    for s in sorted({t.session for t in trials}):
        ts = [t for t in trials if t.session == s]
        L.append(f"| {s} | {ts[0].info.get('machine')} | {ts[0].info.get('server')} ({ts[0].info.get('server_gpu')}) "
                 f"| {len(ts)} |")
    L.append("")
    return "\n".join(L)


def main(reg: Registry, name: str) -> int:
    out = build(reg, name)
    print(f"{out}/report.md (and trials.csv, metrics.csv)")
    return 0
