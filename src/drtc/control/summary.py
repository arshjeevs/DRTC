"""Summaries of robot runs (results/<experiment>/<model>/<run_id>/): `lab results [--experiment E] [--since DATE]`.

Per run, from ticks.csv (one row per control tick) and faults.csv:
- ticks, stalls (ticks without an action), the longest stall in seconds, chunks accepted;
- turnaround median / max, the final latency estimate;
- faults injected by kind, and whether the run ended before its planned duration (with the reason, from
  rollout.log when present).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

RESULTS = Path("results")


def longest_run(flags: list[int]) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def summarize(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").is_file() else {}
    out = {"run_id": run_dir.name, "experiment": info.get("experiment", run_dir.parts[-3:-2]), "model": info.get("model"),
           "note": info.get("note", "")}
    ticks_path = run_dir / "ticks.csv"
    if ticks_path.is_file():
        rows = list(csv.DictReader(open(ticks_path)))
        fps = 30.0
        planned = None
        for part in info.get("command", []):
            if part.startswith("--fps="):
                fps = float(part.split("=", 1)[1])
            if part.startswith("--duration="):
                planned = float(part.split("=", 1)[1])
        stall = [int(r["action"]) == 0 for r in rows]
        # Stalls only count after the first action (start-up waits for the first chunk by design).
        first = next((i for i, s in enumerate(stall) if not s), len(stall))
        stall = stall[first:]
        tt = sorted(float(r["last_turnaround"]) for r in rows if r["last_turnaround"])
        last_t = float(rows[-1]["t"]) if rows else 0.0
        out.update({
            "ticks": len(rows),
            "stalls": sum(stall),
            "longest_stall_s": round(longest_run(stall) / fps, 2),
            "chunks": int(rows[-1]["accepted"]) if rows else 0,
            "turnaround_med_ms": round(1000 * tt[len(tt) // 2]) if tt else None,
            "turnaround_max_ms": round(1000 * tt[-1]) if tt else None,
            "ran_s": round(last_t, 1),
            "ended_early": bool(planned and last_t < planned - 1.0),
        })
    faults_path = run_dir / "faults.csv"
    if faults_path.is_file():
        out["faults"] = dict(Counter(r["fault"] for r in csv.DictReader(open(faults_path))))
    out["outcome"] = info.get("outcome", "completed")
    if info.get("error"):
        out["fault_reason"] = info["error"][-200:]
    rec = run_dir / "recovery.csv"
    if rec.is_file():
        out["recovery"] = dict(Counter(r["action"] for r in csv.DictReader(open(rec))))
    log = run_dir / "rollout.log"
    if log.is_file():
        reasons = [line.strip() for line in log.read_text(errors="replace").splitlines()
                   if "fault" in line.lower() and ("ERROR" in line or "Faulted" in line or "deadline" in line)]
        if reasons and "fault_reason" not in out:
            out["fault_reason"] = reasons[0][-160:]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab results", description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiment", help="only runs of this experiment group/name (prefix)")
    ap.add_argument("--model")
    ap.add_argument("--since", help="only runs with run_id >= this (e.g. 20261118)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    runs = sorted(p.parent for p in RESULTS.rglob("run.json"))
    rows = []
    for d in runs:
        s = summarize(d)
        exp = s["experiment"] if isinstance(s["experiment"], str) else ""
        if a.experiment and not exp.startswith(a.experiment):
            continue
        if a.model and s.get("model") != a.model:
            continue
        if a.since and s["run_id"] < a.since:
            continue
        rows.append(s)
    if a.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{'experiment':28s} {'model':26s} {'ran_s':>5} {'stalls':>6} {'longest':>7} {'chunks':>6} "
          f"{'rtt med/max ms':>14}  faults")
    for s in rows:
        rtt = f"{s.get('turnaround_med_ms')}/{s.get('turnaround_max_ms')}"
        early = f" ENDED EARLY ({s.get('outcome')})" if s.get("ended_early") or s.get("outcome") not in (None, "completed") else ""
        faults = ", ".join(f"{k} {v}" for k, v in (s.get("faults") or {}).items())
        if s.get("recovery"):
            faults += " | recovered: " + ", ".join(f"{k} {v}" for k, v in s["recovery"].items())
        print(f"{str(s['experiment']):28s} {str(s.get('model')):26s} {s.get('ran_s', '-'):>5} {s.get('stalls', '-'):>6} "
              f"{s.get('longest_stall_s', '-'):>7} {s.get('chunks', '-'):>6} {rtt:>14}  {faults}{early}")
        if s.get("fault_reason"):
            print(f"{'':28s}   reason: {s['fault_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
