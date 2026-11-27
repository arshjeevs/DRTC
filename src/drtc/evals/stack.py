"""The final smoothing stack from the E3a bench (setup v2; rule fixed with the user 2026-11-22, before E3a ran):

    .venv/bin/python -m drtc.evals.stack            # reads results/bench/bench_v2_smooth{,_rtc}_<dataset>/metrics.csv

The stack is adaptive playback + the resume ramp + one seam method + one jitter method. Two choices:
- seam: RTC or temporal ensembling, decided on the pairs (dataset, model) that can run RTC (SmolVLA, GR00T N1.5);
  models without RTC use ensembling whatever the choice;
- jitter: zero-lag or Butterworth, decided on every pair.

Per pair, each candidate is compared with DRTC alone (`drtc__mix`) **in its own eval** (RTC's reference is in the RTC
eval). A candidate is ruled out in a pair if, against that reference, it
- adds more than 1 deg of median deviation from the model's plan (M12), or
- raises the median stall fraction (M2) by more than 10% (at least 0.5 percentage points, for references near 0), or
- raises the median recovery time (M11) of any fault by more than 10% (at least 0.05 s), or recovers in fewer runs.
Of the candidates left, the lower jerk relative to its reference (median M7 / the reference's) wins the pair; within
1%, the lower largest step relative to its reference (M8b). The candidate that wins more pairs is chosen; a tie goes
to the lower sum of relative jerks. A slot whose candidates are ruled out in every pair is left empty.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

DATASETS = ("sort_one", "cubebox_v1")
REF = "drtc__mix"
SEAM = {"rtc": ("bench_v2_smooth_rtc_{ds}", "s_rtc__mix"), "ensemble": ("bench_v2_smooth_{ds}", "s_ensemble__mix")}
JITTER = {"zero_lag": ("bench_v2_smooth_{ds}", "s_zero_lag__mix"),
          "butterworth": ("bench_v2_smooth_{ds}", "s_butterworth__mix")}
M12_MAX_DEG = 1.0
REL = 0.10
M2_FLOOR_PP = 0.5
M11_FLOOR_S = 0.05
TIE = 0.01
FAULTS = ("spike", "drops", "outage")


def load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def med(rows: list[dict], col: str) -> float | None:
    xs = [float(r[col]) for r in rows if r.get(col) not in (None, "", "None")]
    return statistics.median(xs) if xs else None


def frac(rows: list[dict], col: str) -> float | None:
    xs = [float(r[col]) for r in rows if r.get(col) not in (None, "", "None")]
    return sum(xs) / len(xs) if xs else None


def judge(cand: list[dict], ref: list[dict]) -> tuple[bool, str, float | None, float | None]:
    """(passes the guards, why not, jerk relative to the reference, largest step relative to the reference)."""
    if not cand or not ref:
        return False, "no runs", None, None
    d12 = (med(cand, "M12_plan_deviation_deg") or 0.0) - (med(ref, "M12_plan_deviation_deg") or 0.0)
    if d12 > M12_MAX_DEG:
        return False, f"M12 +{d12:.2f} deg", None, None
    c2, r2 = med(cand, "M2_stall_pct"), med(ref, "M2_stall_pct")
    if c2 is not None and r2 is not None and c2 > r2 + max(REL * r2, M2_FLOOR_PP):
        return False, f"M2 {r2:.1f} -> {c2:.1f}%", None, None
    for f in FAULTS:
        ct, rt = med(cand, f"M11_recovery_{f}_s"), med(ref, f"M11_recovery_{f}_s")
        if ct is not None and rt is not None and ct > rt + max(REL * rt, M11_FLOOR_S):
            return False, f"M11 {f} {rt:.2f} -> {ct:.2f} s", None, None
        cr, rr = frac(cand, f"M11_recovered_{f}"), frac(ref, f"M11_recovered_{f}")
        if cr is not None and rr is not None and cr < rr:
            return False, f"M11 {f} recovered {rr:.0%} -> {cr:.0%}", None, None
    j, jr = med(cand, "M7_jerk_k"), med(ref, "M7_jerk_k")
    s, sr = med(cand, "M8b_largest_step_deg"), med(ref, "M8b_largest_step_deg")
    return True, "", (j / jr if j is not None and jr else None), (s / sr if s is not None and sr else None)


def decide(options: dict, root: Path, only_models: set[str] | None = None) -> tuple[str | None, list[str]]:
    wins = {k: 0 for k in options}
    jsum = {k: 0.0 for k in options}
    lines = []
    for ds in DATASETS:
        evals = {k: load(root / e.format(ds=ds) / "metrics.csv") for k, (e, _) in options.items()}
        models = sorted({r["model"] for rows in evals.values() for r in rows})
        for m in models:
            if only_models is not None and not any(m.startswith(p) for p in only_models):
                continue
            res = {}
            for k, (e, cond) in options.items():
                rows = evals[k]
                cand = [r for r in rows if r["condition"] == cond and r["model"] == m]
                ref = [r for r in rows if r["condition"] == REF and r["model"] == m]
                res[k] = judge(cand, ref)
            ok = {k: v for k, v in res.items() if v[0] and v[2] is not None}
            for k, v in ok.items():
                jsum[k] += v[2]
            win = None
            if ok:
                best = sorted(ok, key=lambda k: ok[k][2])
                win = best[0]
                if len(best) > 1 and abs(ok[best[1]][2] - ok[best[0]][2]) <= TIE:
                    win = min(best[:2], key=lambda k: ok[k][3] if ok[k][3] is not None else 9e9)
                wins[win] += 1
            cells = [f"{k}: " + (f"jerk x{v[2]:.2f}, step x{v[3]:.2f}" if v[0] and v[2] is not None else f"out ({v[1]})")
                     for k, v in res.items()]
            lines.append(f"  {ds:10s} {m:32s} {' | '.join(cells)} -> {win or 'none'}")
    if not any(wins.values()):
        return None, lines
    top = max(wins.values())
    tied = [k for k, w in wins.items() if w == top]
    choice = min(tied, key=lambda k: jsum[k]) if len(tied) > 1 else tied[0]
    lines.append(f"  wins: {wins}; chosen: {choice}")
    return choice, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m drtc.evals.stack", description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="results/bench")
    a = ap.parse_args(argv)
    root = Path(a.root)
    seam, l1 = decide(SEAM, root, only_models={"smolvla_", "groot_"})
    jitter, l2 = decide(JITTER, root)
    print("seam (RTC-capable pairs):", *l1, sep="\n")
    print("jitter (every pair):", *l2, sep="\n")
    parts = ["adaptive", *(p for p in (seam, jitter) if p), "ramp"]
    print(f"\nfinal stack: {' + '.join(parts)}  (models without RTC use ensemble instead of rtc)")
    print(f"next: .venv/bin/python lab/experiments/v2/_make.py --stack {','.join(parts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
