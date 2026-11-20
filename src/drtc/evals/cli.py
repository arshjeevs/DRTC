"""`lab eval list | check | freeze | status` (lab/EVALUATION.md). More commands come with Phase 5 steps 3-7."""

from __future__ import annotations

import argparse
import sys

from drtc.evals import spec as S
from drtc.evals import store
from drtc.registry import Registry, RegistryError


def cmd_list(reg: Registry, a) -> int:
    d = S.evals_dir(reg)
    names = sorted(p.stem for p in d.glob("*.yaml")) if d.is_dir() else []
    if not names:
        print("no evals yet (lab/evals/<eval>.yaml; see lab/EVALUATION.md)")
        return 0
    for n in names:
        try:
            sp = S.load(reg, n)
        except RegistryError as e:
            print(f"  {n:28s} INVALID: {e}")
            continue
        done = sum(min(v, sp.trials) for v in store.valid_counts(n).values())
        total = len(sp.pairs()) * sp.trials
        state = "frozen" if S.is_frozen(sp) else "not frozen"
        print(f"  {n:28s} {done:4d}/{total:<4d} trials  {state:10s}  {sp.description}")
    return 0


def cmd_check(reg: Registry, a) -> int:
    sp = S.load(reg, a.eval)
    problems = S.check(reg, sp)
    for p in problems:
        print(f"  PROBLEM {p}")
    if problems:
        return 1
    print(f"lab/evals/{a.eval}.yaml: ok ({len(sp.models)} models x {len(sp.conditions)} conditions x "
          f"{sp.trials} trials = {len(sp.pairs()) * sp.trials}; {'frozen' if S.is_frozen(sp) else 'not frozen yet'})")
    return 0


def cmd_freeze(reg: Registry, a) -> int:
    sp = S.load(reg, a.eval)
    problems = S.check(reg, sp)
    if problems:
        for p in problems:
            print(f"  PROBLEM {p}")
        return 1
    current = S.hub_commits(reg, sp.models)
    if sp.frozen == current:
        print(f"{a.eval}: already frozen at the current Hub commits; nothing to do")
        return 0
    started = store.trials(a.eval)
    if sp.frozen and started:
        changed = [m for m in sp.models if sp.frozen.get(m) != current.get(m)]
        print(f"lab eval freeze: {a.eval} already has {len(started)} trials at its frozen versions, and these models "
              f"changed on the Hub since: {', '.join(changed)}.\n"
              f"  A frozen plan with trials is never changed (lab/EVALUATION.md rule 4): make a new eval for the new "
              f"versions, e.g. lab/evals/{a.eval}_v2.yaml.", file=sys.stderr)
        return 1
    path = S.eval_path(reg, a.eval)
    S.write_frozen(path, current)
    for m in sp.models:
        print(f"  {m:32s} {current[m]}")
    print(f"frozen: lab/evals/{a.eval}.yaml. Commit and push it, so every robot computer runs the same plan:\n"
          f"  git add lab/evals/{a.eval}.yaml && git commit -m \"eval {a.eval}: frozen\" && git push")
    return 0


def cmd_status(reg: Registry, a) -> int:
    sp = S.load(reg, a.eval)
    counts = store.valid_counts(a.eval)
    all_trials = store.trials(a.eval)
    voids: dict[tuple[str, str], int] = {}
    for t in all_trials:
        if t.void:
            voids[(t.condition, t.model)] = voids.get((t.condition, t.model), 0) + 1
    total = len(sp.pairs()) * sp.trials
    done = sum(min(v, sp.trials) for v in counts.values())
    print(f"{a.eval}: {sp.description}")
    print(f"  task:    {S.task_of(reg, sp)}")
    print(f"  success: {sp.success}")
    print(f"  plan:    {len(sp.models)} models x {len(sp.conditions)} conditions x {sp.trials} trials = {total}; "
          f"time limit {sp.time_limit_s:g} s; video {'on' if sp.video else 'off'}")
    if S.is_frozen(sp):
        print("  frozen:  yes")
    else:
        missing = [m for m in sp.models if not sp.frozen.get(m)]
        print(f"  frozen:  NO ({', '.join(missing)}): run `lab eval freeze {a.eval}` and commit")
    for c, exp in sp.conditions.items():
        print(f"  condition {c} ({exp})")
        for m in sp.models:
            n, v = counts.get((c, m), 0), voids.get((c, m), 0)
            bar = "#" * min(n, sp.trials) + "." * max(0, sp.trials - n)
            print(f"    {m:32s} {n:3d}/{sp.trials}  {bar}" + (f"  ({v} void)" if v else ""))
    rnd, pairs = S.next_round(sp, counts)
    if rnd:
        print(f"  next:    round {rnd}: " + ", ".join(f"{m}" + (f" [{c}]" if len(sp.conditions) > 1 else "")
                                                     for c, m in pairs))
    else:
        print("  next:    complete")
    sess = store.sessions(a.eval)
    waiting = store.unpushed(a.eval)
    print(f"  done:    {done}/{total} valid trials in {len(sess)} sessions on this machine"
          + (f"; NOT PUSHED: {', '.join(s.name for s in waiting)}" if waiting else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab eval", description="Official evaluations (lab/EVALUATION.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="every eval: trials done, frozen or not").set_defaults(fn=cmd_list)
    for name, fn, help_ in (("check", cmd_check, "check an eval file against the registry"),
                            ("freeze", cmd_freeze, "pin every model to its current Hub commit (then commit the file)"),
                            ("status", cmd_status, "trials done per model and condition, the next round, unpushed "
                                                   "sessions")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("eval", help="the eval's name (lab/evals/<eval>.yaml)")
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    reg = Registry()
    try:
        return a.fn(reg, a)
    except RegistryError as e:
        print(f"lab eval: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
