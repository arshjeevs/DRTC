"""`lab eval list | check | freeze | status | serve | servers` (lab/EVALUATION.md). More come with Phase 5 steps 5-7."""

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
        done = sum(min(v, sp.trials) for v in store.valid_counts(n, S.results_root(sp)).values())
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
    started = store.trials(a.eval, S.results_root(sp))
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
    root = S.results_root(sp)
    counts = store.valid_counts(a.eval, root)
    all_trials = store.trials(a.eval, root)
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
    sess = store.sessions(a.eval, root)
    waiting = store.unpushed(a.eval, root)
    print(f"  done:    {done}/{total} valid trials in {len(sess)} sessions on this machine"
          + (f"; NOT PUSHED: {', '.join(s.name for s in waiting)}" if waiting else ""))
    return 0


def _models(arg: str | None) -> list[str] | None:
    return [m.strip() for m in arg.split(",") if m.strip()] if arg else None


def cmd_serve(reg: Registry, a) -> int:
    from drtc.evals import serving

    return serving.serve(reg, a.eval, _models(a.models), a.device)


def cmd_servers(reg: Registry, a) -> int:
    """Reach the eval's servers on one model server and ask each model to describe itself."""
    from drtc.evals import serving

    sp = S.load(reg, a.eval)
    if not S.is_frozen(sp):
        raise RegistryError(f"{a.eval} is not frozen (lab eval freeze {a.eval})")
    conn = serving.connect(reg, sp, a.server, _models(a.models))
    bad = 0
    try:
        print(f"{a.eval} on {a.server}:")
        for m, ep in conn.endpoints.items():
            d = serving.probe(ep, serving.deployment(sp, m))
            if "error" in d:
                bad += 1
                print(f"  {m:32s} NOT REACHABLE: {d['error']}")
            else:
                state = "ready" if d.get("ready") else "LOADING"
                print(f"  {m:32s} {state:7s} commit {sp.frozen[m][:8]}  checkpoint {d.get('artifact_identity', '?')[:19]}"
                      f"  {ep}")
    finally:
        if not a.keep:
            conn.close()
    if bad:
        print(f"{bad} not reachable: is `lab eval serve {a.eval}` running on {a.server}?")
    return 1 if bad else 0


def cmd_run(reg: Registry, a) -> int:
    from drtc.evals.run import session_run

    return session_run(reg, a.eval, a.server, _models(a.models), a.test, a.offline, a.max_trials)


def cmd_void(reg: Registry, a) -> int:
    from drtc.evals.run import void_trial

    sp = S.load(reg, a.eval)
    number = int(a.trial.lstrip("tT"))
    hits = [t for t in store.trials(a.eval, S.results_root(sp))
            if t.model == a.model and t.number == number and (a.condition is None or t.condition == a.condition)]
    if len(hits) != 1:
        raise RegistryError(f"{len(hits)} trials match {a.model} t{number:02d}"
                            + ("; add --condition" if len(hits) > 1 else ""))
    t = hits[0]
    if t.void:
        print(f"{t.path} is already void ({t.info['void']})")
        return 0
    void_trial(t.path, a.reason)
    print(f"voided {t.path}: {a.reason}\nit is run again in a later round; push the session: lab eval push {a.eval}")
    return 0


def cmd_push(reg: Registry, a) -> int:
    from drtc.evals import sync

    sp = S.load(reg, a.eval)
    kind = "bench" if sp.robot == "replay" else "evals"
    waiting = store.unpushed(a.eval, S.results_root(sp))
    if not waiting:
        print(f"{a.eval}: every session on this machine is pushed")
    for s in waiting:
        oid = sync.push(reg, s, kind)
        print(f"pushed {s.name} ({oid[:10]})")
    return 0


def cmd_pull(reg: Registry, a) -> int:
    from drtc.evals import sync

    sp = S.load(reg, a.eval)
    new = sync.pull(reg, a.eval, "bench" if sp.robot == "replay" else "evals")
    print(f"{a.eval}: {len(new)} new sessions" + (f": {', '.join(new)}" if new else ""))
    return 0


def cmd_report(reg: Registry, a) -> int:
    from drtc.evals import report

    if a.pull:
        cmd_pull(reg, a)
    return report.main(reg, a.eval)


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
    p = sub.add_parser("serve", help="on a model server: serve every model of a frozen eval (Ctrl-C stops them)")
    p.add_argument("eval")
    p.add_argument("--models", help="only these models (comma-separated), e.g. when the GPU is too small for all")
    p.add_argument("--device", default="cuda")
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("servers", help="on the robot computer: check that the eval's models answer on a server")
    p.add_argument("eval")
    p.add_argument("--server", required=True, choices=["laptop", "a100", "h200"])
    p.add_argument("--models", help="only these models (comma-separated)")
    p.add_argument("--keep", action="store_true", help="keep the SSH tunnel open afterwards")
    p.set_defaults(fn=cmd_servers)
    p = sub.add_parser("run", help="on the robot computer: a session of official trials (s/f keys; lab/EVALUATION.md)")
    p.add_argument("eval")
    p.add_argument("--server", required=True, choices=["laptop", "a100", "h200"], help="where the models are served")
    p.add_argument("--models", help="only these models (comma-separated), e.g. the ones a small GPU can serve")
    p.add_argument("--test", action="store_true", help="a test session in results/scratch (not official, not pushed)")
    p.add_argument("--offline", action="store_true", help="do not pull/push (push later with lab eval push)")
    p.add_argument("--max-trials", type=int, help="end the session after this many trials")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("void", help="void a trial for a setup mistake (kept, not counted, run again)")
    p.add_argument("eval")
    p.add_argument("model")
    p.add_argument("trial", help="its number, e.g. t03")
    p.add_argument("--reason", required=True)
    p.add_argument("--condition")
    p.set_defaults(fn=cmd_void)
    for name, fn, help_ in (("push", cmd_push, "upload this machine's sessions that are not pushed yet"),
                            ("pull", cmd_pull, "download the sessions other machines pushed")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("eval")
        p.set_defaults(fn=fn)
    p = sub.add_parser("report", help="rebuild trials.csv, metrics.csv and report.md from every session here")
    p.add_argument("eval")
    p.add_argument("--pull", action="store_true", help="first pull the sessions other machines pushed")
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    reg = Registry()
    try:
        return a.fn(reg, a)
    except RegistryError as e:
        print(f"lab eval: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
