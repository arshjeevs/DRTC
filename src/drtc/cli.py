"""The `lab` command. Subcommands are added as Phase 1.7 proceeds (lab/PLAN.md)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from drtc.registry import Registry, RegistryError, for_host, resolve_device


def _registry() -> Registry:
    try:
        return Registry()
    except RegistryError as e:
        sys.exit(f"registry: {e}")


def cmd_list(args) -> int:
    reg = _registry()
    print(f"models ({reg.lab}/models):")
    for mid in reg.models:
        r = reg.resolve(mid)
        if args.status and r.status not in args.status:
            continue
        ds = r.dataset.id if r.dataset else "-"
        rec = r.recipe.id if r.recipe else "-"
        print(f"  {mid:34s} {r.status:13s} {r.policy:10s} {rec:22s} {ds}")
    if not args.models_only:
        print("datasets:")
        for d in reg.datasets.values():
            parents = f" <- {', '.join(d.parents)}" if d.parents else ""
            print(f"  {d.id:34s} rig {d.rig:9s} {d.episodes or '?':>4} episodes{parents}")
        print("rigs:")
        for r in reg.rigs.values():
            live = "current" if r.known and r.until is None else f"until {r.until}" if r.until else ""
            print(f"  {r.id:34s} {live}")
        print("hosts:")
        for h in reg.hosts.values():
            gpus = ", ".join(f"{g.count}x {g.name}" for g in h.gpus) or "no GPU"
            print(f"  {h.id:34s} {'/'.join(h.roles):32s} {gpus}")
    return 0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def check_rig(reg: Registry) -> list[str]:
    """Devices and calibration of the current rig, as seen from this machine."""
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

    problems, host = [], reg.current_host()
    hid = host.id if host else None
    rig = reg.current_rig()
    print(f"rig {rig.id} on host {hid or '(unknown: set LAB_HOST)'}")
    for name, arm in (("robot", rig.robot), ("leader", rig.leader)):
        if arm is None:
            continue
        port = resolve_device(for_host(arm.port, hid))
        ok = Path(str(port)).exists() if str(port).startswith("/dev/") else True
        print(f"  {name:7s} port {port} {'ok' if ok else 'MISSING'}")
        if not ok:
            problems.append(f"{name} port {port} not found")
        cal = HF_LEROBOT_CALIBRATION / arm.calibration
        if not cal.is_file():
            problems.append(f"{name} calibration {cal} not found")
            continue
        motors = list(json.loads(cal.read_text()))
        print(f"  {name:7s} calibration {cal} sha {_sha(cal)} ({len(motors)} motors)")
        # LeRobot 0.7's default location for this arm (robots/so_follower/<id>.json): warn if it differs,
        # since any tool that does not pass calibration_dir would load that file instead.
        kind = "robots" if name == "robot" else "teleoperators"
        default = HF_LEROBOT_CALIBRATION / kind / ("so_follower" if name == "robot" else "so_leader") / f"{arm.id}.json"
        if default.is_file() and _sha(default) != _sha(cal):
            print(f"  {name:7s} WARNING {default} differs (sha {_sha(default)}): LeRobot's default location holds "
                  f"another calibration; lab always passes {cal.parent}")
    for key, cam in rig.cameras.items():
        path = resolve_device(for_host(cam.path, hid))
        ok = Path(str(path)).exists() if str(path).startswith("/dev/") else True
        print(f"  {key:7s} {cam.role:6s} {path} {'ok' if ok else 'MISSING'}")
        if not ok:
            problems.append(f"{key} ({cam.role}) camera {path} not found")
    return problems


def check_data(reg: Registry, online: bool = False) -> list[str]:
    """Each dataset's local copy (and with `online` its Hub version) against the registry's pin.

    A local copy under a reused name (e.g. the untrimmed v1 cached on a server before 2026-11-18) has the wrong
    episode or frame count and is reported, so no job trains on it.
    """
    from lerobot.utils.constants import HF_LEROBOT_HOME

    problems = []
    api = None
    if online:
        from huggingface_hub import HfApi

        api = HfApi()
    for d in reg.datasets.values():
        info = HF_LEROBOT_HOME / d.repo / "meta" / "info.json"
        if info.is_file():
            meta = json.loads(info.read_text())
            got = (meta["total_episodes"], meta["total_frames"])
            want = (d.episodes, d.frames)
            ok = all(w is None or w == g for w, g in zip(want, got))
            print(f"  {d.id:28s} local {got[0]} episodes {got[1]} frames {'ok' if ok else f'MISMATCH (registry {want[0]} / {want[1]})'}")
            if not ok:
                problems.append(f"dataset {d.id}: local copy {info.parent.parent} is not the registered version "
                                f"({got[0]} episodes / {got[1]} frames, registry {want[0]} / {want[1]}); delete it")
        else:
            print(f"  {d.id:28s} no local copy")
        if api is not None and d.revision:
            tags = {t.name: t.target_commit for t in api.list_repo_refs(d.repo, repo_type="dataset").tags}
            hub = tags.get("v3.0")
            print(f"  {'':28s} Hub v3.0 {hub[:10] if hub else 'missing'} {'ok' if hub == d.revision else 'MISMATCH'}")
            if hub != d.revision:
                problems.append(f"dataset {d.id}: Hub tag v3.0 is {hub}, registry pins {d.revision}")
    return problems


def cmd_check(args) -> int:
    reg = _registry()
    problems = reg.check()
    print(f"registry: {len(reg.models)} models, {len(reg.recipes)} recipes, {len(reg.datasets)} datasets: "
          f"{'ok' if not problems else f'{len(problems)} problems'}")
    if args.rig:
        problems += check_rig(reg)
    if args.data:
        problems += check_data(reg, online=args.online)
    for p in problems:
        print(f"  PROBLEM {p}")
    return 1 if problems else 0


def cmd_stop(args) -> int:
    """Ask a running `lab train <model>` to save a checkpoint, push it and exit (it checks every step)."""
    from drtc.train import OUTPUTS, stop_file

    models = args.models
    if args.all:
        models = [p.name for p in OUTPUTS.iterdir() if p.is_dir() and not p.name.startswith(".")] \
            if OUTPUTS.is_dir() else []
    for m in models:
        f = stop_file(m)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch()
        print(f"  {m}: stop requested ({f}); a running `lab train {m}` saves, pushes and exits at its next step")
    return 0


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "train":  # its own argument parser; unknown flags go to lerobot-train
        from drtc.train import main as train_main

        sys.exit(train_main(argv[1:]))
    if argv and argv[0] in ("submit", "worker", "jobs"):
        from drtc import jobs

        sys.exit({"submit": jobs.cmd_submit, "worker": jobs.cmd_worker, "jobs": jobs.cmd_jobs}[argv[0]](argv[1:]))
    ap = argparse.ArgumentParser(prog="lab", description="Lab tools (lab/PLAN.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", help="models, datasets, rigs, hosts")
    p.add_argument("--status", nargs="*", help="only models with these statuses (trained, planned, ...)")
    p.add_argument("--models-only", action="store_true")
    p.set_defaults(fn=cmd_list)
    p = sub.add_parser("check", help="registry consistency; --rig devices and calibration; --data episode counts")
    p.add_argument("--rig", action="store_true")
    p.add_argument("--data", action="store_true")
    p.add_argument("--online", action="store_true", help="with --data: also check the Hub revision")
    p.set_defaults(fn=cmd_check)
    sub.add_parser("train", help="train a registry model (lab train <model> -h)")
    sub.add_parser("submit", help="queue a training job here or on a host (lab submit <model> --host h200)")
    sub.add_parser("worker", help="run queued jobs on this machine")
    sub.add_parser("jobs", help="list this machine's jobs")
    p = sub.add_parser("stop", help="stop running trainings cleanly (save, push, exit)")
    p.add_argument("models", nargs="*")
    p.add_argument("--all", action="store_true")
    p.set_defaults(fn=cmd_stop)
    args = ap.parse_args(argv)
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
