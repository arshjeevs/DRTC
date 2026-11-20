"""`lab eval run <eval> --server <laptop|a100|h200>`: one session of official trials (lab/EVALUATION.md).

1. Checks: the eval is frozen and valid, the git tree is clean, the plan is the one earlier sessions used, the rig
   check passes (real robot), and every served model answers with its frozen version.
2. Pulls the eval's sessions from the results store (unless offline), so the trials done on any machine count.
3. Runs rounds (lab/METRICS.md rule 2): the operator sets the round's layout, then each trial starts with Enter. During
   a trial: `s` = success now, `f` = failure now; the time limit is a failure. After a trial: Enter = next, `v` = void
   it (setup mistake, with a reason), `q` = end the session.
4. Writes results/evals/<eval>/sessions/<YYYYMMDD-HHMM>-<machine>/, and pushes the session at the end.

`--test` makes the same session in results/scratch/<date>/ (not official: no checks on git, never pulled or pushed).
`--replay` (a setup check, no robot needed): one round on the replay robot against the eval's own servers, as a test.
On the replay robot (`robot: replay` in the eval, the Phase 7-8 bench) there are no keys: every completed run is a
valid trial (M2-M8 only), stored under results/bench/.
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import socket
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path

import yaml

from drtc.evals import serving, store, sync
from drtc.evals import spec as S
from drtc.registry import Registry, RegistryError

METRICS_VERSION = "v1"


# --- keys during a trial ----------------------------------------------------------------------------------------------


class TrialKeys:
    """Reads single keys (no Enter) while a trial runs; `s` / `f` end it through the rollout's shutdown event."""

    def __init__(self):
        self.key: str | None = None
        self.at: float | None = None
        self.event = None  # the rollout's shutdown event (set once the rollout has built it)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._old = None

    def __enter__(self):
        if sys.stdin.isatty():
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                continue
            ch = sys.stdin.read(1).lower()
            if ch in ("s", "f") and self.key is None:
                self.key, self.at = ch, time.monotonic()
                print(f"\n  [{ 'success' if ch == 's' else 'failure'}] ending the trial", flush=True)
                if self.event is not None:
                    self.event.set()

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self._old is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)


# --- helpers ----------------------------------------------------------------------------------------------------------


def git_clean() -> tuple[str, bool]:
    from drtc.control.run import git_state

    commit, dirty = git_state()
    return commit, not dirty


def plan_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def machine(reg: Registry) -> str:
    host = reg.current_host()
    return host.id if host else socket.gethostname()


def camera_roles(reg: Registry, model: str, robot: str) -> dict[str, str]:
    r = reg.resolve(model)
    rig = reg.rigs[r.dataset.rig] if robot == "replay" else reg.current_rig()
    return {key: (cam.role or key) for key, cam in rig.cameras.items()}


def held_out_episodes(reg: Registry, model: str) -> list[int]:
    from drtc.train import select_episodes

    held = select_episodes(dict(reg.resolve(model).train))[1]
    return held or [0]


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return "q"


# --- one trial --------------------------------------------------------------------------------------------------------


def run_trial(reg: Registry, sp: S.EvalSpec, condition: str, model: str, number: int, rnd: int, session: Path,
              endpoint: str, ctx: dict) -> dict:
    from drtc.control import engine as engine_mod
    from drtc.control.run import build, rollout_once
    from drtc.evals.video import TrialVideo

    robot = sp.robot
    tdir = store.trial_dir(session, condition, model, number)
    tdir.mkdir(parents=True, exist_ok=True)
    held = held_out_episodes(reg, model)
    episode = held[(number - 1) % len(held)] if robot == "replay" else 0
    cmd, composed, notes = build(reg, sp.conditions[condition], model, robot, episode, endpoint, sp.time_limit_s,
                                 tdir, serving.deployment(sp, model))
    (tdir / "config.yaml").write_text(yaml.safe_dump(composed, sort_keys=False))
    video = TrialVideo(tdir / "video", camera_roles(reg, model, robot)) if sp.video else None
    holder: dict = {}

    def on_engine(engine):
        holder["engine"] = engine
        if video is not None:
            engine.observers.append(video.observe)

    keys = TrialKeys()
    engine_mod.ON_ENGINE.append(on_engine)
    started = datetime.now()
    try:
        if video is not None:
            video.start()
        with keys if robot == "rig" else _Null():
            outcome, error, took = rollout_once(cmd, tdir, on_context=lambda ev: setattr(keys, "event", ev))
    finally:
        engine_mod.ON_ENGINE.remove(on_engine)
        video_info = video.stop() if video is not None else None
    eng = holder.get("engine")
    first = None
    if eng is not None:
        first = next((row[0] for row in eng._ticks if row[1]), None)
    key_t = None if (keys.at is None or eng is None) else round(keys.at - eng._t0, 3)
    if robot == "replay":
        success, label = None, "replay"
    elif keys.key == "s":
        success, label = 1, "s"
    elif keys.key == "f":
        success, label = 0, "f"
    elif outcome == "completed" and took >= sp.time_limit_s - 1.0:
        success, label = 0, "time_limit"
    else:
        success, label = None, "aborted"  # a fault, an error or Ctrl-C: not a valid trial, run again later
    info = {
        "eval": ctx["eval"], "session": session.name, "condition": condition, "model": model, "trial": number,
        "round": rnd, "success": success, "label": label, "key_time_s": key_t,
        "first_action_s": first, "time_to_success_s": (round(key_t - first, 3) if success == 1 and key_t is not None
                                                        and first is not None else None),
        "time_limit_s": sp.time_limit_s, "outcome": outcome, "error": error,
        "started_at": started.isoformat(timespec="seconds"), "duration_s": round(took, 2),
        "robot": robot, "episode": episode if robot == "replay" else None, "machine": ctx["machine"],
        "server": ctx["server"], "server_gpu": ctx["server_gpu"], "deployment": serving.deployment(sp, model),
        "revision": sp.frozen[model], "artifact_identity": ctx["artifacts"].get(model),
        "git_commit": ctx["git_commit"], "metrics_version": METRICS_VERSION, "video": video_info,
        "not_applied": notes, "command": cmd, "void": None,
    }
    (tdir / "trial.json").write_text(json.dumps(info, indent=2))
    return info


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def void_trial(tdir: Path, reason: str) -> None:
    info = json.loads((tdir / "trial.json").read_text())
    info["void"] = reason
    info["voided_at"] = datetime.now().isoformat(timespec="seconds")
    (tdir / "trial.json").write_text(json.dumps(info, indent=2))
    session = tdir.parents[3]
    (session / store.PUSHED).unlink(missing_ok=True)  # changed since its push: push it again


# --- a session --------------------------------------------------------------------------------------------------------


def session_run(reg: Registry, name: str, server: str, models: list[str] | None, test: bool, offline: bool,
                max_trials: int | None, replay: bool = False) -> int:
    sp = S.load(reg, name)
    if replay:  # a setup check without the robot: one round on the replay robot, against this eval's own servers
        test = True
        sp = sp.model_copy(update={"robot": "replay", "trials": 1, "time_limit_s": min(sp.time_limit_s, 10.0)})
    problems = S.check(reg, sp)
    if problems:
        raise RegistryError(f"{name}: " + "; ".join(problems))
    if not S.is_frozen(sp):
        raise RegistryError(f"{name} is not frozen: lab eval freeze {name}, then commit and push")
    kind = "bench" if sp.robot == "replay" else "evals"
    # A test session starts fresh (its own folder), so earlier tests never count towards it.
    root = (Path("results") / "scratch" / f"{datetime.now():%Y%m%d}" / f"{kind}-test-{datetime.now():%H%M%S}") \
        if test else (Path("results") / kind)
    commit, clean = git_clean()
    if not test and not clean:
        raise RegistryError("the git tree has uncommitted changes: commit them first (official trials record the "
                            "exact code), or use --test")
    psha = plan_sha(S.eval_path(reg, name))
    if not test and not offline:
        try:
            new = sync.pull(reg, name, kind)
            if new:
                print(f"pulled {len(new)} sessions from the results store: {', '.join(new)}")
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: cannot reach the results store ({type(e).__name__}); continuing with this machine's "
                  f"sessions only. Use --offline to skip this check.")
    for s in store.sessions(name, root):
        try:
            other = json.loads((s / "session.json").read_text()).get("plan_sha")
        except (OSError, json.JSONDecodeError):
            continue
        if other and other != psha and not test:
            raise RegistryError(f"lab/evals/{name}.yaml changed since session {s.name} (its plan differs). A plan "
                                f"with trials is never changed: put the change in a new eval")
    if sp.robot == "replay":
        try:
            import lerobot_robot_replay  # noqa: F401
        except ImportError:
            raise RegistryError("the replay robot is not installed here: uv sync --frozen --extra gpu --extra sim "
                                "(add --python /usr/bin/python3 on the H200)") from None
    if sp.robot == "rig":
        from drtc.cli import check_rig

        bad = check_rig(reg)
        if bad:
            raise RegistryError("rig check failed: " + "; ".join(bad))
    chosen = models or sp.models
    if sp.robot == "replay":  # decode every held-out episode once, before the first trial (30 Hz on any CPU)
        import time as _time

        from lerobot_robot_replay.replay import preload

        t0 = _time.monotonic()
        todo = sorted({(reg.resolve(m).dataset.repo, reg.resolve(m).dataset.revision, e)
                       for m in chosen for e in held_out_episodes(reg, m)})
        print(f"decoding {len(todo)} held-out episodes for the replay robot ...", flush=True)
        for repo, rev, e in todo:
            preload(repo, rev, e)
        print(f"  done in {_time.monotonic() - t0:.0f} s", flush=True)
    conn = serving.connect(reg, sp, server, chosen)
    try:
        artifacts, missing = {}, []
        for m in chosen:
            d = serving.probe(conn.endpoints[m], serving.deployment(sp, m))
            if "error" in d or not d.get("ready"):
                missing.append(f"{m} ({d.get('error', 'not ready')})")
            else:
                artifacts[m] = d.get("artifact_identity")
        if missing:
            raise RegistryError(f"not served on {server}: {'; '.join(missing)}. Start `lab eval serve {name}` there"
                                + (" or pass --models with the ones that are" if not models else ""))
        host = reg.current_host() if server == "local" else reg.hosts.get(server)
        gpu = host.gpus[0].name if host and host.gpus else None
        if server == "local" and host is not None:
            server = f"local ({host.id})"
        mach = machine(reg)
        session = store.sessions_dir(name, root) / f"{datetime.now():%Y%m%d-%H%M}-{mach}"
        session.mkdir(parents=True, exist_ok=True)
        ctx = {"eval": name, "machine": mach, "server": server, "server_gpu": gpu, "artifacts": artifacts,
               "git_commit": commit}
        (session / "session.json").write_text(json.dumps({
            "eval": name, "session": session.name, "machine": mach, "robot": sp.robot, "server": server,
            "server_gpu": gpu, "git_commit": commit, "plan_sha": psha, "metrics_version": METRICS_VERSION,
            "models": {m: {"revision": sp.frozen[m], "deployment": serving.deployment(sp, m),
                           "artifact_identity": artifacts[m]} for m in chosen},
            "test": test, "started_at": datetime.now().isoformat(timespec="seconds"),
        }, indent=2))
        print(f"session {session} ({'TEST, not official' if test else 'official'}); robot {sp.robot}, server "
              f"{server} ({gpu})")
        done_here = 0
        quit_ = False
        failures: dict[tuple[str, str], int] = {}  # consecutive not-counted trials per pair in this session
        skipped: set[tuple[str, str]] = set()
        while not quit_:
            counts = store.valid_counts(name, root)
            rnd, pairs = S.next_round(sp, counts)
            pairs = [p for p in pairs if p[1] in chosen and p not in skipped]
            if not rnd:
                print("the eval is complete")
                break
            if not pairs:
                print("every model of this session is done for now" + (" (the others: another session)" if models or skipped
                                                                       else ""))
                break
            print(f"\n=== Round {rnd}: {len(pairs)} trials ===")
            if sp.robot == "rig":
                print("Set this round's layout (the same for every trial of the round).")
            for i, (cond, m) in enumerate(pairs, 1):
                number = 1 + sum(1 for t in store.trials(name, root) if t.condition == cond and t.model == m)
                label = f"{m}" + (f" [{cond}]" if len(sp.conditions) > 1 else "")
                if sp.robot == "rig":
                    ans = ask(f"\nRound {rnd}, trial {i}/{len(pairs)}: {label}, t{number:02d}. Put the cube(s) on the "
                              f"marks, press Enter to start (s = success, f = failure during the trial; q = quit): ")
                    if ans == "q":
                        quit_ = True
                        break
                else:
                    print(f"round {rnd}, {i}/{len(pairs)}: {label} t{number:02d}", flush=True)
                info = run_trial(reg, sp, cond, m, number, rnd, session, conn.endpoints[m], ctx)
                done_here += 1
                tdir = store.trial_dir(session, cond, m, number)
                counted = (info["outcome"] == "completed" and info.get("first_action_s") is not None) \
                    if sp.robot == "replay" else info["success"] in (0, 1)
                failures[(cond, m)] = 0 if counted else failures.get((cond, m), 0) + 1
                if sp.robot == "replay":
                    verdict = "done" if info["outcome"] == "completed" else "not counted (run again later)"
                else:
                    verdict = {1: "SUCCESS", 0: "FAILURE"}.get(info["success"], "not counted (run again later)")
                extra = f" in {info['time_to_success_s']} s" if info.get("time_to_success_s") is not None else ""
                print(f"  t{number:02d}: {verdict}{extra} ({info['label']}; {info['outcome']}"
                      + (f": {info['error']}" if info.get("error") else "") + ")")
                if sp.robot == "rig":
                    ans = ask("  Enter = next trial, v = void this trial (setup mistake), q = end the session: ")
                    if ans == "v":
                        reason = ""
                        while not reason:
                            reason = input("  reason (e.g. cube not on its mark): ").strip()
                        void_trial(tdir, reason)
                        print("  voided: it will be run again in a later round")
                    if ans == "q":
                        quit_ = True
                        break
                if failures.get((cond, m), 0) >= 2:
                    skipped.add((cond, m))
                    print(f"  {label}: 2 trials in a row not counted ({info['outcome']}): skipped for the rest of this "
                          f"session. Check its server log (logs/serve/{name}/{m}.log) and the trial's rollout.log")
                if max_trials and done_here >= max_trials:
                    quit_ = True
                    break
    finally:
        conn.close()
    print(f"\nsession {session.name}: {done_here} trials")
    if not test and not offline and done_here:
        try:
            oid = sync.push(reg, session, kind)
            print(f"pushed to {sync.repo(reg)} ({oid[:10]})")
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: push failed ({type(e).__name__}: {e}); the session stays here: lab eval push {name}")
    elif not test and done_here:
        print(f"not pushed (offline): lab eval push {name} when online")
    return 0
