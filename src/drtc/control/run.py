"""Run an experiment condition: `lab run <experiment> [--model M] [--robot replay] [--trials N]`.

The experiment (lab/experiments/<group>/<name>.yaml, composed _defaults < rig < model < experiment exactly as in the
minor) becomes one lerobot-rollout run with the drtc engine (drtc.control.engine) against a policy server started
with `lab serve <model>`. These are test runs: results/scratch/<date>/<experiment>/<model>/<run_id>/ with config.yaml,
ticks.csv (one row per control tick), run.json; one row per run in results/scratch/index.csv. Official trials are
made only by `lab eval run` (lab/EVALUATION.md); reports never read scratch.

Robot: the current rig (lab/rigs), in the model's dataset joint units, or `--robot replay`: the replay robot
playing an episode of the model's dataset (no hardware; timing, scheduling and faults, not task success).
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from drtc.registry import Registry, RegistryError

RESULTS = Path("results") / "scratch"  # test runs only (lab/EVALUATION.md rule 6)
INDEX_FIELDS = ["run_id", "experiment", "model", "trial", "success", "started_at", "duration_s", "git_commit",
                "git_dirty", "note", "run_dir", "robot", "host", "outcome"]
# Composed experiment keys this run applies; anything else that is set is reported as not applied yet.
APPLIED = {"scheduler", "blend_steps", "blend_weight", "cooldown", "cooldown_cap", "epsilon", "playback_mode", "playback_min_rate", "playback_margin_steps",
           "playback_rate_slew", "action_filter_mode", "action_filter_butterworth_cutoff", "action_filter_gain", *("drop_obs", "drop_action", "dup_obs", "dup_action", "reorder_obs", "reorder_action", "disconnect",
              "spikes", "latency"), "estimator", "s_min", "latency_k", "latency_alpha", "latency_beta", "duration_s", "fps", "task",
           "policy_type", "pretrained_name_or_path", "finetune", "model", "name", "description", "train",
           "rtc_enabled", "num_flow_matching_steps", "actions_per_chunk"}
APPLIED_PREFIXES = ("robot_", "camera", "client_host", "server_host", "gpu")
# Settings that only matter for the features still to come (D5 faults, D6 smoothing) or the old client.
PENDING = {"smoothness_guidance": "smoothing C (not ported yet)", "rtc_sigma_d": "RTC settings are the server's",
           "rtc_prefix_attention_schedule": "RTC settings are the server's (upstream defaults)", 
            
           "full_diagnostics": "old client diagnostics"}
FAULT_KEYS = ("drop_obs", "drop_action", "dup_obs", "dup_action", "reorder_obs", "reorder_action", "disconnect",
              "spikes", "latency")


def append_index(rows: list[dict], index: Path | None = None) -> None:
    """Append rows to results/index.csv, first upgrading an older file to the current columns."""
    index = index or RESULTS / "index.csv"
    index.parent.mkdir(parents=True, exist_ok=True)
    if index.is_file():
        with open(index, newline="") as f:
            reader = csv.DictReader(f)
            header, old = reader.fieldnames or [], list(reader)
        if header != INDEX_FIELDS:
            with open(index, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
                w.writeheader()
                w.writerows({k: r.get(k, "") for k in INDEX_FIELDS} for r in old)
    new = not index.is_file()
    with open(index, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if new:
            w.writeheader()
        w.writerows({k: r.get(k, "") for k in INDEX_FIELDS} for r in rows)


def git_state() -> tuple[str, bool]:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                text=True).stdout.strip())
    return commit, dirty


def robot_flags(reg: Registry, model_id: str, robot: str, episode: int, log_path: Path) -> dict:
    r = reg.resolve(model_id)
    if robot == "replay":
        return {"robot.type": "replay", "robot.repo_id": r.dataset.repo, "robot.revision": r.dataset.revision,
                "robot.episode": episode, "robot.log_path": str(log_path)}
    from drtc.data.record import rig_flags

    flags = {k: v for k, v in rig_flags(reg).items() if k.startswith("robot.")}
    flags["robot.use_degrees"] = bool(r.dataset.use_degrees)  # the model's units, not the rig's default
    return flags


def rtc_requested(reg: Registry, experiment: str) -> bool:
    """Guided RTC is used only when the experiment file itself sets `rtc_enabled: true` (2026-11-20).

    Recipes and models also carry an `rtc_enabled` (the minor's serve hints, e.g. SmolVLA's), which never switched
    anything on before RTC was wired; honouring them would quietly turn running baselines into RTC runs.
    """
    from drtc.registry.registry import load_yaml

    return load_yaml(reg.experiment_path(experiment)).get("rtc_enabled") is True


def build(reg: Registry, experiment: str, model: str | None, robot: str, episode: int, endpoint: str,
          duration: float | None, run_dir: Path, deployment: str | None = None) -> tuple[list[str], dict, list[str]]:
    from drtc.control.serve import camera_renames, deployment_name, semantics_for

    composed = reg.compose(experiment, model_override=model)
    model_id = composed["model"]
    notes = []
    # Only report features that are switched on (their parameters alone change nothing).
    off = set()
    if composed.get("playback_mode", "fixed") == "fixed":
        off |= {k for k in PENDING if k.startswith("playback_")}
    if not composed.get("rtc_enabled"):
        off |= {"rtc_sigma_d", "rtc_prefix_attention_schedule"}
    if composed.get("action_filter_mode", "none") in (None, "none"):
        off |= {k for k in PENDING if k.startswith("action_filter_")}
    else:
        off |= {"action_filter_butterworth_cutoff", "action_filter_gain"}
    for k, v in composed.items():
        if k in FAULT_KEYS:
            continue
        elif k in PENDING and k not in off and v not in (None, False, 0, 0.0, "fixed", "none"):
            notes.append(f"{k}={v}: {PENDING[k]} (not applied yet)")
    sem = semantics_for(reg, model_id)
    flags = {
        **robot_flags(reg, model_id, robot, episode, run_dir / "robot_commands.jsonl"),
        "inference.type": "drtc",
        "inference.endpoint": endpoint,
        "inference.deployment": deployment or deployment_name(model_id),
        "inference.semantics": sem["joints"],
        "inference.mode": "rtc_guided" if rtc_requested(reg, experiment) else "chunk",
        # aligned for plain chunks; upstream allows RTC only with append (`auto` picks it)
        "inference.chunk_merge": "auto" if rtc_requested(reg, experiment) else "aligned",
        # upstream's blending at the chunk switch (a bench baseline): the first blend_steps queued actions are mixed
        # with the new chunk's, (1 - w) * old + w * new; 0 = off (the default)
        "inference.blend_steps": int(composed.get("blend_steps", 0)),
        "inference.blend_weight": float(composed.get("blend_weight", 0.5)),
        # drtc: DRTC's trigger; upstream: LeRobot's own (refill when the queued playback reaches the turnaround)
        "inference.scheduler": str(composed.get("scheduler", "drtc")),
        "inference.estimator": composed.get("estimator", "jk"),
        "inference.s_min": int(composed.get("s_min", 20)),
        "inference.latency_k": float(composed.get("latency_k", 1.5)),
        "inference.latency_alpha": float(composed.get("latency_alpha", 0.125)),
        "inference.latency_beta": float(composed.get("latency_beta", 0.25)),
        "inference.metrics_path": str(run_dir / "ticks.csv"),
        "inference.cooldown": bool(composed.get("cooldown", True)),
        "inference.action_filter": "butterworth" if composed.get("action_filter_mode") == "butterworth" else "none",
        "inference.filter_cutoff_hz": float(composed.get("action_filter_butterworth_cutoff", 3.0)),
        "inference.filter_gain": float(composed.get("action_filter_gain", 1.4)),
        "inference.epsilon": int(composed.get("epsilon", 1)),
        "inference.playback_mode": str(composed.get("playback_mode", "fixed")),
        "inference.playback_min_rate": float(composed.get("playback_min_rate", 0.5)),
        "inference.playback_margin_steps": float(composed.get("playback_margin_steps", 4.0)),
        "inference.playback_rate_slew": float(composed.get("playback_rate_slew", 0.05)),
        "inference.cooldown_cap": str(composed.get("cooldown_cap", "none")),
        # DRTC holds the arm while starved and goes on when actions arrive (stalls are measured: M2, M3); upstream
        # ends the run with a fault after 1 s without actions, which would turn every stall into an aborted trial.
        "inference.action_starvation_grace_s": float(duration if duration is not None else composed.get("duration_s",
                                                                                                         25.0)),
        **({"inference.faults_json": json.dumps({k: composed[k] for k in FAULT_KEYS if composed.get(k)})}
           if any(composed.get(k) for k in FAULT_KEYS) else {}),
        "fps": float(composed.get("fps", 30)),
        "task": composed["task"],
        "duration": float(duration if duration is not None else composed.get("duration_s", 25.0)),
        "play_sounds": False,
    }
    if flags["inference.blend_steps"]:
        # upstream needs the components to blend named explicitly: every action joint of the model's dataset
        from lerobot.datasets import LeRobotDatasetMetadata

        r = reg.resolve(model_id)
        names = list(LeRobotDatasetMetadata(r.dataset.repo, revision=r.dataset.revision).features["action"]["names"])
        flags["inference.blend_components"] = json.dumps(names)
    renames = camera_renames(reg, model_id)
    if renames:  # robot camera keys -> the checkpoint's names, which the server declares (drtc.control.serve)
        flags["rename_map"] = json.dumps(renames)
    cmd = ["lerobot-rollout", *(f"--{k}={v}" for k, v in flags.items())]
    return cmd, composed, notes


def rollout_once(cmd: list[str], run_dir: Path, on_context=None) -> tuple[str, str | None, float]:
    """Run one lerobot-rollout in this process with the drtc engine; its log goes to run_dir/rollout.log.

    on_context(shutdown_event) is called once the rollout has its shutdown event: setting that event ends the
    rollout early and cleanly (teardown, metrics written), as Ctrl-C does. Returns (outcome, error, seconds).
    """
    import logging

    from lerobot.scripts import lerobot_rollout
    from lerobot.utils.import_utils import register_third_party_plugins

    from drtc.control.engine import install

    register_third_party_plugins()
    install()
    handler = logging.FileHandler(run_dir / "rollout.log")
    handler.setLevel(logging.INFO)  # not the libraries' DEBUG chatter
    handler.setFormatter(logging.Formatter("%(levelname)s %(asctime)s %(name)s %(message)s"))
    # lerobot-rollout reconfigures logging when it starts (init_logging), which drops handlers added before;
    # attach ours right after it.
    _init_logging = lerobot_rollout.init_logging
    _build = lerobot_rollout.build_rollout_context

    def init_logging(*args, **kwargs):
        _init_logging(*args, **kwargs)
        logging.getLogger().addHandler(handler)

    def build_rollout_context(cfg, shutdown_event, *args, **kwargs):
        if on_context is not None:
            on_context(shutdown_event)
        return _build(cfg, shutdown_event, *args, **kwargs)

    lerobot_rollout.init_logging = init_logging
    lerobot_rollout.build_rollout_context = build_rollout_context
    t0 = time.monotonic()
    argv, sys.argv = sys.argv, cmd
    outcome, error = "completed", None
    try:
        code = lerobot_rollout.main()
        if code not in (None, 0):
            outcome = f"exit {code}"
    except SystemExit as e:  # a terminal fault ends the rollout through sys.exit
        if e.code not in (None, 0):
            outcome = f"exit {e.code}"
    except Exception as e:  # noqa: BLE001 - a failed run is still a result
        outcome, error = "error", f"{type(e).__name__}: {e}"
    finally:
        took = time.monotonic() - t0
        sys.argv = argv
        logging.getLogger().removeHandler(handler)
        handler.close()
        lerobot_rollout.init_logging = _init_logging
        lerobot_rollout.build_rollout_context = _build
    if outcome != "completed":
        fault = [line for line in (run_dir / "rollout.log").read_text(errors="replace").splitlines()
                 if "terminal fault" in line or line.startswith("ERROR")]
        error = error or (fault[-1][-300:] if fault else None)
    return outcome, error, took


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab run", description=__doc__.split("\n\n")[0])
    ap.add_argument("experiment", help="group/name under lab/experiments, e.g. replication/baseline")
    ap.add_argument("--model", help="use this registry model instead of the experiment's")
    ap.add_argument("--robot", choices=["rig", "replay"], default="rig")
    ap.add_argument("--episode", type=int, default=0, help="replay robot: which episode of the model's dataset")
    ap.add_argument("--endpoint", default="tcp/127.0.0.1:7447", help="the policy server's Zenoh endpoint")
    ap.add_argument("--deployment", help="the deployment name to ask for (default: from the model's name; an eval's "
                                         "servers add the frozen commit, see `lab eval servers`)")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--duration", type=float, help="seconds per trial (default: the experiment's duration_s)")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-label", action="store_true", help="do not ask for a success label after each trial")
    ap.add_argument("--dry-run", action="store_true")
    a, extra = ap.parse_known_args(argv)
    reg = Registry()
    try:
        path = reg.experiment_path(a.experiment)
    except RegistryError as e:
        print(f"lab run: {e}", file=sys.stderr)
        return 1
    exp_id = reg.experiment_id(path)
    label = not a.no_label and a.robot != "replay"
    for trial in range(1, a.trials + 1):
        run_id = f"{datetime.now():%Y%m%d-%H%M%S}-t{trial:02d}"
        try:
            model_id = reg.compose(a.experiment, model_override=a.model)["model"]
            run_dir = RESULTS / f"{datetime.now():%Y%m%d}" / exp_id / model_id / run_id
            cmd, composed, notes = build(reg, a.experiment, a.model, a.robot, a.episode, a.endpoint, a.duration,
                                         run_dir, a.deployment)
        except RegistryError as e:
            print(f"lab run: {e}", file=sys.stderr)
            return 1
        cmd += extra
        print(" \\\n    ".join(cmd), flush=True)
        for n in notes:
            print(f"  NOTE {n}")
        if a.dry_run:
            return 0
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(composed, sort_keys=False))
        if a.robot == "rig" and trial > 1:
            input("Reset the scene, then press Enter to start the trial...")
        started = datetime.now()
        outcome, error, took = rollout_once(cmd, run_dir)
        if outcome != "completed":
            print(f"trial {trial}: {outcome}{f': {error}' if error else ''}", flush=True)
        success = ""
        if label:
            ans = input("Success? [y/n/blank = unknown]: ").strip().lower()
            success = {"y": "1", "n": "0"}.get(ans, "")
        commit, dirty = git_state()
        host = reg.current_host()
        info = {"run_id": run_id, "experiment": exp_id, "model": model_id, "trial": trial, "success": success,
                "started_at": started.isoformat(timespec="seconds"), "duration_s": round(took, 1),
                "git_commit": commit, "git_dirty": dirty, "note": a.note, "run_dir": str(run_dir),
                "robot": a.robot, "host": host.id if host else socket.gethostname(), "outcome": outcome}
        (run_dir / "run.json").write_text(json.dumps({**info, "outcome": outcome, "error": error, "command": cmd,
                                                      "not_applied": notes}, indent=2))
        append_index([info])
        print(f"trial {trial}: {run_dir} (test run; official trials: lab eval run)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
