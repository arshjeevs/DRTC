"""Run an experiment condition for N trials against the policy server.

Each trial writes results/<group>/<name>/<model>/<run_id>/ containing:
    <run_id>.csv / .json   per-tick metrics + trajectories (from the DRTC client)
    config.yaml            fully resolved config (defaults < rig < experiment)
    run.json               provenance: git commit, dirty flag, timing, label, note
and appends one row to results/index.csv.

Usage:
    python -m lab.tools.run smoke/jack_baseline
    python -m lab.tools.run replication/drop_obs --trials 20 --note "new lighting"
    python -m lab.tools.run replication/drop_obs --model act_cube_v1 --trials 20
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

from lab.tools.config import REPO_ROOT, RESULTS_DIR, compose, experiment_id, load_rig, resolve_experiment_path

UPSTREAM_RUNNER = REPO_ROOT / "examples/experiments/run_drtc_experiment.py"
INDEX_FIELDS = [
    "run_id", "experiment", "model", "trial", "success", "started_at", "duration_s",
    "git_commit", "git_dirty", "note", "run_dir",
]


def load_upstream_runner():
    spec = importlib.util.spec_from_file_location("run_drtc_experiment", UPSTREAM_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_state() -> tuple[str, bool]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()

    return git("rev-parse", "--short", "HEAD"), bool(git("status", "--porcelain"))


def server_reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def ask_label() -> str:
    while True:
        answer = input("Success? [y]es / [n]o / [s]kip: ").strip().lower()
        if answer in {"y", "n", "s", ""}:
            return {"y": "1", "n": "0"}.get(answer, "")


def append_index(row: dict) -> None:
    index = RESULTS_DIR / "index.csv"
    new = not index.exists()
    with index.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lab experiment condition")
    parser.add_argument("experiment", help="group/name under lab/experiments, or a YAML path")
    parser.add_argument("--model", help="Model name from lab/models.yaml (overrides the experiment's)")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--no-label", action="store_true", help="Skip the success prompt after each trial")
    parser.add_argument("--note", default="", help="Free-text note stored with every trial")
    args = parser.parse_args()

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    exp_path = resolve_experiment_path(args.experiment)
    exp_id = experiment_id(exp_path)
    resolved = compose(exp_path, args.model)
    rig = load_rig()
    host, port = rig["server"]["host"], int(rig["server"]["port"])

    if not server_reachable(host, port):
        raise SystemExit(f"Policy server not reachable at {host}:{port}. Start lab/bin/server on the laptop.")

    runner = load_upstream_runner()
    config = runner._parse_experiment_dict(resolved)
    model = resolved["model"]
    out_dir = RESULTS_DIR / exp_id / model
    commit, dirty = git_state()
    if dirty:
        print("NOTE: working tree has uncommitted changes; run.json records git_dirty=true.")

    for trial in range(1, args.trials + 1):
        input(f"\n[{exp_id} | {model}] trial {trial}/{args.trials}: reset the scene, then press Enter...")
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-t{trial:02d}"
        started = time.time()
        result = runner.run_experiment(
            config,
            out_dir,
            server_address=f"{host}:{port}",
            trajectory_viz_ws_url=f"ws://{host}:8089",
            task=resolved["task"],
            experiment_name=run_id,
        )
        duration = round(time.time() - started, 1)
        if not result.get("success"):
            print(f"Trial {trial} failed to produce metrics: {result.get('error', 'unknown error')}")
            continue

        run_dir = Path(result["metrics_path"]).parent
        label = "" if args.no_label else ask_label()
        (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
        meta = {
            "run_id": run_dir.name, "experiment": exp_id, "model": model, "trial": trial, "success": label,
            "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
            "duration_s": duration, "git_commit": commit, "git_dirty": dirty, "note": args.note,
            "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        }
        (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
        append_index(meta)
        print(f"Saved {meta['run_dir']}")


if __name__ == "__main__":
    main()
