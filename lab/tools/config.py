"""Config composition: _defaults.yaml < rig.yaml < experiment file.

The rig (hardware) and experiments (conditions) live in separate files so that
hardware changes never touch experiment definitions, and vice versa.

CLI (for shell scripts):
    python -m lab.tools.config get phonecam.device
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

LAB_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = LAB_DIR.parent
RIG_PATH = LAB_DIR / "rig.yaml"
EXPERIMENTS_DIR = LAB_DIR / "experiments"
DEFAULTS_PATH = EXPERIMENTS_DIR / "_defaults.yaml"
RESULTS_DIR = REPO_ROOT / "results"


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def load_rig() -> dict[str, Any]:
    return load_yaml(RIG_PATH)


def rig_get(dotted_key: str) -> Any:
    value: Any = load_rig()
    for part in dotted_key.split("."):
        value = value[part]
    return value


def rig_to_runner_fields(rig: dict[str, Any]) -> dict[str, Any]:
    """Flatten rig.yaml into the field names used by examples/experiments/run_drtc_experiment.py."""
    cams = rig["cameras"]
    return {
        "robot_type": rig["robot"]["type"],
        "robot_port": rig["robot"]["port"],
        "robot_id": rig["robot"]["id"],
        "client_host": rig["hosts"]["client"],
        "server_host": rig["hosts"]["server"],
        "gpu": rig["hosts"]["gpu"],
        "camera1_path": cams["camera1"]["path"],
        "camera2_path": cams["camera2"]["path"],
        "camera_width": cams["width"],
        "camera_height": cams["height"],
        "camera_fps": cams["fps"],
        "camera_fourcc": cams["camera1"].get("fourcc"),
        "camera_use_threaded_async_read": True,
        "camera_allow_stale_frames": True,
    }


def resolve_experiment_path(experiment: str) -> Path:
    """Accept 'group/name', 'group/name.yaml', or a path to a YAML file."""
    path = Path(experiment)
    if path.suffix in {".yaml", ".yml"} and path.exists():
        return path.resolve()
    candidate = EXPERIMENTS_DIR / (experiment if experiment.endswith(".yaml") else f"{experiment}.yaml")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Experiment not found: {experiment} (looked in {EXPERIMENTS_DIR})")


def experiment_id(path: Path) -> str:
    """'lab/experiments/smoke/jack_baseline.yaml' -> 'smoke/jack_baseline'."""
    try:
        return str(path.relative_to(EXPERIMENTS_DIR).with_suffix(""))
    except ValueError:
        return path.stem


def compose(experiment_path: Path) -> dict[str, Any]:
    """Merge defaults, rig and experiment into one flat dict (later wins)."""
    merged: dict[str, Any] = {}
    merged.update(load_yaml(DEFAULTS_PATH))
    merged.update(rig_to_runner_fields(load_rig()))
    merged.update(load_yaml(experiment_path))
    merged.setdefault("name", experiment_id(experiment_path).replace("/", "__"))
    return merged


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "get":
        print(rig_get(sys.argv[2]))
    else:
        sys.exit("usage: python -m lab.tools.config get <dotted.key>")
