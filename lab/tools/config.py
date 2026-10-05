"""Config composition: _defaults.yaml < rig.yaml < models.yaml[model] < experiment file.

Hardware (rig), models and experimental conditions live in separate files so each
can change without touching the others.

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
MODELS_PATH = LAB_DIR / "models.yaml"
EXPERIMENTS_DIR = LAB_DIR / "experiments"
DEFAULTS_PATH = EXPERIMENTS_DIR / "_defaults.yaml"
RESULTS_DIR = REPO_ROOT / "results"

# Policies whose server-side inference accepts RTC inpainting kwargs (flow matching).
RTC_POLICIES = {"smolvla", "pi0", "pi05"}


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


def load_models() -> dict[str, dict[str, Any]]:
    return load_yaml(MODELS_PATH)


def compose(experiment_path: Path, model_override: str | None = None) -> dict[str, Any]:
    """Merge defaults, rig, model and experiment into one flat dict (later wins).

    The resolved dict keeps `model` (registry name) for provenance.
    """
    experiment = load_yaml(experiment_path)
    model_name = model_override or experiment.get("model")
    models = load_models()
    if model_name not in models:
        raise SystemExit(
            f"{experiment_path.name}: unknown model {model_name!r}. Registered: {', '.join(models) or 'none'}"
        )
    if model_override:
        # An explicit --model also replaces model fields the experiment pinned itself.
        experiment = {k: v for k, v in experiment.items() if k not in models[model_name]}

    merged: dict[str, Any] = {}
    merged.update(load_yaml(DEFAULTS_PATH))
    merged.update(rig_to_runner_fields(load_rig()))
    merged.update(models[model_name])
    merged.update(experiment)
    merged["model"] = model_name
    merged.setdefault("name", experiment_id(experiment_path).replace("/", "__"))

    if merged.get("rtc_enabled") and merged["policy_type"] not in RTC_POLICIES:
        raise SystemExit(f"rtc_enabled requires a flow-matching policy ({', '.join(sorted(RTC_POLICIES))}); "
                         f"model {model_name!r} is {merged['policy_type']!r}. Set rtc_enabled: false.")
    if "task" not in merged:
        raise SystemExit(f"No 'task' for model {model_name!r} (set it in models.yaml or the experiment).")
    return merged


def print_registry() -> None:
    print("models (lab/models.yaml):")
    for name, m in load_models().items():
        print(f"  {name:28s} {m['policy_type']:8s} {m.get('fps', '?')} fps  {m['pretrained_name_or_path']}")
    print("experiments (lab/experiments):")
    for path in sorted(EXPERIMENTS_DIR.rglob("*.yaml")):
        if path.name.startswith("_"):
            continue
        exp = load_yaml(path)
        print(f"  {experiment_id(path):28s} model={exp.get('model', '?'):22s} {exp.get('description', '')}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "get":
        print(rig_get(sys.argv[2]))
    elif len(sys.argv) == 2 and sys.argv[1] == "list":
        print_registry()
    else:
        sys.exit("usage: python -m lab.tools.config get <dotted.key> | list")
