"""Config composition: _defaults.yaml < rig.yaml < models.yaml[model] < experiment file.

Hardware (rig), models and experimental conditions live in separate files so each
can change without touching the others.

CLI (for shell scripts):
    python -m lab.tools.config get phonecam.device
"""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

LAB_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = LAB_DIR.parent
# LAB_RIG picks another rig file, e.g. lab/rig.windows.yaml on a Windows laptop.
RIG_PATH = Path(os.environ.get("LAB_RIG") or LAB_DIR / "rig.yaml").resolve()
MODELS_PATH = LAB_DIR / "models.yaml"
EXPERIMENTS_DIR = LAB_DIR / "experiments"
DEFAULTS_PATH = EXPERIMENTS_DIR / "_defaults.yaml"
RESULTS_DIR = REPO_ROOT / "results"

# Policies whose server-side inference accepts RTC inpainting kwargs (flow matching).
RTC_POLICIES = {"smolvla", "pi0", "pi05"}
# Policies whose dependencies conflict with the main .venv run from .venv-<name> (lab/bin/venv <name>).
POLICY_VENVS = {"pi0": "pi", "pi05": "pi"}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_device(path: str) -> str:
    """Expand a /dev glob (e.g. /dev/v4l/by-id/*Brio*-video-index0) to its single match.

    Zero or several matches return the pattern unchanged, so preflight reports it as missing.
    """
    matches = glob.glob(path) if any(c in path for c in "*?[") else [path]
    return matches[0] if len(matches) == 1 else path


def device_present(device: Any) -> bool:
    """Whether a serial port or camera exists: a /dev path, a Windows COM port, or a camera index.

    Camera indexes (Windows) cannot be checked without opening the camera, so they count as present.
    """
    if isinstance(device, int) or str(device).isdigit():
        return True
    if re.fullmatch(r"COM\d+", str(device), re.IGNORECASE):
        from serial.tools import list_ports

        return any(p.device.upper() == str(device).upper() for p in list_ports.comports())
    return Path(device).exists()


def _resolve_devices(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _resolve_devices(v) for k, v in node.items()}
    if isinstance(node, str) and node.startswith("/dev/"):
        return resolve_device(node)
    return node


def load_rig() -> dict[str, Any]:
    return _resolve_devices(load_yaml(RIG_PATH))


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
    """Registry entries; top-level keys starting with "_" hold shared YAML anchors, not models."""
    return {k: v for k, v in load_yaml(MODELS_PATH).items() if not k.startswith("_")}


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


# Naming convention (lab/models.yaml header, lab/PROPOSAL.md 7.2). Entries listed in _legacy are exempt.
HUB_ACCOUNT = "orangemoon599"
ROBOT = "so101"
DATASET_NAME = re.compile(rf"^{ROBOT}_(?P<base>[a-z0-9]+_v\d+)(?:_(?P<variant>[a-z0-9]+))?$")
MODEL_NAME = re.compile(r"^(?P<policy>[a-z0-9]+)_(?P<base>[a-z0-9]+_v\d+)_(?P<method>full|lora)(?P<variants>(?:_[a-z0-9]+)*)$")


def legacy_models() -> set[str]:
    return set(load_yaml(MODELS_PATH).get("_legacy") or [])


def naming_problems(name: str, entry: dict[str, Any], all_models: dict[str, dict[str, Any]]) -> list[str]:
    """Ways a (non-legacy) registry entry breaks the naming convention; [] if it follows it."""
    problems = []
    target = str(entry.get("pretrained_name_or_path", ""))
    m = MODEL_NAME.match(name)
    if not m:
        problems.append(f"name should be <policy>_<task>_v<N>_<full|lora>[_<variant>], e.g. smolvla_cubebox_v1_lora")
    else:
        if m["policy"] != entry.get("policy_type"):
            problems.append(f"name starts with {m['policy']!r} but policy_type is {entry.get('policy_type')!r}")
        if m["method"] != entry.get("finetune", "full"):
            problems.append(f"name says {m['method']!r} but finetune is {entry.get('finetune', 'full')!r}")
    if "/" in target and not target.startswith(("outputs/", "/", ".")) and target != f"{HUB_ACCOUNT}/{name}":
        problems.append(f"pretrained_name_or_path should be {HUB_ACCOUNT}/{name} (is {target})")
    others = [other for other, e in all_models.items() if other != name and e.get("pretrained_name_or_path") == target]
    if others:
        problems.append(f"shares its Hub repo {target} with {', '.join(others)}")
    dataset = str((entry.get("train") or {}).get("dataset", ""))
    if dataset:
        account, _, ds_name = dataset.partition("/")
        d = DATASET_NAME.match(ds_name)
        if account != HUB_ACCOUNT or not d:
            problems.append(f"dataset should be {HUB_ACCOUNT}/{ROBOT}_<task>_v<N>[_<variant>] (is {dataset})")
        elif m and d["base"] != m["base"]:
            problems.append(f"name says {m['base']!r} but the dataset is {d['base']!r}")
        elif m and d["variant"] and d["variant"] not in m["variants"].split("_"):
            problems.append(f"dataset variant {d['variant']!r} missing from the name (add _{d['variant']})")
    return problems


def check_naming() -> dict[str, list[str]]:
    """{entry: problems} for every non-legacy registry entry that breaks the convention."""
    models, legacy = load_models(), legacy_models()
    found = {name: naming_problems(name, entry, models) for name, entry in models.items() if name not in legacy}
    return {name: problems for name, problems in found.items() if problems}


def venv_for(model_name: str) -> str:
    """Side venv a registry model needs ('' = the main .venv)."""
    entry = load_models().get(model_name)
    return POLICY_VENVS.get(entry["policy_type"], "") if entry else ""


def print_registry() -> None:
    print("models (lab/models.yaml):")
    for name, m in load_models().items():
        print(f"  {name:28s} {m['policy_type']:8s} {m.get('fps', '?')} fps  {m['pretrained_name_or_path']}")
    for name, problems in check_naming().items():
        print(f"  NAMING {name}: " + "; ".join(problems))
    print("experiments (lab/experiments):")
    for path in sorted(EXPERIMENTS_DIR.rglob("*.yaml")):
        if path.name.startswith("_"):
            continue
        exp = load_yaml(path)
        print(f"  {experiment_id(path):28s} model={exp.get('model', '?'):22s} {exp.get('description', '')}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "get":
        print(rig_get(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "venv":
        print(venv_for(sys.argv[2]))
    elif len(sys.argv) == 2 and sys.argv[1] == "list":
        print_registry()
    else:
        sys.exit("usage: python -m lab.tools.config get <dotted.key> | venv <model> | list")
