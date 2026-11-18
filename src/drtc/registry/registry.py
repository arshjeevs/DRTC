"""Load the lab registry (lab/), expand sweeps, resolve models, check consistency.

Layout:
    lab/rigs/*.yaml       id -> Rig
    lab/hosts.yaml        network + hosts
    lab/datasets/*.yaml   id -> Dataset
    lab/recipes/*.yaml    id -> Recipe
    lab/models/*.yaml     id -> Model
    lab/experiments/      conditions for robot runs (_defaults.yaml + <group>/<name>.yaml)
"""

from __future__ import annotations

import glob
import itertools
import os
import re
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .schema import Dataset, Host, HostsFile, Model, Recipe, Rig, Settings

ROBOT = "so101"
DATASET_NAME = re.compile(rf"^{ROBOT}_(?P<base>[a-z0-9]+_v\d+)(?:_(?P<variant>[a-z0-9]+))?$")
MODEL_NAME = re.compile(
    r"^(?P<policy>[a-z0-9]+)_(?P<base>[a-z0-9]+_v\d+)_(?P<method>full|lora)(?P<variants>(?:_[a-z0-9]+)*)$"
)
# Model-name prefix per policy where it is not the policy type without underscores: GR00T N1.5 models were always
# named groot_* (N1.7 ones are grootn17_*).
NAME_PREFIX = {"groot_n15": "groot"}
# Policies whose server-side inference accepts RTC inpainting (flow matching).
RTC_POLICIES = {"smolvla", "pi0", "pi05", "groot", "groot_n15"}


class RegistryError(Exception):
    pass


def find_lab_dir() -> Path:
    """LAB_DIR, else the nearest lab/ with hosts.yaml above the working directory, else the source checkout's."""
    if os.environ.get("LAB_DIR"):
        return Path(os.environ["LAB_DIR"]).expanduser().resolve()
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / "lab" / "hosts.yaml").is_file():
            return d / "lab"
    here = Path(__file__).resolve().parents[3] / "lab"  # src/drtc/registry -> repo root
    if (here / "hosts.yaml").is_file():
        return here
    raise RegistryError("no lab/ directory found (set LAB_DIR)")


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_device(path: Any) -> Any:
    """Expand a /dev glob to its single match; zero or several matches return the pattern unchanged."""
    if not isinstance(path, str) or not path.startswith("/dev/"):
        return path
    matches = glob.glob(path) if any(c in path for c in "*?[") else [path]
    return matches[0] if len(matches) == 1 else path


def for_host(value: Any, host: str | None) -> Any:
    """A per-host value ({default: ..., <host id>: ...}) -> the value for this host."""
    if isinstance(value, dict) and "default" in value:
        return value.get(host, value["default"]) if host else value["default"]
    return value


def _load_kind(lab: Path, kind: str, cls):
    entries: dict[str, Any] = {}
    for path in sorted((lab / kind).glob("*.yaml")):
        for key, raw in (load_yaml(path) or {}).items():
            if key in entries:
                raise RegistryError(f"{kind}: {key!r} defined twice (second time in {path.name})")
            try:
                entries[key] = cls.model_validate({**(raw or {}), "id": key})
            except Exception as e:  # pydantic's message names the field
                raise RegistryError(f"{path.relative_to(lab.parent)}: {key}: {e}") from e
    return entries


def _expand_sweeps(models: dict[str, Model]) -> dict[str, Model]:
    out: dict[str, Model] = {}
    for key, m in models.items():
        if not m.sweep:
            out[key] = m
            continue
        names, values = zip(*m.sweep.items())
        for combo in itertools.product(*values):
            new_id = key
            overrides = dict(m.overrides)
            for name, value in zip(names, combo):
                new_id = new_id.replace("{" + name + "}", str(value))
                overrides[name] = value
            if "{" in new_id:
                raise RegistryError(f"models: sweep {key!r} leaves a placeholder in {new_id!r}")
            if new_id in models or new_id in out:
                raise RegistryError(f"models: sweep {key!r} produces {new_id!r}, which already exists")
            out[new_id] = m.model_copy(update={"id": new_id, "sweep": None, "overrides": overrides})
    return out


@dataclass
class Resolved:
    """A model with its recipe, dataset and overrides applied."""

    id: str
    policy: str
    finetune: str | None
    hub: str
    fps: int
    task: str
    serve: dict[str, Any]
    status: str
    dataset: Dataset | None = None
    recipe: Recipe | None = None
    base: str | None = None
    train: dict[str, Any] = field(default_factory=dict)  # epochs|steps, batch_size, save_freq, episodes, split
    flags: dict[str, Any] = field(default_factory=dict)
    lora: dict[str, Any] | None = None

    @property
    def trainable(self) -> bool:
        return self.dataset is not None and self.recipe is not None

    def legacy_entry(self) -> dict[str, Any]:
        """The entry as the old lab/models.yaml wrote it (what the old tools and the policy server read)."""
        d: dict[str, Any] = {"policy_type": self.policy, "pretrained_name_or_path": self.hub, "fps": self.fps,
                             "task": self.task, **self.serve}
        if self.finetune is not None:
            d["finetune"] = self.finetune
        if self.trainable:
            t: dict[str, Any] = {"dataset": self.dataset.repo}
            if self.base is not None:
                t["from"] = self.base
            t.update(self.train)
            if self.flags:
                t["flags"] = dict(self.flags)
            if self.lora is not None:
                t["lora"] = dict(self.lora)
            d["train"] = t
        return d


class Registry:
    def __init__(self, lab: Path | None = None):
        self.lab = lab or find_lab_dir()
        self.settings = Settings.model_validate(load_yaml(self.lab / "lab.yaml"))
        self.renames: dict[str, Any] = load_yaml(self.lab / "renames.yaml") if (self.lab / "renames.yaml").is_file() else {}
        self.rigs: dict[str, Rig] = _load_kind(self.lab, "rigs", Rig)
        hosts_raw = load_yaml(self.lab / "hosts.yaml")
        hosts_raw["hosts"] = {k: {**(v or {}), "id": k} for k, v in (hosts_raw.get("hosts") or {}).items()}
        hf = HostsFile.model_validate(hosts_raw)
        self.network = hf.network
        self.hosts: dict[str, Host] = hf.hosts
        self.datasets: dict[str, Dataset] = _load_kind(self.lab, "datasets", Dataset)
        self.recipes: dict[str, Recipe] = _load_kind(self.lab, "recipes", Recipe)
        self.models: dict[str, Model] = _expand_sweeps(_load_kind(self.lab, "models", Model))

    # --- where am I ---------------------------------------------------------------------------------------------

    def current_host(self) -> Host | None:
        """LAB_HOST, else the host whose `address`/id matches this machine's hostname."""
        name = os.environ.get("LAB_HOST")
        if name:
            if name not in self.hosts:
                raise RegistryError(f"LAB_HOST={name!r} is not in hosts.yaml ({', '.join(self.hosts)})")
            return self.hosts[name]
        hostname = socket.gethostname()
        for h in self.hosts.values():
            if hostname in (h.id, h.label, h.address):
                return h
        return None

    def current_rig(self) -> Rig:
        """The rig in use: the newest known rig without an end date."""
        live = [r for r in self.rigs.values() if r.known and r.until is None]
        if len(live) != 1:
            raise RegistryError(f"expected exactly one current rig (no `until`), found {[r.id for r in live]}")
        return live[0]

    # --- models -------------------------------------------------------------------------------------------------

    def resolve(self, model_id: str) -> Resolved:
        if model_id not in self.models:
            new = (self.renames.get("models") or {}).get(model_id)
            hint = f" (renamed to {new!r} on 2026-11-18)" if new else ""
            raise RegistryError(f"unknown model {model_id!r}{hint}")
        m = self.models[model_id]
        hub = m.hub or f"{self.settings.hub_account}/{m.id}"
        if m.recipe is None:  # external checkpoint
            missing = [k for k in ("policy", "fps", "task") if getattr(m, k) is None]
            if missing:
                raise RegistryError(f"model {model_id!r} has no recipe, so it needs {missing}")
            return Resolved(id=m.id, policy=m.policy, finetune=m.finetune, hub=hub, fps=m.fps, task=m.task,
                            serve=dict(m.serve), status=m.status)

        r = self.recipes[m.recipe]
        ds = self.datasets[m.dataset] if m.dataset else None
        sections: dict[str, Any] = {
            "train": dict(r.train), "flags": dict(r.flags), "serve": dict(r.serve),
            "lora": dict(r.lora) if r.lora is not None else None,
            "base": r.base, "task": ds.task if ds else m.task, "fps": ds.fps if ds else m.fps,
        }
        if ds and ds.split and "split" not in sections["train"]:
            sections["train"]["split"] = {"total": ds.split.total or ds.episodes, "held_out": ds.split.held_out,
                                          "seed": ds.split.seed}
        if r.cameras and ds:
            rig = self.rigs[ds.rig]
            rename = {f"observation.images.{k}": r.cameras[c.role] for k, c in rig.cameras.items()
                      if c.role in r.cameras}
            if "rename_map" in sections["flags"]:
                raise RegistryError(f"recipe {r.id!r} sets both cameras and flags.rename_map")
            sections["flags"]["rename_map"] = rename
        for dotted, value in m.overrides.items():
            section, _, key = dotted.partition(".")
            if section not in sections:
                raise RegistryError(f"model {model_id!r}: override {dotted!r}: unknown section {section!r}")
            if not key:
                sections[section] = value
            elif value is None and section != "serve":
                (sections[section] or {}).pop(key, None)  # `train.split: null` removes it
            else:
                if sections[section] is None:
                    sections[section] = {}
                sections[section][key] = value
        if sections["task"] is None or sections["fps"] is None:
            raise RegistryError(f"model {model_id!r}: no dataset, so it needs overrides task and fps")
        return Resolved(id=m.id, policy=r.policy, finetune=r.finetune, hub=hub, fps=sections["fps"],
                        task=sections["task"], serve=sections["serve"], status=m.status, dataset=ds, recipe=r,
                        base=sections["base"], train=sections["train"], flags=sections["flags"],
                        lora=sections["lora"])

    def legacy_models(self) -> dict[str, dict[str, Any]]:
        return {k: self.resolve(k).legacy_entry() for k in self.models}

    # --- robot runs (old compose: _defaults < rig < model < experiment) -----------------------------------------

    def runner_fields(self, rig: Rig | None = None, host: Host | None = None) -> dict[str, Any]:
        """The rig as the robot-run tools read it (old rig_to_runner_fields), for the machine running the client."""
        rig = rig or self.current_rig()
        host = host or self.current_host()
        hid = host.id if host else None
        server_host = self.hosts.get(for_host(self.network.legacy_server.get("host_id"), hid))
        client = host if host and "client" in host.roles else \
            self.hosts.get(for_host(self.network.legacy_server.get("client_id"), hid))
        cams = list(rig.cameras.values())
        first = cams[0]
        return {
            "robot_type": rig.robot.type.removesuffix("_follower"),
            "robot_port": resolve_device(for_host(rig.robot.port, hid)),
            "robot_id": rig.robot.id,
            "client_host": (client.label or client.id) if client else "unknown",
            "server_host": (server_host.label or server_host.id) if server_host else "unknown",
            "gpu": server_host.gpus[0].name if server_host and server_host.gpus else "unknown",
            **{f"{k}_path": resolve_device(for_host(c.path, hid)) for k, c in rig.cameras.items()},
            "camera_width": first.width,
            "camera_height": first.height,
            "camera_fps": first.fps,
            "camera_fourcc": first.fourcc,
            "camera_use_threaded_async_read": True,
            "camera_allow_stale_frames": True,
        }

    def experiment_path(self, experiment: str) -> Path:
        path = Path(experiment)
        if path.suffix in {".yaml", ".yml"} and path.exists():
            return path.resolve()
        cand = self.lab / "experiments" / (experiment if experiment.endswith(".yaml") else f"{experiment}.yaml")
        if cand.exists():
            return cand
        raise RegistryError(f"experiment not found: {experiment}")

    def experiment_id(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.lab / "experiments").with_suffix(""))
        except ValueError:
            return path.stem

    def compose(self, experiment: str | Path, model_override: str | None = None,
                host: Host | None = None) -> dict[str, Any]:
        path = self.experiment_path(str(experiment))
        exp = load_yaml(path)
        model_id = model_override or exp.get("model")
        entry = self.resolve(model_id).legacy_entry()
        if model_override:
            exp = {k: v for k, v in exp.items() if k not in entry}
        merged: dict[str, Any] = {}
        merged.update(load_yaml(self.lab / "experiments" / "_defaults.yaml"))
        merged.update(self.runner_fields(host=host))
        merged.update(entry)
        merged.update(exp)
        merged["model"] = model_id
        merged.setdefault("name", self.experiment_id(path).replace("/", "__"))
        if merged.get("rtc_enabled") and merged["policy_type"] not in RTC_POLICIES:
            raise RegistryError(f"rtc_enabled requires a flow-matching policy ({', '.join(sorted(RTC_POLICIES))});"
                                f" model {model_id!r} is {merged['policy_type']!r}")
        return merged

    # --- checks -------------------------------------------------------------------------------------------------

    def check(self) -> list[str]:
        """Problems that make the registry inconsistent; [] if none. Does not touch the network."""
        p: list[str] = []
        for d in self.datasets.values():
            if d.rig not in self.rigs:
                p.append(f"dataset {d.id}: unknown rig {d.rig!r}")
            for parent in d.parents:
                if parent not in self.datasets:
                    p.append(f"dataset {d.id}: unknown parent {parent!r}")
            if not d.legacy and (d.revision is None or d.frames is None or d.episodes is None):
                p.append(f"dataset {d.id}: needs revision, episodes and frames (the pin of its Hub version)")
            if not d.legacy and d.use_degrees is None:
                p.append(f"dataset {d.id}: needs use_degrees (its joint units: false = -100..100, true = degrees)")
            if d.split and not (d.split.total or d.episodes):
                p.append(f"dataset {d.id}: split needs `total` or the dataset's `episodes`")
            if d.split and d.episodes and d.split.total and d.split.total > d.episodes:
                p.append(f"dataset {d.id}: split.total {d.split.total} > episodes {d.episodes}")
            if not d.legacy:
                account, _, name = d.repo.partition("/")
                if account != self.settings.hub_account or not DATASET_NAME.match(name) or name != d.id:
                    p.append(f"dataset {d.id}: should be {self.settings.hub_account}/{ROBOT}_<task>_v<N>[_<variant>] named as its id")
        hubs: dict[str, list[str]] = {}
        for m in self.models.values():
            if m.recipe and m.recipe not in self.recipes:
                p.append(f"model {m.id}: unknown recipe {m.recipe!r}")
                continue
            if m.dataset and m.dataset not in self.datasets:
                p.append(f"model {m.id}: unknown dataset {m.dataset!r}")
                continue
            try:
                r = self.resolve(m.id)
            except RegistryError as e:
                p.append(str(e))
                continue
            hubs.setdefault(r.hub, []).append(m.id)
            if r.serve.get("rtc_enabled") and r.policy not in RTC_POLICIES:
                p.append(f"model {m.id}: rtc_enabled but {r.policy!r} is not flow matching")
            if not m.legacy and r.trainable:
                p += [f"model {m.id}: {x}" for x in self._naming(r)]
        # Legacy entries (before the naming convention) are exempt, as in the old checker.
        p += [f"models {', '.join(ids)} share the Hub repo {hub}" for hub, ids in hubs.items()
              if len(ids) > 1 and not all(self.models[i].legacy for i in ids)]
        return p

    def _naming(self, r: Resolved) -> list[str]:
        out = []
        mm = MODEL_NAME.match(r.id)
        if not mm:
            return ["name should be <policy>_<task>_v<N>_<full|lora>[_<variant>]"]
        if mm["policy"] != NAME_PREFIX.get(r.policy, r.policy.replace("_", "")):
            out.append(f"name starts with {mm['policy']!r} but the policy is {r.policy!r}")
        if mm["method"] != r.finetune:
            out.append(f"name says {mm['method']!r} but the recipe fine-tunes {r.finetune!r}")
        if r.hub != f"{self.settings.hub_account}/{r.id}":
            out.append(f"Hub repo should be {self.settings.hub_account}/{r.id} (is {r.hub})")
        dm = DATASET_NAME.match(r.dataset.repo.partition("/")[2])
        if dm and dm["base"] != mm["base"]:
            out.append(f"name says {mm['base']!r} but the dataset is {dm['base']!r}")
        elif dm and dm["variant"] and dm["variant"] not in mm["variants"].split("_"):
            out.append(f"dataset variant {dm['variant']!r} missing from the name (add _{dm['variant']})")
        return out


def load_registry(lab: Path | None = None) -> Registry:
    try:
        return Registry(lab)
    except RegistryError as e:
        sys.exit(f"registry: {e}")
