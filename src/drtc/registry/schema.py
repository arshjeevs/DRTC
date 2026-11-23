"""What the lab registry holds: rigs, hosts, datasets, recipes, models.

Every file under lab/<kind>/ is a mapping of id -> entry (hosts: lab/hosts.yaml). Unknown keys are errors, so a
typo cannot silently fall back to a default.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Entry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = ""  # filled from the mapping key
    notes: str = ""


# --- rigs ---------------------------------------------------------------------------------------------------------


# A device path, or per host: {default: /dev/..., windows-laptop: COM5}
PerHost = str | int | dict[str, str | int]


class Arm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str  # LeRobot robot / teleoperator type, e.g. so101_follower
    port: PerHost  # /dev/serial/by-id/... (globs allowed)
    id: str  # calibration id
    # Calibration file, relative to LeRobot's calibration root. Always passed explicitly: LeRobot 0.7 looks in
    # robots/so_follower/ by default, where this laptop has an older calibration.
    calibration: str
    # Joint units for NEW recordings. From 2026-11-18: degrees (LeRobot 0.7's convention, the user's choice). Datasets
    # record their own units (Dataset.use_degrees); a model runs the robot in its dataset's units.
    use_degrees: bool = True


class Camera(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str  # wrist | scene | ...: recipes map roles to their policy's image names
    path: PerHost
    width: int
    height: int
    fps: int
    fourcc: str | None = None
    source: str | None = None  # e.g. phonecam (DroidCam bridge)


class Rig(Entry):
    """One physical setup. A new version whenever cameras, their placement or the scene change."""

    since: date | None = None
    until: date | None = None
    robot: Arm | None = None
    leader: Arm | None = None
    # Order matters: dataset feature names are observation.images.<key>.
    cameras: dict[str, Camera] = Field(default_factory=dict)
    phonecam: dict[str, Any] | None = None
    known: bool = True  # False: an old setup whose details were not recorded


# --- hosts --------------------------------------------------------------------------------------------------------

Role = Literal["control", "record", "client", "serve", "train"]


class Gpu(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    memory_gb: float
    count: int = 1


class Host(Entry):
    label: str = ""  # name recorded with runs (defaults to the id)
    hostname: str | None = None  # this machine's hostname, so `lab` knows where it runs (else set LAB_HOST)
    roles: list[Role]
    address: str | None = None
    ssh: str | None = None  # user@host, for hosts the laptop reaches over SSH
    container: str | None = None  # docker container the lab runs in
    repo: str | None = None  # checkout path on that host
    gpus: list[Gpu] = Field(default_factory=list)
    shared: bool = False  # other people's jobs use the GPUs too
    secrets: str | None = None  # KEY=VALUE file with HF_TOKEN, WANDB_API_KEY (drtc.secrets; not in git)
    # CPU share per policy type when training jobs share this machine: {policy: {threads, workers}}. Applied by
    # lab train when another lab train already runs here (drtc.train.cpu_budget); a job alone gets the machine.
    cpu_budget: dict[str, dict[str, int]] = Field(default_factory=dict)


class Network(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Old DRTC policy server (gRPC): what the Pi connects to.
    legacy_server: dict[str, Any] = Field(default_factory=dict)
    # New system: Zenoh router every policy server and client connects to.
    zenoh_router: str | None = None


class HostsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    network: Network = Field(default_factory=Network)
    hosts: dict[str, Host]


# --- datasets -----------------------------------------------------------------------------------------------------


class Split(BaseModel):
    """Seeded random held-out episodes. `total` pins the shuffle; it defaults to the dataset's episode count."""

    model_config = ConfigDict(extra="forbid")
    held_out: int
    seed: int = 0
    total: int | None = None


class Derivation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["autotrim", "camera_ablation", "merge", "subset", "edit"]
    params: dict[str, Any] = Field(default_factory=dict)


class Dataset(Entry):
    repo: str
    rig: str
    task: str
    fps: int
    # Pin of the exact Hub version (commit of the v3.0 tag) and its size: a local copy that does not match,
    # e.g. a stale cache under a reused name, is refused (`lab check --data`, every job).
    revision: str | None = None
    episodes: int | None = None
    frames: int | None = None
    # Joint units of this dataset: False = -100..100 (gripper 0..100; LeRobot 0.4.3, every dataset before
    # 2026-11-18), True = degrees (LeRobot 0.7). Models trained on it must run the robot in the same units.
    use_degrees: bool | None = None
    parents: list[str] = Field(default_factory=list)
    derivation: Derivation | None = None
    split: Split | None = None
    legacy: bool = False  # recorded before the naming convention


class Settings(BaseModel):
    """lab/lab.yaml"""

    model_config = ConfigDict(extra="forbid")
    hub_account: str
    trackio_project: str
    trackio_space: str | None = None  # Hugging Face Space all machines log to; None = local database only
    trackio_private: bool = True
    wandb_project: str | None = None  # every training run also logs here (drtc.compat.wandb_mirror); None = off
    wandb_entity: str | None = None
    results_repo: str | None = None  # the one official copy of evaluation results (lab/docs/EVALUATION.md), a Hub dataset


# --- recipes and models -------------------------------------------------------------------------------------------

SERVE_KEYS = {"rtc_enabled", "num_flow_matching_steps", "actions_per_chunk", "s_min"}
# stop_at: end the run cleanly (save, push) at this step; the LR schedule stays that of `steps` (a run cut short for time)
TRAIN_KEYS = {"epochs", "steps", "stop_at", "batch_size", "save_freq", "episodes", "split"}
DATA_KEYS = {"frames"}  # dataset preparation for speed (drtc.data.prepare)


class Recipe(Entry):
    """How to train one policy: what the authors or others used to make it work, and where that came from."""

    policy: str  # LeRobot policy type
    finetune: Literal["full", "lora"] = "full"
    base: str | None = None  # checkpoint to start from (repo or repo@revision); None = from scratch
    source: str = ""
    train: dict[str, Any] = Field(default_factory=dict)  # epochs | steps, batch_size, save_freq
    flags: dict[str, Any] = Field(default_factory=dict)  # extra lerobot-train flags (policy.*, rename_map, ...)
    # Camera role -> the policy's image feature; becomes rename_map for the dataset's rig.
    cameras: dict[str, str] = Field(default_factory=dict)
    lora: dict[str, Any] | None = None
    serve: dict[str, Any] = Field(default_factory=dict)  # rtc_enabled, num_flow_matching_steps, ...
    data: dict[str, Any] = Field(default_factory=dict)  # frames: [H, W] (a frame cache, drtc.data.prepare)
    # Speed settings (Phase 3): lerobot-train flags that change how fast training runs, not what it learns
    # (compile, workers). Kept apart from `flags` so the recipe itself stays fixed.
    speed: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _keys(self):
        bad = set(self.data) - DATA_KEYS
        if bad:
            raise ValueError(f"unknown data keys {sorted(bad)} (known: {sorted(DATA_KEYS)})")
        bad = set(self.serve) - SERVE_KEYS
        if bad:
            raise ValueError(f"unknown serve keys {sorted(bad)} (known: {sorted(SERVE_KEYS)})")
        bad = set(self.train) - TRAIN_KEYS
        if bad:
            raise ValueError(f"unknown train keys {sorted(bad)} (known: {sorted(TRAIN_KEYS)})")
        return self


Status = Literal["planned", "trained", "retired", "not_trainable", "external"]


class Model(Entry):
    """A trained (or to-be-trained) checkpoint: dataset x recipe, plus overrides.

    Overrides use dotted keys into the recipe: train.epochs, serve.actions_per_chunk, flags.policy.dtype, ...
    External checkpoints (trained elsewhere) set policy / fps / task / serve directly and have no recipe.
    """

    recipe: str | None = None
    dataset: str | None = None
    hub: str | None = None  # defaults to <hub account>/<id>
    status: Status = "trained"
    legacy: bool = False  # before the naming convention: exempt from the naming check
    overrides: dict[str, Any] = Field(default_factory=dict)
    # Sweep: one entry -> one model per value; the id holds {name} placeholders, e.g. ..._ep{train.episodes}
    sweep: dict[str, list[Any]] | None = None
    # Only for external checkpoints (no recipe):
    policy: str | None = None
    finetune: Literal["full", "lora"] | None = None
    fps: int | None = None
    task: str | None = None
    serve: dict[str, Any] = Field(default_factory=dict)
