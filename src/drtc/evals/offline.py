"""`lab eval offline <eval>`: M9 (held-out action error) and M10 (inference time, GPU memory) for an eval's models,
without the robot (lab/METRICS.md). Run it on a GPU server, never during a robot session on that server.

Every model is built exactly as `lab eval serve` builds it (the same server config, LeRobot's load_deployment and its
PolicyRunner), so the numbers are those of the served model. Checkpoints, per model:
- `root`: the repo root at the eval's frozen commit, the served checkpoint (the best by held-out loss);
- `last`: checkpoints/<step>/pretrained_model at the same commit, the end of training;
- with `--local`, every checkpoint under outputs/train/<model>/checkpoints/ on this machine.

M9: from every 10th frame of the dataset's held-out episodes (the seeded split, never trained on), the model predicts a
chunk from the recorded observation; its first 15 actions are compared with the 15 recorded actions that follow.
Mean absolute error in the dataset's units (degrees for so101_v2 datasets, -100..100 for v0/v1): the five arm joints
(mean, and each) and the gripper. One sample per prediction, torch seed 0.
M10 (served checkpoint only): the runner's own timings (preprocessing + policy + postprocessing, as M6) over 200
predictions after 10 warm-up ones, and the peak GPU memory.

Results: results/offline/<dataset>/<model>.json, pushed to the results store (offline/<dataset>/<model>.json).
"""

from __future__ import annotations

import json
import socket
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from drtc.evals import spec as S
from drtc.registry import Registry, RegistryError

FRAMES_EVERY = 10
HORIZON = 15
BENCH_WARMUP, BENCH_REQUESTS = 10, 200
ARM = 5


def out_path(dataset: str, model: str) -> Path:
    return Path("results") / "offline" / dataset / f"{model}.json"


def checkpoint_paths(reg: Registry, model: str, revision: str, local: bool) -> dict[str, Path]:
    """name -> local checkpoint folder (downloaded from the Hub at `revision` where needed)."""
    from huggingface_hub import HfApi, snapshot_download

    from drtc.control.serve import fetch_checkpoint

    hub = reg.resolve(model).hub
    out = {"root": fetch_checkpoint(hub, revision)}
    files = HfApi().list_repo_files(hub, revision=revision)
    steps = sorted({f.split("/")[1] for f in files if f.startswith("checkpoints/") and "/pretrained_model/" in f})
    if steps:
        last = steps[-1]
        snap = Path(snapshot_download(hub, revision=revision, allow_patterns=[f"checkpoints/{last}/pretrained_model/*"]))
        out[f"last:{int(last)}"] = snap / "checkpoints" / last / "pretrained_model"
    if local:
        for p in sorted((Path("outputs") / "train" / model / "checkpoints").glob("*/pretrained_model")):
            if p.parent.name.isdigit():
                out[f"local:{int(p.parent.name)}"] = p
    return out


def build_runner(reg: Registry, model: str, path: Path, device: str, revision: str | None = None):
    """The served model at `path`: the same config as lab serve, LeRobot's load_deployment."""
    import draccus
    import yaml
    from lerobot.remote_inference.configs import ServerConfig
    from lerobot.remote_inference.deployment import load_deployment
    from lerobot.utils.import_utils import register_third_party_plugins

    from drtc.compat.serve import install_loader, needs_compat
    from drtc.control.chunk_adapters import install as install_chunk_adapters
    from drtc.control.serve import server_config

    register_third_party_plugins()
    install_chunk_adapters()
    cfg = server_config(reg, model, str(path), device)
    hub = reg.resolve(model).hub
    if needs_compat(hub, revision):  # a model kept from the minor (old checkpoint format)
        from drtc.compat.serve import prepare_checkpoint

        install_loader()
        if path.name != "pretrained_model":
            cfg["model"]["repo_or_path"] = str(prepare_checkpoint(hub, revision))
    tmp = Path("outputs") / "offline" / f"{model}.yaml"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    server_cfg = draccus.parse(ServerConfig, config_path=str(tmp), args=[])
    runner, identity = load_deployment(server_cfg)
    return runner, identity, cfg


def samples(reg: Registry, model: str):
    """The held-out dataset, and the rows predicted from: (row, the recorded actions that follow)."""
    from lerobot.datasets import LeRobotDataset

    from drtc.evals.run import held_out_episodes

    r = reg.resolve(model)
    held = held_out_episodes(reg, model)
    ds = LeRobotDataset(r.dataset.repo, revision=r.dataset.revision, episodes=held)
    cols = ds.hf_dataset.select_columns(["action", "episode_index", "frame_index"]).with_format("numpy")
    actions = np.stack(cols["action"]).astype(np.float32)
    episodes = np.asarray(cols["episode_index"])
    frames = np.asarray(cols["frame_index"])
    rows = [i for i in range(len(frames)) if frames[i] % FRAMES_EVERY == 0 and i + HORIZON <= len(frames)
            and episodes[i + HORIZON - 1] == episodes[i]]
    return ds, held, rows, actions


def snapshot(item: dict, cfg: dict, renames: dict):
    """A dataset frame as the robot would send it: images HWC uint8, the state float32."""
    import torch
    from lerobot.inference import ObservationSnapshot

    names = {f["name"]: f for f in cfg["features"]}
    feats = {}
    for key, value in item.items():
        name = renames.get(key, key)
        if name not in names or not isinstance(value, torch.Tensor):
            continue
        if names[name].get("kind") == "rgb":
            feats[name] = (value.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        else:
            feats[name] = value.float().cpu().numpy()
    return ObservationSnapshot(feats, time.monotonic(), item.get("task", ""), 0, observation_id="offline")


def evaluate(reg: Registry, model: str, revision: str, device: str, local: bool) -> dict:
    import torch
    from lerobot.inference import ExecutionMode

    from drtc.control.serve import camera_renames

    r = reg.resolve(model)
    renames = camera_renames(reg, model)
    ds, held, rows, actions = samples(reg, model)
    names = list(ds.meta.features["action"]["names"])
    result = {"metrics_version": "v1", "model": model, "dataset": r.dataset.id, "dataset_revision": r.dataset.revision,
              "revision": revision, "held_out_episodes": held, "frames_every": FRAMES_EVERY, "horizon": HORIZON,
              "predictions": len(rows), "units": "degrees" if r.dataset.use_degrees else "-100..100 (so101 v0/v1)",
              "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "made_at": datetime.now().isoformat(timespec="seconds"), "checkpoints": {}}
    for name, path in checkpoint_paths(reg, model, revision, local).items():
        print(f"  {model} [{name}]: {len(rows)} predictions from {len(held)} held-out episodes", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        runner, identity, cfg = build_runner(reg, model, path, device, revision)
        err, times = [], []
        order = rows + rows * ((BENCH_WARMUP + BENCH_REQUESTS) // max(len(rows), 1) + 1) if name == "root" else rows
        for k, i in enumerate(order):
            if k >= len(rows) and k >= BENCH_WARMUP + BENCH_REQUESTS:
                break
            torch.manual_seed(0)
            chunk = runner.predict(snapshot(ds[i], cfg, renames), mode=ExecutionMode.CHUNK, inference_delay=0,
                                   model_continuation=None)
            if k < len(rows):
                pred = chunk.canonical_actions[:HORIZON].float().cpu().numpy()
                n = min(len(pred), HORIZON)
                err.append(np.abs(pred[:n] - actions[i:i + n]).mean(axis=0))
            if BENCH_WARMUP <= k < BENCH_WARMUP + BENCH_REQUESTS:
                times.append(sum(chunk.server_durations.values()))
        e = np.mean(err, axis=0)
        entry = {"path": str(path), "artifact_identity": identity, "M9_arm_mae": float(e[:ARM].mean()),
                 "M9_gripper_mae": float(e[ARM]) if len(e) > ARM else None,
                 "M9_joint_mae": {names[j]: float(e[j]) for j in range(len(e))}}
        if name == "root" and times:
            entry["M10"] = {"server_ms_median": 1000 * float(np.median(times)),
                            "server_ms_p95": 1000 * float(np.percentile(times, 95)), "requests": len(times),
                            "peak_gpu_gb": (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available()
                            else None}
        result["checkpoints"][name] = entry
        print(f"    arm MAE {entry['M9_arm_mae']:.2f}, gripper {entry['M9_gripper_mae']:.2f}"
              + (f"; M10 {entry['M10']['server_ms_median']:.0f} ms median, {entry['M10']['peak_gpu_gb']:.1f} GB"
                 if "M10" in entry else ""), flush=True)
        del runner
    return result


def main(reg: Registry, name: str, models: list[str] | None, device: str, local: bool, push: bool) -> int:
    sp = S.load(reg, name)
    if not S.is_frozen(sp):
        raise RegistryError(f"{name} is not frozen (lab eval freeze {name})")
    for m in models or sp.models:
        res = evaluate(reg, m, sp.frozen[m], device, local)
        p = out_path(res["dataset"], m)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2))
        print(f"  -> {p}")
        if push:
            from huggingface_hub import HfApi

            from drtc.evals import sync

            HfApi().upload_file(path_or_fileobj=str(p), path_in_repo=f"offline/{res['dataset']}/{m}.json",
                                repo_id=sync.repo(reg), repo_type="dataset",
                                commit_message=f"offline: {m} ({res['dataset']})")
            print("     pushed to the results store")
    return 0
