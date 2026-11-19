"""Record a dataset with the leader arm: `lab record <name> --task "..."`.

    raw recording -> checks -> trim -> push both -> registry entries

- The hardware comes from the current rig (lab/rigs) for this machine (hosts.yaml): ports, cameras, the explicit
  calibration files and the joint units (degrees for new recordings, LeRobot 0.7's convention).
- Recording is LeRobot 0.7's lerobot-record; its terminal key listener works on Wayland and over SSH:
  -> / n next, <- / r re-record, Esc / q stop.
- The untrimmed recording is <name>_raw; the trimmed copy, the one to train on, is <name> (lab trim).
- Both are pushed (public, like every dataset here), and their entries, pinned to the pushed revision, are written
  to lab/datasets/recorded.yaml for you to review and commit.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import yaml

from drtc.data import video
from drtc.registry import Registry, for_host, resolve_device

REGISTRY_FILE = "recorded.yaml"


def rig_flags(reg: Registry) -> dict:
    """lerobot-record robot/teleop flags for the current rig, on this machine."""
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

    rig = reg.current_rig()
    host = reg.current_host()
    hid = host.id if host else None
    cams = {}
    for key, c in rig.cameras.items():
        spec = {"type": "opencv", "index_or_path": resolve_device(for_host(c.path, hid)), "width": c.width,
                "height": c.height, "fps": c.fps}
        if c.fourcc:
            spec["fourcc"] = c.fourcc
        cams[key] = spec
    flags = {
        "robot.type": rig.robot.type,
        "robot.port": resolve_device(for_host(rig.robot.port, hid)),
        "robot.id": rig.robot.id,
        "robot.calibration_dir": str((HF_LEROBOT_CALIBRATION / rig.robot.calibration).parent),
        "robot.use_degrees": rig.robot.use_degrees,
        "robot.cameras": cams,
    }
    if rig.leader:
        flags.update({
            "teleop.type": rig.leader.type,
            "teleop.port": resolve_device(for_host(rig.leader.port, hid)),
            "teleop.id": rig.leader.id,
            "teleop.calibration_dir": str((HF_LEROBOT_CALIBRATION / rig.leader.calibration).parent),
            "teleop.use_degrees": rig.leader.use_degrees,
        })
    return flags


def flag_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return str(v)


def check_recording(root: Path, fps: int) -> list[str]:
    """Problems in a freshly recorded dataset: episodes whose rows and video windows disagree, or gaps in time."""
    import numpy as np
    import pandas as pd

    problems = []
    info = json.loads((root / "meta/info.json").read_text())
    eps = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta/episodes").rglob("*.parquet"))])
    cams = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    data = pd.concat([pd.read_parquet(f, columns=["episode_index", "timestamp"]) for f in
                      sorted((root / "data").rglob("*.parquet"))])
    for _, row in eps.iterrows():
        ep, n = int(row["episode_index"]), int(row["length"])
        ts = np.sort(data[data.episode_index == ep]["timestamp"].to_numpy())
        if len(ts) != n:
            problems.append(f"episode {ep}: {len(ts)} rows, metadata says {n}")
        gaps = np.diff(ts)
        if len(gaps) and gaps.max() > 1.5 / fps:
            problems.append(f"episode {ep}: a {gaps.max():.3f} s gap between frames (expected {1 / fps:.3f})")
        for cam in cams:
            dur = float(row[f"videos/{cam}/to_timestamp"]) - float(row[f"videos/{cam}/from_timestamp"])
            if abs(dur - n / fps) > 1.5 / fps:
                problems.append(f"episode {ep} {cam}: video window {dur:.2f} s for {n / fps:.2f} s of frames")
    return problems


def registry_entries(reg: Registry, name: str, raw_repo: str, trimmed_repo: str, task: str, fps: int,
                     held_out: int) -> dict:
    from huggingface_hub import HfApi
    from lerobot.datasets import LeRobotDatasetMetadata

    api = HfApi()
    out = {}
    for ds_id, repo in ((f"{name}_raw", raw_repo), (name, trimmed_repo)):
        meta = LeRobotDatasetMetadata(repo)
        tags = {t.name: t.target_commit for t in api.list_repo_refs(repo, repo_type="dataset").tags}
        entry = {"repo": repo, "revision": tags.get("v3.0"), "rig": reg.current_rig().id, "task": task, "fps": fps,
                 "episodes": meta.total_episodes, "frames": meta.total_frames,
                 "use_degrees": reg.current_rig().robot.use_degrees}
        if ds_id == name:
            entry["parents"] = [f"{name}_raw"]
            entry["derivation"] = {"kind": "autotrim", "params": {"tool": "drtc.data.trim"}}
            entry["split"] = {"held_out": held_out, "seed": 0}
        out[ds_id] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab record", description=__doc__.split("\n\n")[0], allow_abbrev=False)
    ap.add_argument("name", help="dataset name, e.g. so101_cubebox_v2 (records <name>_raw, trims into <name>)")
    ap.add_argument("--task", required=True, help="the exact instruction, the same for every episode")
    ap.add_argument("--episodes", type=int, default=50,
                    help="total episodes in the recording (with --resume: records the ones still missing)")
    ap.add_argument("--episode-time", type=float, default=30)
    ap.add_argument("--reset-time", type=float, default=10)
    ap.add_argument("--held-out", type=int, default=10, help="held-out episodes in the trimmed dataset's split")
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing <name>_raw recording up to --episodes; the trimmed copy is rebuilt")
    ap.add_argument("--no-push", action="store_true", help="keep both datasets local")
    ap.add_argument("--display", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a, extra = ap.parse_known_args(argv)
    reg = Registry()
    import os

    from lerobot.utils.constants import HF_LEROBOT_HOME

    account = reg.settings.hub_account
    raw_repo, trimmed_repo = f"{account}/{a.name}_raw", f"{account}/{a.name}"
    raw_root, trimmed_root = HF_LEROBOT_HOME / raw_repo, HF_LEROBOT_HOME / trimmed_repo
    fps = next(iter(reg.current_rig().cameras.values())).fps
    done = 0
    if a.resume:
        info = raw_root / "meta" / "info.json"
        if not info.is_file():
            print(f"lab record: --resume, but {raw_root} has no recording", file=sys.stderr)
            return 1
        done = json.loads(info.read_text())["total_episodes"]
        if done >= a.episodes:
            print(f"lab record: {raw_repo} already has {done} of {a.episodes} episodes; nothing to record",
                  file=sys.stderr)
            return 1
        print(f"{raw_repo}: {done} episodes recorded, {a.episodes - done} to go (total {a.episodes})", flush=True)
    display = a.display if a.display is not None else bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    flags = {**rig_flags(reg), "dataset.repo_id": raw_repo,
             "dataset.root": str(raw_root),  # the default location, explicit: 0.7's resume requires a root
             "dataset.single_task": a.task, "dataset.fps": fps,
             "dataset.num_episodes": a.episodes - done, "dataset.episode_time_s": a.episode_time,
             "dataset.reset_time_s": a.reset_time, "dataset.push_to_hub": False, "resume": a.resume,
             "dataset.no_stamp": True,  # lab manages the names (0.7 otherwise appends the date and time)
             **video.encoder_flags(),  # the same AV1 settings for every recording (drtc.data.video)
             "display_data": display, "play_sounds": False}
    cmd = ["lerobot-record", *(f"--{k}={flag_value(v)}" for k, v in flags.items()), *extra]
    print(" \\\n    ".join(shlex.quote(p) for p in cmd), flush=True)
    print(f"\nrecord {raw_repo} -> trim -> {trimmed_repo}" + ("" if a.no_push else " -> push both"))
    from drtc.cli import check_rig

    problems = check_rig(reg)
    if trimmed_root.exists() and not a.resume:
        problems.append(f"{trimmed_root} exists: the trimmed copy is made after recording (choose a new name, "
                        f"or --resume to add episodes)")
    rebuild = trimmed_root.with_name(trimmed_root.name + ".rebuild")
    if a.resume and rebuild.exists():
        problems.append(f"{rebuild} exists (an interrupted earlier rebuild?); look at it, then remove it")
    if a.dry_run:
        for p in problems:
            print(f"  PROBLEM {p}")
        return 0
    if problems:
        for p in problems:
            print(f"  PROBLEM {p}", file=sys.stderr)
        return 1

    from lerobot.scripts import lerobot_record

    sys.argv = cmd
    lerobot_record.main()

    issues = check_recording(raw_root, fps)
    for p in issues:
        print(f"  CHECK {p}")
    if issues:
        print("lab record: the recording has problems; not trimming or pushing. Look at them first.", file=sys.stderr)
        return 1
    from drtc.data import trim

    ds = trim.RawDataset(raw_root)
    results, noise, floor = trim.detect(ds, trim.Params())
    trim.report(ds, results, noise, floor, trim.Params())
    if trimmed_root.exists():
        # --resume: the trimmed copy is rebuilt from the whole recording; the old one (also on the Hub, in the
        # repo's history) is replaced only after the rebuild succeeded. Its videos are hardlinks into the raw copy.
        import shutil

        trim.apply(ds, results, rebuild)
        shutil.rmtree(trimmed_root)
        rebuild.rename(trimmed_root)
        print(f"rebuilt the trimmed copy from all {len(results)} episodes: {trimmed_root}")
    else:
        trim.apply(ds, results, trimmed_root)
    if a.no_push:
        print(f"kept local: {raw_root}, {trimmed_root}")
        return 0
    from lerobot.datasets import LeRobotDataset

    for repo, root in ((raw_repo, raw_root), (trimmed_repo, trimmed_root)):
        LeRobotDataset(repo, root=root).push_to_hub(allow_patterns=["data/*", "meta/*", "videos/*"])
        print(f"pushed https://huggingface.co/datasets/{repo}")
    entries = registry_entries(reg, a.name, raw_repo, trimmed_repo, a.task, fps, a.held_out)
    path = reg.lab / "datasets" / REGISTRY_FILE
    existing = yaml.safe_load(path.read_text()) if path.is_file() else {}
    existing.update(entries)
    path.write_text("# Written by lab record; review and commit.\n\n" + yaml.safe_dump(existing, sort_keys=False))
    print(f"registry: added {', '.join(entries)} to {path} (review and commit)")
    # Ready for every recipe in use: relative-action stats etc. (lab train would also make them on first use).
    from drtc.data.prepare import prepare, recipes_in_use

    reg = Registry()
    prepare(reg, a.name, recipes_in_use(reg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
