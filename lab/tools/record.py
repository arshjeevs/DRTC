"""Record teleoperated demonstrations with the leader arm, using rig.yaml for all hardware.

Runs on whichever machine the arms and cameras are plugged into. The Rerun viewer opens
automatically when a display is available (laptop), and stays off over SSH (Pi).
Controls come from this terminal (lab/tools/keys.py): -> / n next, <- / r re-record, Esc / q stop.

Any other lerobot-record flag is passed through and overrides the generated value for the same key
(`lerobot-record --help` lists them all). --dry-run prints the final command without running it.

Usage:
    python -m lab.tools.record <hf_user>/<dataset> --task "Pick up the cube ..." --episodes 50
    python -m lab.tools.record <hf_user>/<dataset> --task "..." --dataset.video=false --robot.max_relative_target=10
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from lab.tools.config import load_rig


def parse_passthrough(tokens: list[str]) -> dict[str, str]:
    """['--a.b=1', '--c', '2', '--flag'] -> {'--a.b': '1', '--c': '2', '--flag': 'true'}."""
    flags: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            sys.exit(f"Unexpected argument {token!r}: lerobot-record flags look like --key=value")
        if "=" in token:
            key, value = token.split("=", 1)
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            key, value = token, tokens[i + 1]
            i += 1
        else:
            key, value = token, "true"
        flags[key] = value
        i += 1
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record demonstrations (lerobot-record with rig.yaml). "
                    "Unknown --key=value flags are passed to lerobot-record and override defaults.",
        allow_abbrev=False,  # otherwise --dataset.* could be read as an abbreviation of our options
    )
    parser.add_argument("repo_id", help="<hf_user>/<dataset_name>")
    parser.add_argument("--task", required=True, help="One fixed language instruction for every episode")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--episode-time", type=float, default=30)
    parser.add_argument("--reset-time", type=float, default=10)
    parser.add_argument("--resume", action="store_true", help="Append to an existing dataset")
    parser.add_argument("--no-push", action="store_true", help="Keep the dataset local (no Hugging Face upload)")
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=None,
                        help="Rerun viewer (default: on when a display is available)")
    parser.add_argument("--dry-run", action="store_true", help="Print the lerobot-record command and exit")
    args, passthrough = parser.parse_known_args()
    display = args.display if args.display is not None else bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    rig = load_rig()
    cams = rig["cameras"]

    def cam_spec(name: str) -> str:
        cam = cams[name]
        fourcc = f", fourcc: {cam['fourcc']}" if cam.get("fourcc") else ""
        return (f"{name}: {{type: opencv, index_or_path: {cam['path']}, "
                f"width: {cams['width']}, height: {cams['height']}, fps: {cams['fps']}{fourcc}}}")

    flags = {
        "--robot.type": f"{rig['robot']['type']}_follower",
        "--robot.port": rig["robot"]["port"],
        "--robot.id": rig["robot"]["id"],
        "--robot.cameras": f"{{{cam_spec('camera1')}, {cam_spec('camera2')}}}",
        "--teleop.type": f"{rig['robot']['type']}_leader",
        "--teleop.port": rig["leader"]["port"],
        "--teleop.id": rig["leader"]["id"],
        "--dataset.repo_id": args.repo_id,
        "--dataset.single_task": args.task,
        "--dataset.fps": cams["fps"],
        "--dataset.num_episodes": args.episodes,
        "--dataset.episode_time_s": args.episode_time,
        "--dataset.reset_time_s": args.reset_time,
        "--dataset.push_to_hub": str(not args.no_push).lower(),
        "--resume": str(args.resume).lower(),
        "--display_data": str(display).lower(),
        "--play_sounds": "false",  # voice prompts need spd-say; useless over SSH anyway
    }
    overrides = parse_passthrough(passthrough)
    flags.update(overrides)

    cmd = ["lerobot-record", *(f"{key}={value}" for key, value in flags.items())]
    print(" \\\n    ".join(shlex.quote(part) for part in cmd), flush=True)
    if overrides:
        print(f"overrides: {', '.join(overrides)}", flush=True)

    devices = [flags["--robot.port"], flags["--teleop.port"]]
    if "--robot.cameras" not in overrides:
        devices += [cams[name]["path"] for name in ("camera1", "camera2")]
    missing = [d for d in devices if not Path(d).exists()]
    if missing:
        print(f"missing devices: {', '.join(missing)}  (lab/bin/preflight record)", flush=True)
    if args.dry_run:
        return
    if missing:
        sys.exit(1)
    # Venv bin first, so the Rerun viewer binary resolves without activating the venv.
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"

    # Run lerobot-record in-process with terminal key controls (pynput misses keys on Wayland/SSH).
    from lerobot.scripts import lerobot_record

    from lab.tools import keys

    lerobot_record.init_keyboard_listener = keys.init_keyboard_listener
    sys.argv = cmd
    lerobot_record.main()


if __name__ == "__main__":
    main()
