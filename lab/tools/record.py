"""Record teleoperated demonstrations with the leader arm, using rig.yaml for all hardware.

Runs on whichever machine the arms and cameras are plugged into. The Rerun viewer opens
automatically when a display is available (laptop), and stays off over SSH (Pi).

Usage:
    python -m lab.tools.record <hf_user>/<dataset> --task "Pick up the cube ..." --episodes 50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from lab.tools.config import load_rig


def main() -> None:
    parser = argparse.ArgumentParser(description="Record demonstrations (lerobot-record with rig.yaml)")
    parser.add_argument("repo_id", help="<hf_user>/<dataset_name>")
    parser.add_argument("--task", required=True, help="One fixed language instruction for every episode")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--episode-time", type=float, default=30)
    parser.add_argument("--reset-time", type=float, default=10)
    parser.add_argument("--resume", action="store_true", help="Append to an existing dataset")
    parser.add_argument("--no-push", action="store_true", help="Keep the dataset local (no Hugging Face upload)")
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=None,
                        help="Rerun viewer (default: on when a display is available)")
    args = parser.parse_args()
    display = args.display if args.display is not None else bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    rig = load_rig()
    cams = rig["cameras"]

    def cam_spec(name: str) -> str:
        cam = cams[name]
        fourcc = f", fourcc: {cam['fourcc']}" if cam.get("fourcc") else ""
        return (f"{name}: {{type: opencv, index_or_path: {cam['path']}, "
                f"width: {cams['width']}, height: {cams['height']}, fps: {cams['fps']}{fourcc}}}")

    cmd = [
        "lerobot-record",
        f"--robot.type={rig['robot']['type']}_follower",
        f"--robot.port={rig['robot']['port']}",
        f"--robot.id={rig['robot']['id']}",
        f"--robot.cameras={{{cam_spec('camera1')}, {cam_spec('camera2')}}}",
        f"--teleop.type={rig['robot']['type']}_leader",
        f"--teleop.port={rig['leader']['port']}",
        f"--teleop.id={rig['leader']['id']}",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.single_task={args.task}",
        f"--dataset.fps={cams['fps']}",
        f"--dataset.num_episodes={args.episodes}",
        f"--dataset.episode_time_s={args.episode_time}",
        f"--dataset.reset_time_s={args.reset_time}",
        f"--dataset.push_to_hub={str(not args.no_push).lower()}",
        f"--resume={str(args.resume).lower()}",
        f"--display_data={str(display).lower()}",
        "--play_sounds=false",  # voice prompts need spd-say; useless over SSH anyway
    ]
    # Venv bin first: lerobot-record and the Rerun viewer binary resolve without activating the venv.
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    print(" ".join(cmd), flush=True)  # flush: execvp discards buffered output
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
