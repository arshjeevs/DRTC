#!/usr/bin/env python3
"""Pre-run checks for the lab setup (run on the Raspberry Pi before every session).

Reads robot/camera settings from the experiment YAML so there is one source of truth.

Usage:
    python scripts/lab/preflight.py
    python scripts/lab/preflight.py --config examples/experiments/configs/laptop_baseline.yaml --server 10.42.0.1:8080
"""

import argparse
import shutil
import socket
import subprocess
from pathlib import Path

import yaml

DEFAULT_CONFIG = "examples/experiments/configs/laptop_baseline.yaml"
CALIB_DIR = Path.home() / ".cache/huggingface/lerobot/calibration"

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, hint: str = "") -> bool:
    results.append((ok, name, hint))
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + ("" if ok or not hint else f"\n         -> {hint}"))
    return ok


def check_camera(name: str, path: str, cfg: dict, fourcc: str | None) -> None:
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    w, h = cfg.get("camera_width", 800), cfg.get("camera_height", 600)
    try:
        cam = OpenCVCamera(
            OpenCVCameraConfig(index_or_path=path, width=w, height=h, fps=cfg.get("camera_fps", 30), fourcc=fourcc)
        )
        cam.connect()
        frame = cam.read()
        cam.disconnect()
        check(frame.shape == (h, w, 3), f"{name} ({path}) -> {frame.shape}", f"expected ({h}, {w}, 3)")
    except Exception as e:  # noqa: BLE001
        hint = "is scripts/lab/start_phone_camera.sh running?" if "video10" in path else "is the camera plugged in?"
        check(False, f"{name} ({path}): {e}", hint)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab preflight checks")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--server", default="10.42.0.1:8080", help="Policy server host:port")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    print(f"Preflight using {args.config}\n")

    print("Robot")
    port = cfg.get("robot_port", "/dev/ttyACM0")
    check(Path(port).exists(), f"follower port {port}", "run lerobot-find-port; ttyACM0/1 can swap after re-plugging")
    calib = CALIB_DIR / "robots/so101_follower" / f"{cfg.get('robot_id')}.json"
    check(calib.is_file(), f"follower calibration {calib}", "copy follower.json from the laptop (docs/LAB_SETUP.md)")
    leader_calib = CALIB_DIR / "teleoperators/so101_leader/leader.json"
    check(leader_calib.is_file(), f"leader calibration {leader_calib} (needed for recording only)",
          "copy leader.json from the laptop (docs/LAB_SETUP.md)")

    print("\nCameras")
    check_camera("camera1", cfg["camera1_path"], cfg, cfg.get("camera_fourcc"))
    check_camera("camera2", cfg["camera2_path"], cfg, None)

    print("\nNetwork")
    host, _, port_s = args.server.partition(":")
    try:
        with socket.create_connection((host, int(port_s or 8080)), timeout=2):
            check(True, f"policy server reachable at {args.server}")
    except OSError as e:
        check(False, f"policy server at {args.server}: {e}",
              "start ./scripts/lab/start_laptop_server.sh on the laptop; is the Pi on the laptop hotspot?")

    print("\nClock")
    if shutil.which("chronyc"):
        out = subprocess.run(["chronyc", "tracking"], capture_output=True, text=True).stdout
        line = next((l for l in out.splitlines() if l.startswith("System time")), "")
        ref = next((l for l in out.splitlines() if l.startswith("Reference ID")), "")
        synced = bool(line) and host in ref
        check(synced, f"chrony: {line.split(':', 1)[-1].strip() or 'no data'} ({ref.split(':', 1)[-1].strip()})",
              "laptop must run chrony as a server (docs/LAB_SETUP.md); round-trip latency is still valid without it")
    else:
        check(False, "chrony not installed", "run ./scripts/lab/pi_setup_once.sh")

    failed = [r for r in results if not r[0]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed." + (" Ready." if not failed else ""))


if __name__ == "__main__":
    main()
