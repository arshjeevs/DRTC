"""Hardware and connectivity checks before a session (run on the Pi).

Usage:
    python -m lab.tools.preflight
"""

from __future__ import annotations

import shutil
import socket
import subprocess
from pathlib import Path

from lab.tools.config import load_rig

CALIB_DIR = Path.home() / ".cache/huggingface/lerobot/calibration"


def report(ok: bool, name: str, hint: str = "") -> bool:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"\n        -> {hint}" if not ok and hint else ""))
    return ok


def check_camera(name: str, cam: dict, fmt: dict) -> bool:
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    w, h = fmt["width"], fmt["height"]
    try:
        c = OpenCVCamera(OpenCVCameraConfig(
            index_or_path=cam["path"], width=w, height=h, fps=fmt["fps"], fourcc=cam.get("fourcc")))
        c.connect()
        shape = c.read().shape
        c.disconnect()
        return report(shape == (h, w, 3), f"{name} {shape}", f"expected ({h}, {w}, 3)")
    except Exception as e:  # noqa: BLE001
        hint = "is lab/bin/phonecam running?" if "video10" in cam["path"] else "is the camera plugged in?"
        return report(False, f"{name}: {e}", hint)


def main() -> None:
    rig = load_rig()
    results = []

    port = rig["robot"]["port"]
    results.append(report(Path(port).exists(), f"follower port {port}", "lerobot-find-port (ttyACM0/1 can swap)"))
    results.append(report(Path(rig["leader"]["port"]).exists(), f"leader port {rig['leader']['port']}",
                          "needed for recording only"))
    for kind, sub, cid in [("robots", "so101_follower", rig["robot"]["id"]),
                           ("teleoperators", "so101_leader", rig["leader"]["id"])]:
        path = CALIB_DIR / kind / sub / f"{cid}.json"
        results.append(report(path.is_file(), f"calibration {kind}/{sub}/{cid}.json", "see lab/README.md"))

    cams = rig["cameras"]
    for name in ("camera1", "camera2"):
        results.append(check_camera(name, cams[name], cams))

    host, sport = rig["server"]["host"], int(rig["server"]["port"])
    try:
        with socket.create_connection((host, sport), timeout=2):
            results.append(report(True, f"policy server {host}:{sport}"))
    except OSError:
        results.append(report(False, f"policy server {host}:{sport}", "start lab/bin/server on the laptop"))

    if shutil.which("chronyc"):
        out = subprocess.run(["chronyc", "tracking"], capture_output=True, text=True).stdout
        ref = next((l.split(":", 1)[1].strip() for l in out.splitlines() if l.startswith("Reference ID")), "")
        results.append(report(host in ref, f"clock synced to laptop ({ref or 'no source'})",
                              "laptop must run chrony as a server; RTT is still valid without it"))
    else:
        results.append(report(False, "chrony installed", "run lab/bin/pi-setup"))

    print(f"\n{sum(results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
