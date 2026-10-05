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
        if "video10" in cam["path"]:
            hint = ("phonecam not running: start lab/bin/session" if Path(cam["path"]).exists()
                    else "v4l2loopback not loaded: start lab/bin/session, or lab/bin/pi-setup to load it at boot")
        else:
            hint = f"connected cameras: {camera_ids()}"
        return report(False, f"{name}: {e}", hint)


def camera_ids() -> str:
    by_id = Path("/dev/v4l/by-id")
    links = sorted(p for p in by_id.iterdir() if p.name.endswith("index0")) if by_id.is_dir() else []
    return "; ".join(str(p) for p in links) or "none; check cables and dmesg"


def serial_ids() -> str:
    by_id = Path("/dev/serial/by-id")
    links = sorted(by_id.iterdir()) if by_id.is_dir() else []
    return "; ".join(f"{p} -> {p.resolve().name}" for p in links) or "no USB serial devices; check cables and dmesg"


def main() -> None:
    rig = load_rig()
    results = []

    port = rig["robot"]["port"]
    results.append(report(Path(port).exists(), f"follower port {port}", serial_ids()))
    results.append(report(Path(rig["leader"]["port"]).exists(), f"leader port {rig['leader']['port']}",
                          f"needed for recording only. {serial_ids()}"))
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
        field = {k.strip(): v.strip() for k, _, v in (l.partition(":") for l in out.splitlines())}
        ref = field.get("Reference ID", "")
        results.append(report(field.get("Leap status") == "Normal" and host in ref,
                              f"clock synced to laptop ({ref or 'no source'})",
                              "run lab/bin/laptop-setup (laptop) and lab/bin/pi-setup (Pi); RTT is valid without it"))
    else:
        results.append(report(False, "chrony installed", "run lab/bin/pi-setup"))

    print(f"\n{sum(results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
