"""Hardware and connectivity checks before a session.

    run     (default) DRTC client: follower, cameras, policy server, clock   -- Pi
    record  teleoperated demos: follower + leader, cameras                    -- laptop or Pi

Usage:
    python -m lab.tools.preflight [run|record]
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
from pathlib import Path

from lab.tools.config import device_present, load_rig

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
        if "video10" in str(cam["path"]):
            hint = ("phonecam not running: start lab/bin/session" if Path(cam["path"]).exists()
                    else "v4l2loopback not loaded: start lab/bin/session (lab/bin/*-setup loads it at boot)")
        else:
            hint = f"connected cameras: {camera_ids()}"
        return report(False, f"{name}: {e}", hint)


def camera_ids() -> str:
    if os.name == "nt":
        return "list camera indexes with: lerobot-find-cameras opencv"
    by_id = Path("/dev/v4l/by-id")
    links = sorted(p for p in by_id.iterdir() if p.name.endswith("index0")) if by_id.is_dir() else []
    return "; ".join(str(p) for p in links) or "none; check cables and dmesg"


def serial_ids() -> str:
    if os.name == "nt":
        from serial.tools import list_ports

        ports = [f"{p.device} ({p.description})" for p in list_ports.comports()]
        return "; ".join(ports) or "no COM ports; check cables and Device Manager"
    by_id = Path("/dev/serial/by-id")
    links = sorted(by_id.iterdir()) if by_id.is_dir() else []
    return "; ".join(f"{p} -> {p.resolve().name}" for p in links) or "no USB serial devices; check cables and dmesg"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-session hardware checks")
    parser.add_argument("mode", nargs="?", choices=["run", "record"], default="run")
    mode = parser.parse_args().mode
    rig = load_rig()
    results = []

    arms = [("follower", rig["robot"], "robots", "so101_follower")]
    if mode == "record":
        arms.append(("leader", rig["leader"], "teleoperators", "so101_leader"))
    for name, arm, kind, sub in arms:
        results.append(report(device_present(arm["port"]), f"{name} port {arm['port']}", serial_ids()))
        calib = f"{kind}/{sub}/{arm['id']}.json"
        results.append(report((CALIB_DIR / calib).is_file(), f"calibration {calib}", "see lab/README.md"))

    cams = rig["cameras"]
    for name in ("camera1", "camera2"):
        results.append(check_camera(name, cams[name], cams))

    if mode == "record":
        print(f"\n{sum(results)}/{len(results)} checks passed")
        return

    host, sport = rig["server"]["host"], int(rig["server"]["port"])
    try:
        with socket.create_connection((host, sport), timeout=2):
            results.append(report(True, f"policy server {host}:{sport}"))
    except OSError:
        results.append(report(False, f"policy server {host}:{sport}", "start lab/bin/server on the laptop"))

    if host in ("0.0.0.0", "127.0.0.1", "localhost"):
        print("  skip  clock sync (server and client on this machine)")
    elif os.name == "nt":
        print("  skip  clock sync check (no chrony on Windows; RTT is valid without it)")
    elif shutil.which("chronyc"):
        out = subprocess.run(["chronyc", "tracking"], capture_output=True, text=True).stdout
        field = {k.strip(): v.strip() for k, _, v in (l.partition(":") for l in out.splitlines())}
        ref = field.get("Reference ID", "")
        # Reference ID is the source's IPv4 in hex, then its name: "0A2A0001 (arshjeevs)" for 10.42.0.1.
        from_laptop = ref.split(" ")[0] == socket.inet_aton(host).hex().upper() or host in ref
        results.append(report(field.get("Leap status") == "Normal" and from_laptop,
                              f"clock synced to laptop ({ref or 'no source'})",
                              "run lab/bin/laptop-setup (laptop) and lab/bin/pi-setup (Pi); RTT is valid without it"))
    else:
        results.append(report(False, "chrony installed", "run lab/bin/pi-setup"))

    print(f"\n{sum(results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
