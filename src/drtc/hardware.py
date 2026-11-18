"""Hardware helpers on the current rig: `lab rig <key>` (values for shell scripts) and `lab calibrate`."""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime

from drtc.registry import Registry, for_host, resolve_device

# Old rig.yaml keys still used by shell scripts (phonecam, session, laptop-setup) -> the value from the new rig.
def rig_value(reg: Registry, key: str):
    rig = reg.current_rig()
    host = reg.current_host()
    hid = host.id if host else None
    first = next(iter(rig.cameras.values()))
    simple = {
        "cameras.width": first.width, "cameras.height": first.height, "cameras.fps": first.fps,
        "robot.type": rig.robot.type.removesuffix("_follower"), "robot.id": rig.robot.id,
        "robot.port": resolve_device(for_host(rig.robot.port, hid)),
        "leader.id": rig.leader.id if rig.leader else None,
        "leader.port": resolve_device(for_host(rig.leader.port, hid)) if rig.leader else None,
        "server.port": reg.network.legacy_server.get("port"),
    }
    if key in simple:
        return simple[key]
    if key.startswith("phonecam.") and rig.phonecam:
        return rig.phonecam[key.split(".", 1)[1]]
    raise KeyError(f"unknown rig key {key!r} (known: {', '.join(sorted(simple))}, phonecam.*)")


def cmd_rig(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: lab rig <key>   e.g. lab rig phonecam.device", file=sys.stderr)
        return 1
    try:
        print(rig_value(Registry(), argv[0]))
    except KeyError as e:
        print(f"lab rig: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_calibrate(argv: list[str]) -> int:
    """Calibrate an arm with the rig's port and its explicit calibration file (the old one is backed up)."""
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

    if len(argv) != 1 or argv[0] not in ("follower", "leader"):
        print("usage: lab calibrate follower|leader", file=sys.stderr)
        return 1
    reg = Registry()
    rig = reg.current_rig()
    host = reg.current_host()
    hid = host.id if host else None
    arm = rig.robot if argv[0] == "follower" else rig.leader
    kind = "robot" if argv[0] == "follower" else "teleop"
    port = resolve_device(for_host(arm.port, hid))
    cal = HF_LEROBOT_CALIBRATION / arm.calibration
    if cal.exists():
        backup = cal.with_name(f"{cal.name}.{datetime.now():%Y%m%d-%H%M%S}.bak")
        shutil.copy2(cal, backup)
        print(f"backed up {cal} -> {backup}")
    cmd = ["lerobot-calibrate", f"--{kind}.type={arm.type}", f"--{kind}.port={port}", f"--{kind}.id={arm.id}",
           f"--{kind}.calibration_dir={cal.parent}"]
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)
