"""Stage 3 evaluation: task success on LIBERO under network faults, LeRobot as shipped / patient vs DRTC.

    MUJOCO_GL=egl ~/Projects/drtc-sim/.venv/bin/python sim/eval.py [--out results/sim/libero_smolvla.jsonl]

The plan (fixed before the runs, lab/docs/SIM-PLAN.md): SmolVLA (lerobot/smolvla_libero) served on this GPU;
libero_spatial tasks 0-9 x seeds 0-1 x faults none / spike+drops+outage (mix) / drops / outage x the three clients;
15 s per episode at 20 Hz; the bench's 250 ms reply latency in every run; fault times shifted per seed in [0, 4] s
as on the bench (the same shift for every client and fault). The clients run back to back for each (task, seed,
fault), so slow drifts hit all three alike. Primary measure: task success (the env's is_success) within the episode.
Each run is a fresh process; finished runs are skipped when the script is started again.
"""

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = Path.home() / "Projects" / "drtc-sim" / ".venv" / "bin" / "python"
CLIENTS = ["shipped", "patient", "drtc"]
FAULTS = ["none", "mix", "drops", "outage"]
TASKS = range(10)
SEEDS = range(2)
DURATION = 15.0


def shift_for(seed: int) -> float:
    return round(random.Random(f"fault-shift-0-{seed + 1}").uniform(0.0, 4.0), 3)  # as lab's fault_shift


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/sim/libero_smolvla.jsonl")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["task"], r["seed"], r["fault"], r["client"]))
    log_dir = out.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    env = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
    server_log = open(log_dir / "server.log", "w")
    server = subprocess.Popen([str(PY.parent / "lerobot-policy-server"),
                               f"--config_path={HERE / 'smolvla_libero_server.yaml'}"],
                              stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        t0 = time.monotonic()
        while "Policy server ready" not in (log_dir / "server.log").read_text(errors="replace"):
            if server.poll() is not None or time.monotonic() - t0 > 600:
                print("the policy server did not start; see", log_dir / "server.log", file=sys.stderr)
                return 1
            time.sleep(2)
        total = len(TASKS) * len(SEEDS) * len(FAULTS) * len(CLIENTS)
        n = len(done)
        for task in TASKS:
            for seed in SEEDS:
                for fault in FAULTS:
                    for client in CLIENTS:
                        key = (task, seed, fault, client)
                        if key in done:
                            continue
                        name = f"t{task}_s{seed}_{fault}_{client}"
                        cmd = [str(PY), str(HERE / "smoke.py"), client, fault, str(task), str(seed),
                               str(shift_for(seed)), str(DURATION), str(log_dir / f"{name}_ticks.csv")]
                        with open(log_dir / f"{name}.log", "w") as fh:
                            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                           env=env, timeout=240)
                        last = (log_dir / f"{name}.log").read_text(errors="replace").strip().splitlines()
                        try:
                            rec = json.loads(last[-1])
                        except (IndexError, json.JSONDecodeError):
                            rec = {"client": client, "fault": fault, "task": task, "seed": seed, "outcome": "crashed",
                                   "success_at_s": None}
                        with open(out, "a") as fo:
                            fo.write(json.dumps(rec) + "\n")
                        n += 1
                        print(f"{n}/{total} {name}: {rec.get('outcome')} success_at={rec.get('success_at_s')}",
                              flush=True)
                        time.sleep(1.5)  # the server closes the session
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)  # only the server this script started
            server.wait(timeout=20)
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
