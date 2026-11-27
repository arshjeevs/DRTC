"""Stage 3 smoke test: one LIBERO episode through lerobot-rollout with the sim robot and remote inference.

    python smoke.py remote|drtc [none|mix]

Prints when the task first succeeded (from the env's is_success) and how the run ended.
"""
import json
import sys
import time

client = sys.argv[1] if len(sys.argv) > 1 else "remote"
fault = sys.argv[2] if len(sys.argv) > 2 else "none"
task_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
shift = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
duration = float(sys.argv[6]) if len(sys.argv) > 6 else 14.0
metrics = sys.argv[7] if len(sys.argv) > 7 else None
from libero.libero import benchmark  # noqa: E402

instruction = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(task_id).language

from lerobot.robots.sim_robot import sim_robot as sr  # noqa: E402

state = {"t0": None, "success_at": None, "steps": 0}
_send = sr.SimRobot.send_action


def send_action(self, action):  # latch the first success
    out = _send(self, action)
    if state["t0"] is None:
        state["t0"] = time.monotonic()
    state["steps"] += 1
    try:
        ok = bool(self.latest.is_success[0])
    except Exception:  # noqa: BLE001
        ok = False
    if ok and state["success_at"] is None:
        state["success_at"] = round(time.monotonic() - state["t0"], 2)
    return out


sr.SimRobot.send_action = send_action
_start = sr.SimRobot.start_episode


def start_episode(self, seed_=None):  # the same initial scene for every client
    return _start(self, seed)


sr.SimRobot.start_episode = start_episode

from lerobot.scripts import lerobot_rollout  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

register_third_party_plugins()
argv = ["lerobot-rollout", "--strategy.type=base", "--robot.type=sim", "--robot.env.type=libero",
        "--robot.env.task=libero_spatial", f"--robot.env.task_ids=[{task_id}]", f"--robot.task_id={task_id}",
        "--robot.env.observation_height=256", "--robot.env.observation_width=256",
        '--robot.env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}',
        "--fps=20", f"--duration={duration}", "--play_sounds=false",
        f"--task={instruction}",
        "--inference.endpoint=tcp/127.0.0.1:7960", "--inference.deployment=smolvla-libero",
        "--inference.semantics=libero-delta-v1"]
ENGINE = {  # our engine with LeRobot's own scheduling (as the bench's conditions) or DRTC's
    "drtc": ["--inference.scheduler=drtc", "--inference.estimator=jk", "--inference.cooldown=true",
             "--inference.cooldown_cap=none", "--inference.action_starvation_grace_s=14.0",
             "--inference.action_timeout_s=14.0"],
    "shipped": ["--inference.scheduler=upstream", "--inference.estimator=upstream", "--inference.cooldown=false",
                "--inference.action_starvation_grace_s=1.0"],
    "patient": ["--inference.scheduler=upstream", "--inference.estimator=upstream", "--inference.cooldown=false",
                "--inference.action_starvation_grace_s=14.0", "--inference.action_timeout_s=14.0"],
}
if metrics and client in ENGINE:
    argv.append(f"--inference.metrics_path={metrics}")
if client in ENGINE:
    from drtc.control.engine import install

    install()
    argv += ["--inference.type=drtc", "--inference.mode=chunk", "--inference.chunk_merge=aligned",
             "--inference.s_min=20"] + ENGINE[client]
else:
    argv += ["--inference.type=remote"]
if client in ENGINE and fault == "none":
    argv += [f"--inference.faults_json={json.dumps({'latency': {'base_ms': 250, 'jitter_ms': 20, 'seed': 0}})}"]
if fault != "none":
    parts = {"spike": {"spikes": [{"start_s": 2.0 + shift, "delay_ms": 1000}]},
             "drops": {"drop_action": [{"start_s": 4.0 + shift, "duration_s": 1.0}]},
             "outage": {"disconnect": [{"start_s": 6.0 + shift, "duration_s": 1.5}]}}
    plan = {"latency": {"base_ms": 250, "jitter_ms": 20, "seed": 0}}
    for k in (["spike", "drops", "outage"] if fault == "mix" else [fault]):
        plan.update(parts[k])
    if client not in ENGINE:
        print("faults need the drtc engine (it carries the injector); use: drtc mix", file=sys.stderr)
        sys.exit(2)
    argv += [f"--inference.faults_json={json.dumps(plan)}"]
sys.argv = argv
outcome = "completed"
try:
    lerobot_rollout.main()
except SystemExit as e:
    outcome = f"exit {e.code}"
print(json.dumps({"client": client, "fault": fault, "task": task_id, "seed": seed, "shift": shift,
                  "outcome": outcome, "success_at_s": state["success_at"],
                  "steps": state["steps"]}))
