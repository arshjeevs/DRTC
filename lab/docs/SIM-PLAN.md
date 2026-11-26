# Stage 3: task success in simulation (smoke test passed 2026-11-26)

Research by Gemini workers on 2026-11-26 (LeRobot 0.7 source, Hugging Face Hub), checked here where it matters.

## Recommendation: go, starting with one smoke test

**What makes it cheap:** LeRobot 0.7 already ships a simulated robot, `SimRobot` (`--robot.type=sim`,
`lerobot/robots/sim_robot/`). It wraps a simulation environment behind the same robot interface and works with
`--inference.type=remote`. So our own client, DRTC engine and fault injector can drive a MuJoCo task unchanged, and
the environment reports task success (`SimRobot.latest.is_success`). No custom bridge is needed.

**The task suite:** LIBERO (MuJoCo through robosuite; LeRobot's `libero` env), with published checkpoints for
policy families this project already uses:

| Checkpoint | Policy | Chunk | Reported success | Where it runs |
|---|---|---|---|---|
| `lerobot/smolvla_libero` | SmolVLA | 50 | ~80-87% | the laptop (about 2 GB) |
| `nvidia/gr00t17-lerobot-libero_spatial-640` (and `_object`, `_goal`, `_10`) | GR00T N1.7 | 16 | 94-98% | the H200 |
| `lerobot/pi05_libero_finetuned` | π0.5 | 50 | ~96-98% | the H200 (gated PaliGemma weights) |

GR00T N1.7's 16-action chunk is the short-chunk case of the report (Section "Short-chunk policies"), now with task
success.

**The experiment (draft):** per policy, LeRobot as shipped / patient / DRTC × no faults / the fault mix, on LIBERO's
tasks with fixed seeds, 10+ episodes per cell. The primary measure is task success; stall and recovery as on the
bench.

## Risks to check first

1. **Install.** `lerobot[libero]` adds `hf-libero` (robosuite, MuJoCo, bddl, hydra). Possible conflicts: numpy 2.x
   with robosuite, `opencv-python` vs our `opencv-python-headless`. It would go in a separate optional extra
   (`mujoco-sim`), so the Pi and A100 installs stay untouched; test it in a scratch environment first.
2. **Control rate.** LIBERO steps at 20 Hz, and `SimRobot` requires the client's fps to match. DRTC's settings are in
   steps (s_min, cooldown), so at 20 Hz they mean different times than on the arm; report this.
3. **Headless rendering.** `MUJOCO_GL=egl` needs `libegl1` on the GPU host (the H200 container: check; the A100 has
   no sudo).
4. **GPU time.** The full experiment needs H200 time; the smoke test runs on the laptop.

## Steps (each needs your OK)

1. Smoke test on the laptop (about 1-2 h): install `lerobot[libero]` in a scratch venv, serve `lerobot/smolvla_libero`,
   run one LIBERO episode through `lerobot-rollout --robot.type=sim --inference.type=drtc` with and without a fault.
2. If it works: the extra in pyproject, three conditions, a small eval plan (frozen like the others), laptop runs for
   SmolVLA; H200 for GR00T N1.7 and π0.5.
3. Report section and a video clip of the simulated arm under faults.

## Smoke test (2026-11-26, laptop RTX 4060): passed

- **Install:** a separate environment (`~/Projects/drtc-sim/.venv`) with the project's locked packages plus
  `hf-libero`: no conflicts (torch 2.14.1+cu130, numpy 2.2.6, MuJoCo 3.8.1, robosuite 1.4.0). LIBERO asks two setup
  questions on first use (answer N). The project's `.venv` is untouched.
- **Serving:** `lerobot-policy-server --config_path=sim/smolvla_libero_server.yaml`. The checkpoint expects
  `camera1`/`camera2` at 256x256 and an 8-number state; the LIBERO env is set to match
  (`camera_name_mapping`, `observation_height/width=256`).
- **Running:** `MUJOCO_GL=egl python sim/smoke.py <remote|shipped|patient|drtc> <none|mix>` (one 14 s episode of
  libero_spatial task 0 at 20 Hz; success read from the env).
- **Result, one episode per cell (a smoke test, not a result):**

| Client | No faults | Spike + drops + outage (bench timing, 250 ms latency) |
|---|---|---|
| LeRobot remote client | success at 3.9 s | (no fault injector in the plain client) |
| LeRobot as shipped (our engine, upstream rules) | success at 3.8 s | run ended at the first fault (starvation grace) |
| LeRobot, patient | success at 3.9 s | run ended (stuck after a lost request; 10 s startup deadline) |
| DRTC | success at 3.8 s | **success at 4.0 s** |

Next, with your OK: a small frozen eval (the three clients x none/mix/drops/outage x 10 LIBERO tasks x a few seeds),
SmolVLA on the laptop first; then GR00T N1.7 (16-action chunks) on the H200.

## The evaluation (plan fixed 2026-11-26, before the runs)

`sim/eval.py`: SmolVLA (`lerobot/smolvla_libero`) on the laptop; libero_spatial tasks 0-9 x seeds 0-1 x faults
none / mix (spike + drops + outage) / drops / outage x LeRobot as shipped / patient / DRTC = 240 episodes of 15 s at
20 Hz, the bench's 250 ms reply latency in all, fault times shifted per seed in [0, 4] s as on the bench, the same
initial scene (seed) for every client. Primary measure: task success within the episode, per client and fault (20
episodes per cell, Wilson intervals; DRTC vs patient compared on the same task and seed). Secondary: stall fraction
from the engine's ticks. Results: `results/sim/libero_smolvla.jsonl`. No other analysis is planned; anything else is
labelled exploratory.
