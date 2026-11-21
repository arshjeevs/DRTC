# Report and presentation material

*Everything the project report and the presentation need, in one place: the story, the setup, what we built, the
method, every result with its numbers, and where each number and figure comes from. Numbers are copied from the
result files named next to them (never estimated); **pending** means not measured yet. Claude keeps this file
current as results come in. The plan and its status: [PLAN.md](PLAN.md).*

**Contents:** [1 Summary](#1-summary) · [2 Problem](#2-problem-and-motivation) · [3 Claim](#3-the-claim) ·
[4 Setup](#4-setup) · [5 System](#5-what-we-built) · [6 Datasets](#6-datasets) · [7 Models](#7-models-and-training) ·
[8 Method](#8-method) · [9 Results](#9-results) · [10 Limitations](#10-limitations) ·
[11 Future work](#11-future-work) · [12 Report and slides](#12-the-report-and-the-presentation) · [13 Sources](#13-where-everything-is) ·
[14 References](#14-references)

---

## 1. Summary

- **Problem.** Vision-language-action models are too large for the robot's own computer, so they run on a GPU
  server and every action depends on the network. When the network fails, the robot stops.
- **Claim.** Network faults lower the task scores of client-server robot inference; **DRTC recovers them**;
  **smoothing** then improves the motion.
- **Main result** (sorting task, SO-101 arm, 10 trials per cell, the laptop as the robot computer, models on an A100):

  | | SmolVLA | GR00T N1.5 |
  |---|---|---|
  | Normal network, LeRobot as shipped | 90% | 70% |
  | + network faults | **0%** | **0%** |
  | + faults, with DRTC | **60%** (p = 0.011 against faults) | **80%** (p = 0.001) |
  | + faults, DRTC + smoothing | 80% | 100% |

  Smoothing cut the jerk of the motion about 5x (SmolVLA, 41.7 to 8.5) and 7.5x (GR00T, 43.2 to 5.8), p < 0.001,
  and GR00T's time without new actions from 65% to 24% of the trial.

## 2. Problem and motivation

- **Models are heavy.** SmolVLA has ~450M parameters, GR00T N1.5 and N1.7 3B, π0.5 ~3.4B. A robot computer (a laptop,
  a Raspberry Pi) cannot run them at the control rate of 30 actions per second, so they run on a remote GPU.
- **Remote inference is fragile.** Each request sends the camera images to the server and gets back a chunk of
  actions. A latency spike, lost replies or an outage empty the robot's action queue and the arm stops.
- **What LeRobot ships.** LeRobot 0.7's remote inference (the baseline here) requests a new chunk when the queued
  actions fall to about one round trip, keeps one request in flight, and **ends the run when no action arrives for
  1 s** (its starvation grace). A short outage therefore ends the task.
- **DRTC** (Distributed Real-Time Chunking, Vial 2026) is a scheduling method for exactly this setting: request
  early (when the queue is down to the chunk length H minus s_min), estimate the latency with the Jacobson-Karels
  estimator used by TCP, and re-request with a fresh observation when a reply is late (the cooldown), so lost and
  late replies are recovered instead of waited for.

## 3. The claim

Locked with the user on 2026-11-20 ([PLAN.md](PLAN.md)):

1. Network faults lower the task score of LeRobot's remote inference (condition 2 against 1).
2. DRTC recovers it (condition 3 against 2).
3. Smoothing then improves the motion (condition 4 against 3).

## 4. Setup

**State in the report and the presentation: the laptop was the robot computer for every experiment.** The Raspberry
Pi was set up afterwards (section 11).

| Part | What |
|---|---|
| Arm | SO-101 follower (6 Feetech servos; joint targets in degrees), and the SO-101 leader arm for recording demonstrations |
| Cameras | wrist: Zebronics Zeb Live Pro, 800 x 600 MJPG, 30 fps; scene: an Android phone through DroidCam over USB, 800 x 600, 30 fps (rig `so101_v2`, [rigs/so101_v2.yaml](rigs/so101_v2.yaml)) |
| Robot computer | the laptop (Linux, RTX 4060 Laptop 8 GB), running the robot, the cameras and the client at 30 Hz |
| Model server | A100 80 GB PCIe (a shared GPU server) for every robot experiment |
| Training, offline tests | H200 141 GB (shared, inside a container) and the A100 |
| Network | laptop to A100 over WiFi and the local network; each request carries both camera images uncompressed (2 x 800 x 600 x 3 bytes = 2.9 MB) |
| Software | LeRobot 0.7.0 (pinned), PyTorch 2.14.1, Python 3.12; our layer on top: the `lab` command ([README.md](README.md)) |

Measured round trip (median of the trial medians, normal network, `sort_one_drtc_main`): SmolVLA **311 ms**, of
which 53 ms is inference on the server; GR00T N1.5 **452 ms**, of which 194 ms is inference. The remaining ~260 ms
is the network and queueing, mostly the uncompressed images.

## 5. What we built

### 5.1 The baseline: LeRobot remote inference as shipped

LeRobot 0.7's policy server and client (Zenoh transport, `ChunkRuntime`), unchanged: request when the queued
playback falls to max(refill time, turnaround + one action); turnaround estimate = the slowest of the last 100
replies; one request in flight; a reply older than its deadline is a fault; **ends the run after 1 s without
actions**.

### 5.2 DRTC on LeRobot 0.7 (our port, [src/drtc/control/engine.py](../src/drtc/control/engine.py))

DRTC's scheduling inside LeRobot's own runtime, so every upstream safety check stays:
- **Trigger:** request when the queued actions are at most H - s_min (H = the last chunk's length).
- **Latency estimate:** Jacobson-Karels (smoothed mean + k x smoothed deviation, as TCP's RFC 6298).
- **Cooldown re-request:** a request not answered within its cooldown is given up and a fresh observation is sent
  (the freshest one wins); duplicate and stale replies are ignored.
- **Holds the arm through a stall and carries on** (no 1 s give-up).

**Our fixes (2026-11-20), each found on the real setup:**

| Problem | Cause | Fix |
|---|---|---|
| GR00T never got a chunk | DRTC caps the cooldown at H/2. GR00T's chunk is 16 actions = 0.53 s, the replies take 0.4-0.7 s, so every request was given up before its reply | the cooldown is never shorter than the measured latency (no cap; DRTC's cap stays as an option) |
| a storm of re-requests | replies close to the estimate were given up; the busy server got slower still | the cooldown also covers the slowest of the last 10 replies |
| the robot never started on a slow link | before the first reply DRTC waits only s_min + epsilon steps | until the first reply, wait up to the start-up timeout |
| "Invalid RTC delay" errors | the requested delay could exceed the server's chunk | the delay is capped at the server's prediction steps |

### 5.3 Smoothing (condition 4)

- **Butterworth filter** ([action_filter.py](../src/drtc/control/action_filter.py)): a causal low-pass per joint
  (order 2, 3 Hz cutoff at 30 Hz) and a gain of 1.4 on each step to make up the filter's amplitude loss. It keeps its
  state across chunks, which smooths the seam between chunks.
- **Adaptive playback** (method A, [playback.py](../src/drtc/control/playback.py)): when the queue would run dry
  before the next chunk arrives, play the actions slower instead of stopping:
  rate = clip(queued / (ticks until the next chunk + margin), min_rate, 1), changing by at most a slew per tick and
  interpolating between actions. With a full queue the rate is 1 (normal playback).

### 5.4 Network fault injection ([faults.py](../src/drtc/control/faults.py))

Faults are injected where a real network would cause them (the client's transport), so the system's own logic
reacts: observation drop, duplicate and reorder; reply drop, duplicate and reorder; latency spikes; disconnects
(both directions); and, on the bench, a constant latency with jitter. Every injected fault is logged (`faults.csv`).

**The fault mix** (conditions 2-4, times from the trial start): a latency spike of +1 s at 2 s; every reply dropped
from 4 to 5 s; a full outage from 6 to 7.5 s.

### 5.5 The evaluation system ([docs/EVALUATION.md](docs/EVALUATION.md))

- An **eval file** fixes the task, the success rule, the models (pinned to their Hub commits) and the conditions
  before the first trial (`lab eval freeze`).
- Trials run in **rounds**: each round runs every model x condition once, in a seeded random order, so drift (light,
  motor heat) affects all alike. The operator presses `s` (success) or `f` (failure) before the eval's time limit
  (30 s for sorting).
- Every trial is stored with its raw files (ticks, chunks, faults, recoveries, video) and pushed to a private
  results store; the report is rebuilt from the raw files (`lab eval report`).
- Statistics: Wilson 95% intervals for success rates; Fisher's exact test (success) and the Mann-Whitney U test
  (everything else) for comparisons.

## 6. Datasets

All recorded on the SO-101 with the leader arm at 30 fps, idle start and end trimmed automatically. The held-out
episodes (seeded split) are never trained on and are used for the offline test (M9).

| Dataset | Task (as given to the models) | Episodes (train + held out) | Frames | Joint units | Role |
|---|---|---|---|---|---|
| `so101_cubebox_v0` | Pick the green cube and place it in the box | 50 (45 + 5) | 12,102 | -100..100 (LeRobot's normalised range) | initial, temporary research (older rig `so101_v1`) |
| `so101_cubebox_v1` | Pick up the green cube and put it inside the box. | 110 (100 + 10) | 25,067 | -100..100 | baseline for all 7 models |
| `so101_sort_one_v1` | Put the cube in the square of the same color. | 88 (80 + 8) | 20,935 | degrees | baseline for all 7 models; **the DRTC experiment** |

## 7. Models and training

Each model is trained with its **original authors' recipe** (fixed; [recipes/](recipes/)), so differences come from
the models, not from our tuning.

| Model | Recipe (source) | Fine-tuning | Steps x batch | Chunk (actions) |
|---|---|---|---|---|
| SmolVLA | the SmolVLA authors (arXiv 2506.01844) | action expert only, VLM frozen | 20k x 64 | 50 |
| GR00T N1.5 | NVIDIA's SO-101 recipe | projector + diffusion transformer, LLM and vision frozen | 10k x 32 | 16 |
| GR00T N1.7 | NVIDIA / LeRobot SO-101 recipe | projector + DiT + VLLN; relative arm actions | 20k x 64 | 16 |
| π0.5 | openpi `pi05_aloha_pen_uncap` | full; relative joints | 20k x 64 | 50 |
| MolmoAct 2 | the MolmoAct 2 authors (arXiv 2605.02881) | LoRA rank 64 on the VLM, connector and action expert trained | 50k x 16 | 30 |
| ACT | Zhao et al. 2023 | full | 100k x 8 | 100 |
| Diffusion Policy | Chi et al. 2023, real-robot config | full, ResNet-18 from scratch | 600 epochs x 64 | 48 predicted, 24 executed |

Training: on the H200 and the A100, logged to wandb (project `drtc-lab`) and Trackio; each model is pushed to the
Hub (`orangemoon599/<model>`) with its checkpoints; the served checkpoint is the one with the lowest held-out loss.
Current status of every training: [PLAN.md](PLAN.md), "Trainings".

## 8. Method

### 8.1 Conditions (the same on every task)

| # | Condition | Network | Robot side |
|---|---|---|---|
| 1 | baseline | normal | LeRobot remote inference as shipped (no DRTC, no filter, ends the run after 1 s without actions) |
| 2 | baseline + faults | the fault mix | as 1 |
| 3 | DRTC + faults | the fault mix | DRTC with our fixes; no filter; holds through stalls |
| 4 | DRTC + smoothing + faults | the fault mix | as 3 + the Butterworth filter + adaptive playback |

### 8.2 What runs where

| Dataset | Models | Conditions |
|---|---|---|
| `so101_sort_one_v1` | SmolVLA, GR00T N1.5 | 1-4 (**the DRTC experiment**) |
| `so101_sort_one_v1` | all 7 | 1 (baseline) |
| `so101_cubebox_v1` | all 7 | 1 (the DRTC experiment only if time remains) |
| `so101_cubebox_v0` | SmolVLA, GR00T N1.5 | 1 |
| bench (replay robot) | SmolVLA, GR00T N1.5 on sort_one | 1-4, each fault alone and the mix |

### 8.3 Rules

- **10 valid trials** per condition, per model, per task, in rounds (section 5.5).
- **Success** (M1): the operator's key before the 30 s limit, by the success rule written in the eval file before
  the first trial (sorting: *the cube is fully inside the square of its colour and the gripper has let go of it*).
- **A fault that ends a trial after the arm has started moving is a failure** (M1 = 0, label `fault`; M1f counts
  them). A fault before the first action is a setup problem and the trial is repeated.
- A trial is voided only for a setup mistake, never for a model's failure; voided trials are listed in the report.

### 8.4 Metrics (full definitions: [METRICS.md](METRICS.md), v2)

| # | Metric | In one line |
|---|---|---|
| M1, M1f, M1t | success rate; ended by a fault; time to success | the task score |
| M2 | stall fraction | % of ticks after the first action with no new action (the arm holds) |
| M3 | stall events, longest stall | how often and how long the arm stopped |
| M4 | turnaround | time from sending an observation to receiving its chunk (median, 95th percentile) |
| M5 | latency estimate error | the scheduler's estimate minus the measured turnaround |
| M6 | server inference time | the server's own time per chunk; M4 - M6 = network + queueing |
| M7 | jerk | RMS of the third derivative of the commanded arm joints (lower = smoother) |
| M8a, M8b | chunk switch disagreement; largest step | the jump between chunks; the largest change between two ticks |
| M9 | held-out action error | offline: mean absolute error of the first 15 predicted actions on held-out episodes |
| M10 | inference time, GPU memory | offline: 200 requests per model, per GPU |

## 9. Results

### 9.1 The DRTC experiment (main result)

`sort_one_drtc_main`, 2026-11-20: 2 models x 4 conditions x 10 trials = 80 valid trials, laptop -> A100, metrics v2.
One more trial was not counted: the arm's motor bus was not found at the start (a setup problem). Source:
`results/evals/sort_one_drtc_main/report.md`.

**Success (M1), Wilson 95% interval, trials ended by a fault (M1f):**

| Condition | SmolVLA | GR00T N1.5 |
|---|---|---|
| 1 baseline | 90% (60-98%), 0/10 | 70% (40-89%), 0/10 |
| 2 baseline + faults | 0% (0-28%), 4/10 | 0% (0-28%), 9/10 |
| 3 DRTC + faults | 60% (31-83%), 0/10 | 80% (49-94%), 0/10 |
| 4 DRTC + smoothing + faults | 80% (49-94%), 0/10 | 100% (72-100%), 0/10 |

**Tests (Fisher's exact):**

| Comparison | SmolVLA | GR00T N1.5 | Meaning |
|---|---|---|---|
| 2 against 1 | 0% vs 90%, p < 0.001 | 0% vs 70%, p = 0.003 | the faults break the shipped system |
| 3 against 2 | 60% vs 0%, p = 0.011 | 80% vs 0%, p = 0.001 | **DRTC recovers the task** |
| 3 against 1 | 60% vs 90%, p = 0.30 | 80% vs 70%, p = 1.0 | under faults, DRTC is not significantly below the fault-free baseline |
| 4 against 3 | 80% vs 60%, p = 0.63 | 100% vs 80%, p = 0.47 | smoothing does not lower success (higher, not significant with 10 trials) |

**Motion and timing, median over trials** (stall = no new action that tick; jerk in 10^3 deg/s^3):

| Condition | Model | M2 stall | M3 longest stall | M4 turnaround (median / p95) | M6 inference | M7 jerk | M8b largest step | M1t time to success |
|---|---|---|---|---|---|---|---|---|
| 1 baseline | SmolVLA | 0.0% | 0.00 s | 311 / 315 ms | 53 ms | 66.8 | 15.1 deg | 5.8 s |
| 2 + faults | SmolVLA | 28.2% | 1.11 s | 313 / 315 ms | 53 ms | 29.1 | 4.2 deg | - |
| 3 DRTC + faults | SmolVLA | 11.4% | 0.64 s | 311 / 314 ms | 53 ms | 41.7 | 7.0 deg | 6.8 s |
| 4 DRTC + smoothing + faults | SmolVLA | 14.6% | 0.74 s | 311 / 317 ms | 53 ms | **8.5** | 4.4 deg | 7.5 s |
| 1 baseline | GR00T N1.5 | 58.7% | 0.89 s | 452 / 469 ms | 194 ms | 52.4 | 15.1 deg | 14.6 s |
| 2 + faults | GR00T N1.5 | 65.7% | 1.34 s | 452 / 457 ms | 193 ms | 39.5 | 8.8 deg | - |
| 3 DRTC + faults | GR00T N1.5 | 64.7% | 2.78 s | 454 / 736 ms | 195 ms | 43.2 | 14.0 deg | 16.3 s |
| 4 DRTC + smoothing + faults | GR00T N1.5 | **23.8%** | 1.64 s | 455 / 475 ms | 196 ms | **5.8** | 6.2 deg | 17.4 s |

**Smoothing (4 against 3, Mann-Whitney):** jerk SmolVLA 41.7 -> 8.5 (p < 0.001), GR00T 43.2 -> 5.8 (p < 0.001);
largest step SmolVLA 7.0 -> 4.4 deg (p = 0.002), GR00T 14.0 -> 6.2 deg (p = 0.001); GR00T stall 64.7% -> 23.8%
(p < 0.001), longest stall 2.78 -> 1.64 s (p = 0.005). Against the fault-free baseline, condition 4 is smoother
too: jerk SmolVLA 66.8 -> 8.5, GR00T 52.4 -> 5.8 (both p < 0.001).

**What the numbers say (for the discussion):**
- GR00T stalls 59% of the time **even without faults**: its chunk (16 actions, 0.53 s) is barely longer than a round
  trip (0.45 s), so the queue empties before the next chunk is ready. Adaptive playback slows down instead of
  stopping, which is why it helps GR00T most (65% -> 24%).
- The low jerk of condition 2 is not smoothness: in those runs the arm stood still for long stretches or the run
  ended early (the system gave up), so there was less motion to measure.
- Under faults with DRTC, GR00T's slow replies show in the 95th percentile turnaround (736 ms) and in a latency
  overestimate (M5 +132 ms); with smoothing both return to normal (475 ms, +18 ms).
- The turnaround is ~260 ms longer than the inference for both models: the network share, mostly the 2.9 MB of
  uncompressed images per request.

**Figures to make** (raw files in `results/evals/sort_one_drtc_main/sessions/*/`):
1. Success per condition and model, bars with the Wilson intervals (the headline slide).
2. Jerk (M7) per trial, conditions 3 and 4, both models (box or dot plot).
3. One trial's commanded joint angles over time (`ticks.csv`, `cmd.*`) for conditions 2, 3 and 4, with the fault
   windows shaded (2 s, 4-5 s, 6-7.5 s): the arm stopping, recovering, and moving smoothly.
4. Stall fraction (M2) per condition for GR00T.
5. Videos: `.../trials/<trial>/video/{wrist,scene}.mp4`, one per condition, for the presentation.

### 9.2 The bench: each fault on its own (replay robot)

The four conditions on the replay robot (held-out episodes played back as the camera input, no arm), with the
network simulated as measured on the robot (250 +- 20 ms on every reply), for each fault of the mix alone (spike,
drops, outage) and the whole mix. 2 models x 4 conditions x 8 trials per fault type. Replay: M2-M8 only (no task
success). Run on the A100, 2026-11-20; source: `results/bench/bench_main_{spike,drops,outage,mix}/report.md`.

Conditions as in 8.1 (1 baseline, 2 baseline + fault, 3 DRTC + fault, 4 DRTC + smoothing + fault); medians over
8 runs; jerk in 10^3 deg/s^3:

| Fault alone | Model | Ended by a fault (2 / 3 / 4) | Stall % (1 / 2 / 3 / 4) | Jerk (1 / 2 / 3 / 4) |
|---|---|---|---|---|
| Latency spike | SmolVLA | 8/8 / 0/8 / 0/8 | 0.0 / 26.9 / 7.1 / 7.5 | 54.1 / 30.1 / 78.7 / 11.5 |
| Latency spike | GR00T N1.5 | 8/8 / 0/8 / 0/8 | 52.2 / 60.3 / 54.9 / 15.8 | 158.2 / 97.3 / 113.6 / 19.2 |
| Dropped replies | SmolVLA | 8/8 / 0/8 / 0/8 | 0.0 / 20.8 / 4.5 / 4.3 | 53.3 / 35.8 / 66.3 / 14.9 |
| Dropped replies | GR00T N1.5 | 7/8 / 0/8 / 0/8 | 52.0 / 56.3 / 55.5 / 22.7 | 154.5 / 111.7 / 97.4 / 19.6 |
| Outage | SmolVLA | 8/8 / 0/8 / 0/8 | 0.0 / 14.2 / 9.1 / 8.7 | 55.3 / 53.1 / 79.1 / 12.6 |
| Outage | GR00T N1.5 | 8/8 / 0/8 / 0/8 | 51.6 / 54.7 / 55.1 / 22.5 | 132.6 / 119.3 / 191.7 / 22.2 |
| The whole mix | SmolVLA | 8/8 / 0/8 / 0/8 | 0.0 / 26.7 / 20.6 / 19.5 | 54.6 / 36.0 / 47.7 / 17.5 |
| The whole mix | GR00T N1.5 | 8/8 / 0/8 / 0/8 | 52.0 / 63.0 / 66.0 / 46.4 | 158.3 / 74.4 / 188.8 / 16.1 |

**What it shows:** each fault alone, even one 1-second latency spike, is enough for the shipped client to end the run
(7-8 of 8); under DRTC no run is ended by any fault type; smoothing gives the lowest jerk under every fault type, for
both models. GR00T N1.5 stalls about half the time even with no fault (its short chunk), as on the robot. The bench
jerk is higher than the robot's (replayed observations, not the arm's own), so compare bench numbers only with each
other. Figure: `report/generated/fig_bench.pdf`.

### 9.3 Offline: held-out error (M9) and inference cost (M10)

Measured on the H200 (shared with two running trainings; counted as a realistic server), 2026-11-20. M9 on the
served checkpoint and on the last one; sort_one in degrees, cubebox in -100..100 units, so compare within a dataset
only. Source: `results/offline/<dataset>/<model>.json` (results store `offline/`).

| Dataset | Model | M9 arm (served) | M9 arm (last checkpoint) | M9 gripper | M10 median | M10 p95 | GPU memory |
|---|---|---|---|---|---|---|---|
| sort_one | SmolVLA | 2.13 deg | 2.13 | 2.32 | 97 ms | 125 ms | 1.0 GB |
| sort_one | GR00T N1.5 | 3.03 deg | 3.03 | 2.86 | 167 ms | 235 ms | 5.7 GB |
| sort_one | ACT | 3.20 deg | 3.20 | 2.50 | 42 ms | 68 ms | 0.3 GB |
| sort_one | Diffusion | 2.43 deg | 2.43 | 2.88 | 179 ms | 209 ms | 1.3 GB |
| cubebox v0 | SmolVLA | 2.78 | 2.45 | 3.52 | 99 ms | 122 ms | 1.0 GB |
| cubebox v0 | GR00T N1.5 | 3.37 | 3.37 | 4.32 | 183 ms | 268 ms | 5.7 GB |
| cubebox v1 | SmolVLA | 2.98 | 2.98 | 2.90 | 96 ms | 118 ms | 1.0 GB |
| cubebox v1 | GR00T N1.5 | 4.28 | 3.72 | 3.20 | 180 ms | 233 ms | 5.7 GB |

Held out: sort_one 8 episodes (180 predictions), cubebox v0 5 (124), cubebox v1 10 (212). On sort_one SmolVLA has the
lowest error, ACT is the fastest and smallest; for two models (SmolVLA cubebox v0, GR00T N1.5 cubebox v1) the last
checkpoint has a lower error than the one served (the lowest held-out loss), so the two criteria don't always agree.
M10 on the busy H200 is higher than M6 on the A100 during the robot runs (SmolVLA 97 vs 53 ms). Figure:
`report/generated/fig_offline.pdf`.

The other models are added when they are trained (GR00T N1.7, π0.5, MolmoAct 2 on both tasks; ACT and Diffusion on
cubebox v1).

### 9.4 Baselines on every model (condition 1)

**pending** (run by the user as the trainings finish; [PLAN.md](PLAN.md) step 5):

| Dataset | Models | Trials | Status |
|---|---|---|---|
| `so101_cubebox_v0` | SmolVLA, GR00T N1.5 | 20 | both trained; eval `cubebox_v0_baseline` frozen 2026-11-20, ready for trials (laptop) |
| `so101_cubebox_v1` | all 7 | 70 | 2 trained, 2 training, 3 to do |
| `so101_sort_one_v1` | all 7 | 70 (SmolVLA and GR00T N1.5 already have 10 each from 9.1) | 4 trained, 1 training, 2 to do |

### 9.5 Side results (before the scope was locked, 2026-11-20)

Valid, but under other settings, so shown separately (`results/archive/2026-11-20-before-scope/`, metrics v1, 20
trials):
- `sort_one_baseline_v2`, DRTC with its filter, normal network (80 trials): SmolVLA 80%, Diffusion 80%, ACT 75%,
  GR00T N1.5 20% (an earlier version of the scheduler; the eval was restarted once after GR00T fixes).
- `sort_one_faults` (DRTC with its filter under spikes and outages): stopped at 81 of 180 trials when the scope
  changed.
- The minor project's results (DRTC replication on the earlier setup): `results/archive/minor/`.

## 10. Limitations

- **10 trials per cell:** the intervals are wide (60% is 31-83%), so only large differences are significant; the
  smoothing gain in success (60 -> 80%, 80 -> 100%) is a trend, not a proven effect.
- **One task, two models** for the DRTC experiment (sorting; SmolVLA and GR00T N1.5).
- **The operator judges success** (not blinded); the success rule was written before the first trial.
- **Shared GPU servers:** other users' jobs run on the A100 and H200, so inference times include contention.
- **Uncompressed images** make the round trip ~260 ms longer than the inference; compressing them (JPEG) would
  shorten it.
- **The fault mix is fixed in time** (the same windows in every trial), unlike a real network.
- **The bench** uses a replay robot and a simulated latency, so it has no task success.
- **Metrics changed once** (v1 -> v2, 2026-11-20, before the main experiment): a trial ended by a fault now counts as a
  failure instead of being repeated.

## 11. Future work

- **The Raspberry Pi 5 as the robot computer** (edge setup): set up and tested 2026-11-20 (CPU PyTorch, the same
  `lab` commands; it can use the A100, the H200 or the laptop's GPU as server). First test (`cubebox_v0_baseline
  --test`): 29.4-29.9 Hz control, no video frames dropped, but the round trip about doubles (SmolVLA 594 ms, GR00T
  N1.5 855 ms, over WiFi to the laptop's hotspot and on to the A100), so the shipped client starved GR00T N1.5 even
  without faults: the edge case DRTC and image compression are for. Official edge trials: not run yet.
- The DRTC experiment on the cube-in-box task (`so101_cubebox_v1`), if time remains.
- Compressed images (JPEG) to cut the round trip.
- RTC in-painting (Black et al. 2025) for the flow-matching models on top of DRTC.
- An edge fallback: a small model (ACT) on the robot computer takes over while the server is unreachable.

## 12. The report and the presentation

**Report (LaTeX):** `report/` (`make` there: results -> numbers, tables, figures -> `report/report.pdf`). How to
add things: [report/README.md](../report/README.md). Photos to add: [report/assets/README.md](../report/assets/README.md).

**Presentation:** the Slides deck https://claude.ai/artifact/JsUD9Vp56hRnTS5qU7VGcJ (private to the owner; 16:9;
plays videos; downloads as PowerPoint or PDF). Its figures are the report's, at slide size
(`report/generated/slides/fig_*.png`, uploaded to the deck); after `make` in `report/`, Claude re-uploads the changed
ones and points the slides at them.

| # | Slide (id) | Content | Placeholders |
|---|---|---|---|
| 1 | cover | title, names, institution | names, guide, institution, date |
| 2 | problem | why remote inference; the network fails | video: a condition-2 trial |
| 3 | remote | diagram: robot computer, network, GPU server; round-trip numbers | |
| 4 | failure | the 1 s give-up rule; the fault mix timeline | |
| 5 | claim | the three-part claim, one line per click | |
| 6 | drtc | trigger, estimator, cooldown; our four fixes | |
| 7 | smoothing | Butterworth filter, adaptive playback | |
| 8 | setup | hardware facts; **the laptop as the robot computer** | photo: the setup |
| 9 | design | the four conditions in the figures' colours; 80 trials | |
| 10 | success | figure: success per condition; 0% -> 60% / 80% | |
| 11 | videos | the same faults, three ways | videos: conditions 2, 3, 4 |
| 12 | motion | figure: jerk; 5-7.5x smoother; GR00T stalls 65% -> 24% | |
| 13 | timeline | figure: one trial per condition over time | |
| 14 | bench | figure: each fault alone; 8 of 8 vs 0 of 8 | |
| 15 | offline | figure: M9/M10; baselines in progress | (the baselines, when run) |
| 16 | limits | limits and next steps | |
| 17 | close | the three results, one per click; questions | |

The trial videos exist for every robot trial (`results/evals/<eval>/sessions/*/trials/<cond>/<model>/tNN/video/
{scene,wrist}.mp4`), so the video placeholders can be filled from real trials.

## 13. Where everything is

| What | Where |
|---|---|
| Robot trials (raw files: ticks, chunks, faults, recoveries, video) | `results/evals/<eval>/sessions/<session>/...`; the report: `results/evals/<eval>/report.md` (`lab eval report <eval> --pull`) |
| Bench | `results/bench/<bench>/` |
| Offline M9/M10 | `results/offline/<dataset>/<model>.json` |
| All official results (one copy) | private Hub dataset `orangemoon599/drtc-results` |
| Earlier results | `results/archive/` |
| Models | Hub `orangemoon599/<model>` (every checkpoint) |
| Training curves | wandb project `drtc-lab`, Trackio |
| The conditions | [experiments/main/](experiments/main/), the bench's [experiments/bench_main/](experiments/bench_main/) |
| Eval plans (frozen) | [evals/](evals/) |
| Code | [src/drtc/control/](../src/drtc/control/) (DRTC, faults, smoothing), [src/drtc/evals/](../src/drtc/evals/) (evaluation) |
| Metrics | [METRICS.md](METRICS.md) |

## 14. References

- J. Vial, *Distributed Real-Time Chunking*, technical report, 2026.
  https://jackvial.com/posts/distributed-real-time-chunking.html
- K. Black et al., *Real-Time Execution of Action Chunking Flow Policies*, 2025 (RTC).
- M. Shukor et al., *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics*, arXiv
  2506.01844, 2025.
- NVIDIA, *GR00T N1.5* and *GR00T N1.7*, github.com/NVIDIA/Isaac-GR00T.
- Physical Intelligence, *π0.5*, github.com/Physical-Intelligence/openpi.
- *MolmoAct 2*, arXiv 2605.02881.
- T. Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT), arXiv 2304.13705, 2023.
- C. Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, arXiv 2303.04137, 2023.
- Hugging Face, *LeRobot* 0.7, github.com/huggingface/lerobot.
- V. Paxson et al., *Computing TCP's Retransmission Timer*, RFC 6298, 2011 (the Jacobson-Karels estimator).
- M. Shapiro et al., *Conflict-free Replicated Data Types*, 2011 (DRTC's last-write-wins registers).
