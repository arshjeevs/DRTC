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
- **Main result so far: the first DRTC run** (2026-11-20; sorting task, SO-101 arm, 10 trials per cell, the laptop as
  the robot computer, models on an A100). Its fault-free and faulted baselines were LeRobot as shipped, which ends a
  run 1 s after the queue runs dry; the redesigned experiment (setup v2, section 8.5) compares DRTC with a patient
  upstream instead and is pending:

  | | SmolVLA | GR00T N1.5 |
  |---|---|---|
  | Normal network, LeRobot as shipped | 90% | 70% |
  | + network faults | **0%** | **0%** |
  | + faults, with DRTC | **60%** (p = 0.011 against faults) | **80%** (p = 0.001) |
  | + faults, DRTC + smoothing | 80% | 100% |

  Smoothing cut the jerk of the motion about 5x (SmolVLA, 43.9 to 9.2) and 11x (GR00T, 67.9 to 6.3), p < 0.001 (metrics v3),
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

Locked with the user on 2026-11-20 and sharpened with setup v2 on 2026-11-22 ([PLAN.md](PLAN.md)):

1. Network faults lower the task score of client-server inference, even when the client waits instead of giving up
   (setup v2: upstream patient + faults against upstream patient; first run: condition 2 against 1).
2. DRTC recovers it, **more than just waiting does** (setup v2: DRTC against upstream patient; first run: condition 3
   against 2, against a client that gave up).
3. Smoothing then improves the motion without lowering success (setup v2: the stack against DRTC; first run:
   condition 4 against 3).

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

### 5.3 Smoothing

The first run's condition 4 used the first two; setup v2 adds four more and picks a stack from them on the bench
([smoothing.py](../src/drtc/control/smoothing.py)):

- **Butterworth filter** ([action_filter.py](../src/drtc/control/action_filter.py)): a causal low-pass per joint
  (order 2, 3 Hz cutoff at 30 Hz) and a gain of 1.4 on each step to make up the filter's amplitude loss. It keeps its
  state across chunks, which smooths the seam between chunks.
- **Adaptive playback** (method A, [playback.py](../src/drtc/control/playback.py)): when the queue would run dry
  before the next chunk arrives, play the actions slower instead of stopping:
  rate = clip(queued / (ticks until the next chunk + margin), min_rate, 1), changing by at most a slew per tick and
  interpolating between actions. With a full queue the rate is 1 (normal playback).
- **RTC** (Black et al. 2025): the server is told which actions run while it thinks and generates the new chunk to
  continue them, so the seam is smooth at the source. SmolVLA and GR00T N1.5 (flow matching, absolute actions).
- **Temporal ensembling** (ACT's): each queued step becomes the weighted average of every chunk that predicted it.
- **Zero-lag queue smoothing**: the recent executed actions plus the queued future are smoothed forward and backward
  (zero-phase Butterworth), so jitter goes without the causal filter's lag; redone at every accepted chunk.
- **Resume ramp**: after a stall, a minimum-jerk blend from the held pose into the stream instead of a jump. (Its
  soft stop before the stall was dropped after the local check.)
- **The stack** (setup v2): adaptive playback + the resume ramp + RTC or ensembling + zero-lag or Butterworth, chosen
  from the bench by a rule fixed before the bench ran (lowest jerk, unless a method adds > 1 deg of deviation from the
  model's plan or makes stalls or recovery > 10% worse; METRICS.md rule 9).

### 5.4 Network fault injection ([faults.py](../src/drtc/control/faults.py))

Faults are injected where a real network would cause them (the client's transport), so the system's own logic
reacts: observation drop, duplicate and reorder; reply drop, duplicate and reorder; latency spikes; disconnects
(both directions); and, on the bench, a constant latency with jitter. Every injected fault is logged (`faults.csv`).

**The fault mix** (conditions 2-4, times from the trial start): a latency spike of +1 s at 2 s; every reply dropped
from 4 to 5 s; a full outage from 6 to 7.5 s. Setup v2 adds a lasting latency rise (+400 ms for 5 s) and random reply
loss (10%), and moves the timed faults by a seeded random 0-4 s per trial number (the same for trial k in every
condition), so faults also hit the grasp.

### 5.5 The evaluation system ([docs/EVALUATION.md](docs/EVALUATION.md))

- An **eval file** fixes the task, the success rule, the models (pinned to their Hub commits) and the conditions
  before the first trial (`lab eval freeze`).
- Trials run in **rounds**: each round runs every model x condition once, in a seeded random order, so drift (light,
  motor heat) affects all alike. The operator presses `s` (success) or `f` (failure) before the eval's time limit
  (30 s for sorting).
- Every trial is stored with its raw files (ticks, chunks, faults, recoveries, video) and pushed to a private
  results store; the report is rebuilt from the raw files (`lab eval report`).
- Statistics: Wilson 95% intervals for success rates; Fisher's exact test (success) and the Mann-Whitney U test
  (everything else) for comparisons. Since metrics v3 each eval names its comparisons; only its primary ones are
  tested (Holm-corrected), the rest are exploratory.
- A run the system ends after the first action is labelled `fault` from the run's own record, whatever key the
  operator presses. Evals grow: more trials or added models, never other changes.

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

### 8.1 The first run's conditions (2026-11-20; setup v2 replaces them, section 8.5)

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

### 8.5 Setup v2: the redesigned experiment (locked 2026-11-22, pending)

Why: in the first run every faulted baseline trial was ended by LeRobot's own 1 s give-up at the first fault, so
DRTC was compared with a client that gave up by design, and DRTC differed from it in two things at once (its
scheduling and its patience). Details: [PLAN.md](PLAN.md).

| Part | What |
|---|---|
| Baseline | **upstream patient**: LeRobot's scheduler with both give-up limits (1 s starvation grace, 5 s request deadline) raised to the whole trial |
| Bench E1 + E2 | sort_one and cubebox v1, 5 models, 8 runs per cell: upstream patient, as shipped and DRTC under none, spike, drops, outage, mix, rise, loss; DRTC without each of its parts (cooldown, estimator, trigger, our H/2 fix, `s_min` x0.5 / x1.5) |
| Bench E3a, E3b | each smoothing method alone on DRTC + mix; then the chosen stack and the stack without each part |
| Robot E4 | sort_one, SmolVLA + GR00T N1.5, 10 trials: upstream patient (none, mix), DRTC + mix, DRTC + stack + mix |
| Primary comparisons | bench: DRTC against upstream patient per fault (stalls, recovery); robot: the three claims (success) |

Found while building it (local replay checks, ACT and SmolVLA): upstream never re-sends a request, so a lost reply
left patient upstream waiting to the end of the run (55% stalled, longest stall 7.5 s under the mix): replacing a
lost reply is what DRTC's cooldown adds. GR00T N1.7 gets no RTC (relative actions). The first RTC request after a
server start compiles for over 5 s.

### 8.4 Metrics (full definitions: [METRICS.md](METRICS.md), v3 since 2026-11-22)

| # | Metric | In one line |
|---|---|---|
| M1, M1f, M1t | success rate; ended by a fault; time to success | the task score |
| M2 | stall fraction | % of ticks after the first action with no new action (the arm holds) |
| M3 | stall events, longest stall | how often and how long the arm stopped |
| M4 | turnaround | time from sending an observation to receiving its chunk (median, 95th percentile) |
| M5 | latency estimate error | the scheduler's estimate minus the measured turnaround |
| M6 | server inference time | the server's own time per chunk; M4 - M6 = network + queueing |
| M7 | jerk | RMS of the third derivative of the commanded arm joints while the arm moves (lower = smoother) |
| M8a, M8b | chunk switch disagreement; largest step | the jump between chunks; the largest change between two ticks |
| M9 | held-out action error | offline: mean absolute error of the first 15 predicted actions on held-out episodes |
| M10 | inference time, GPU memory | offline: 200 requests per model, per GPU |
| M11 | recovery time (v3) | per fault: from its end until the arm moves again; or "not recovered" |
| M12 | deviation from the plan (v3) | how far smoothing moves the arm off the model's own actions (from 2026-11-22 trials on) |

## 9. Results

### 9.1 The DRTC experiment (main result)

`sort_one_drtc_main`, 2026-11-20: 2 models x 4 conditions x 10 trials = 80 valid trials, laptop -> A100, metrics v2 at the time; every number
below is recomputed with v3 (2026-11-22) from the raw files.
One more trial was not counted: the arm's motor bus was not found at the start (a setup problem). Source:
`results/evals/sort_one_drtc_main/report.md`.

**Success (M1), Wilson 95% interval, trials ended by a fault (M1f):**

| Condition | SmolVLA | GR00T N1.5 |
|---|---|---|
| 1 baseline | 90% (60-98%), 0/10 | 70% (40-89%), 0/10 |
| 2 baseline + faults | 0% (0-28%), 10/10 | 0% (0-28%), 10/10 |
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
| 1 baseline | SmolVLA | 0.0% | 0.00 s | 311 / 315 ms | 53 ms | 66.9 | 15.1 deg | 5.8 s |
| 2 + faults | SmolVLA | 28.2% | 1.11 s | 313 / 315 ms | 53 ms | 34.1 | 4.2 deg | - |
| 3 DRTC + faults | SmolVLA | 11.4% | 0.64 s | 311 / 314 ms | 53 ms | 43.9 | 7.0 deg | 6.8 s |
| 4 DRTC + smoothing + faults | SmolVLA | 14.6% | 0.74 s | 311 / 317 ms | 53 ms | **9.2** | 4.4 deg | 7.5 s |
| 1 baseline | GR00T N1.5 | 58.7% | 0.89 s | 452 / 469 ms | 194 ms | 75.8 | 15.1 deg | 14.6 s |
| 2 + faults | GR00T N1.5 | 65.7% | 1.34 s | 452 / 457 ms | 193 ms | 65.0 | 8.8 deg | - |
| 3 DRTC + faults | GR00T N1.5 | 64.7% | 2.78 s | 454 / 736 ms | 195 ms | 67.9 | 14.0 deg | 16.3 s |
| 4 DRTC + smoothing + faults | GR00T N1.5 | **23.8%** | 1.64 s | 455 / 475 ms | 196 ms | **6.3** | 6.2 deg | 17.4 s |

**Smoothing (4 against 3, Mann-Whitney):** jerk SmolVLA 43.9 -> 9.2 (p < 0.001), GR00T 67.9 -> 6.3 (p < 0.001);
largest step SmolVLA 7.0 -> 4.4 deg (p = 0.002), GR00T 14.0 -> 6.2 deg (p = 0.001); GR00T stall 64.7% -> 23.8%
(p < 0.001), longest stall 2.78 -> 1.64 s (p = 0.005). Against the fault-free baseline, condition 4 is smoother
too: jerk SmolVLA 66.9 -> 9.2, GR00T 75.8 -> 6.3 (both p < 0.001). Jerk is metrics v3 (only while the arm moves,
2026-11-22); under v2 GR00T's long holds had lowered its jerk (e.g. condition 3: 43.2 in v2, 67.9 in v3).

**Recovery (M11, v3), from the end of each fault until the arm moves again** (recovered / counted; median of the
recovered; a run that success ended before a fault ended is not counted for it):

| Condition | Model | spike | drops | outage |
|---|---|---|---|---|
| 2 + faults | SmolVLA | 0/10 | 0/10 | 0/10 |
| 2 + faults | GR00T N1.5 | 0/10 | 0/10 | 0/10 |
| 3 DRTC + faults | SmolVLA | 10/10, 0.95 s | 8/8, 0 s | 2/2, 0.50 s |
| 3 DRTC + faults | GR00T N1.5 | 10/10, 0.42 s | 10/10, 0.66 s | 9/9, 1.45 s |
| 4 DRTC + smoothing + faults | SmolVLA | 10/10, 0.95 s | 10/10, 0 s | 6/6, 0.62 s |
| 4 DRTC + smoothing + faults | GR00T N1.5 | 10/10, 0.69 s | 10/10, 0.49 s | 10/10, 0.65 s |

Upstream as shipped never recovered: its first fault ended every run. DRTC recovered from every fault in every
counted run; with smoothing, GR00T's outage recovery fell from 1.45 to 0.65 s.

**What the numbers say (for the discussion):**
- GR00T stalls 59% of the time **even without faults**: its chunk (16 actions, 0.53 s) is barely longer than a round
  trip (0.45 s), so the queue empties before the next chunk is ready. Adaptive playback slows down instead of
  stopping, which is why it helps GR00T most (65% -> 24%).
- The lower jerk of condition 2 is not smoothness: those runs ended early (the system gave up), so there was less
  motion to measure. (v3 already leaves out the time the arm stood still.)
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
8 runs; jerk in 10^3 deg/s^3 (metrics v3: only while the arm moves):

| Fault alone | Model | Ended by a fault (2 / 3 / 4) | Stall % (1 / 2 / 3 / 4) | Jerk (1 / 2 / 3 / 4) |
|---|---|---|---|---|
| Latency spike | SmolVLA | 8/8 / 0/8 / 0/8 | 0.0 / 26.9 / 7.1 / 7.5 | 54.1 / 35.0 / 81.3 / 12.0 |
| Latency spike | GR00T N1.5 | 8/8 / 0/8 / 0/8 | 52.2 / 60.3 / 54.9 / 15.8 | 212.1 / 149.8 / 156.8 / 20.9 |
| Dropped replies | SmolVLA | 8/8 / 0/8 / 0/8 | 0.0 / 20.8 / 4.5 / 4.3 | 53.3 / 39.9 / 67.5 / 15.1 |
| Dropped replies | GR00T N1.5 | 7/8 / 0/8 / 0/8 | 52.0 / 56.3 / 55.5 / 22.7 | 202.7 / 157.1 / 127.8 / 21.0 |
| Outage | SmolVLA | 8/8 / 0/8 / 0/8 | 0.0 / 14.2 / 9.1 / 8.7 | 55.3 / 57.1 / 82.6 / 13.1 |
| Outage | GR00T N1.5 | 8/8 / 0/8 / 0/8 | 51.6 / 54.7 / 55.1 / 22.5 | 173.5 / 163.7 / 270.6 / 24.7 |
| The whole mix | SmolVLA | 8/8 / 0/8 / 0/8 | 0.0 / 26.7 / 20.6 / 19.5 | 54.6 / 41.8 / 52.5 / 19.5 |
| The whole mix | GR00T N1.5 | 8/8 / 0/8 / 0/8 | 52.0 / 63.0 / 66.0 / 46.4 | 210.0 / 109.0 / 315.6 / 23.0 |

**What it shows:** each fault alone, even one 1-second latency spike, is enough for the shipped client to end the run
(7-8 of 8); under DRTC no run is ended by any fault type; smoothing gives the lowest jerk under every fault type, for
both models. GR00T N1.5 stalls about half the time even with no fault (its short chunk), as on the robot. The bench
jerk is higher than the robot's (replayed observations, not the arm's own), so compare bench numbers only with each
other. Figure: `report/generated/fig_bench.pdf`.

### 9.3 Offline: held-out error (M9) and inference cost (M10)

Measured on the H200 (shared with running trainings; counted as a realistic server), 2026-11-20 and 21. M9 on the
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
| cubebox v1 | ACT | 5.26 | 5.26 | 4.08 | 31 ms | 57 ms | 0.3 GB |
| cubebox v1 | Diffusion | 3.26 | 3.26 | 3.51 | 127 ms | 162 ms | 1.3 GB |

Held out: sort_one 8 episodes (180 predictions), cubebox v0 5 (124), cubebox v1 10 (212). On sort_one SmolVLA has the
lowest error, ACT is the fastest and smallest; for two models (SmolVLA cubebox v0, GR00T N1.5 cubebox v1) the last
checkpoint has a lower error than the one served (the lowest held-out loss), so the two criteria don't always agree.
M10 on the busy H200 is higher than M6 on the A100 during the robot runs (SmolVLA 97 vs 53 ms). Figure:
`report/generated/fig_offline.pdf`.

On cubebox v1, SmolVLA has the lowest error and ACT is again the fastest. The other models are added when their
offline evaluation has run (GR00T N1.7 on both tasks: trained, ready to run; π0.5 and MolmoAct 2 after training).

### 9.4 Baselines on every model

**pending** (run by the user; [PLAN.md](PLAN.md)). Condition: upstream patient, normal network (setup v2's reference).
The evals start with the trained models and grow as the others are trained:

| Dataset | Eval | Models now | Trials | Status |
|---|---|---|---|---|
| `so101_sort_one_v1` | `sort_one_baseline` | SmolVLA, GR00T N1.5, GR00T N1.7, ACT, Diffusion (+ π0.5, MolmoAct 2) | 50 (+20) | ready to freeze |
| `so101_cubebox_v1` | `cubebox_v1_baseline` | the same 5 (+2) | 50 (+20) | ready to freeze |
| `so101_cubebox_v0` | `cubebox_v0_baseline` | SmolVLA, GR00T N1.5 (as-shipped condition) | 20 | optional; frozen 2026-11-20 |

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
- **One task, two models** for the robot DRTC experiment (sorting; SmolVLA and GR00T N1.5); setup v2's bench covers
  both tasks and five models, without task success.
- **The operator judges success** (not blinded); the success rule was written before the first trial.
- **Shared GPU servers:** other users' jobs run on the A100 and H200, so inference times include contention.
- **Uncompressed images** make the round trip ~260 ms longer than the inference; compressing them (JPEG) would
  shorten it.
- **The fault mix was fixed in time** in the first run (the same windows in every trial, during the reach); setup v2
  moves it by a random 0-4 s per trial.
- **The bench** uses a replay robot and a simulated latency, so it has no task success.
- **Metrics changed twice**: v1 -> v2 (2026-11-20, before the first run: a trial ended by a fault counts as a failure
  instead of being repeated) and v2 -> v3 (2026-11-22: the fault label from the run's record, jerk only while moving,
  recovery and deviation-from-plan metrics, primary comparisons). Every number here is recomputed with v3.

## 11. Future work

- **The Raspberry Pi 5 as the robot computer** (edge setup): set up and tested 2026-11-20 (CPU PyTorch, the same
  `lab` commands; it can use the A100, the H200 or the laptop's GPU as server). First test (`cubebox_v0_baseline
  --test`): 29.4-29.9 Hz control, no video frames dropped, but the round trip about doubles (SmolVLA 594 ms, GR00T
  N1.5 855 ms, over WiFi to the laptop's hotspot and on to the A100), so the shipped client starved GR00T N1.5 even
  without faults: the edge case DRTC and image compression are for. Official edge trials: not run yet.
- The DRTC experiment on the cube-in-box task (`so101_cubebox_v1`), if time remains.
- Compressed images (JPEG) to cut the round trip.
- RTC for GR00T N1.7 (relative actions need re-anchored leftovers) and the later models.
- An edge fallback: a small model (ACT) on the robot computer takes over while the server is unreachable.

## 12. The report and the presentation

**Report (LaTeX):** `report/` (`make` there: results -> numbers, tables, figures -> `report/report.pdf`). How to
add things: [report/README.md](../report/README.md). Photos to add: [report/assets/README.md](../report/assets/README.md).

**Presentation:** (still showing the first run, with v2 metrics' "5-7.5x"; to be updated with setup v2's results)
the Slides deck https://claude.ai/artifact/JsUD9Vp56hRnTS5qU7VGcJ (private to the owner; 16:9;
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
| 12 | motion | figure: jerk; 5-11x smoother; GR00T stalls 65% -> 24% | |
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
| The conditions | setup v2: [experiments/v2/](experiments/v2/) (bench) and [experiments/v2_robot/](experiments/v2_robot/), made by `v2/_make.py`; the first run: [experiments/main/](experiments/main/), [experiments/bench_main/](experiments/bench_main/) |
| Eval plans (frozen) | [evals/](evals/) |
| Code | [src/drtc/control/](../src/drtc/control/) (DRTC, faults, smoothing), [src/drtc/evals/](../src/drtc/evals/) (evaluation, metrics, the stack rule `stack.py`) |
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
