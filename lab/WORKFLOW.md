# Workflow

Everything you do in this project, with every command, in order. Each command here was checked against the code
(last on 2026-11-22). If one fails, the error message usually says what to do; otherwise paste it to Claude.

- **[Machines](#machines)**: what each machine is for, how to log in, how to update its code.
- **[The whole path at a glance](#the-whole-path-at-a-glance)**: dataset → trained models → evaluated → report.
- **[The plan, step by step](#the-plan-step-by-step)**: the remaining steps of PLAN.md with their exact commands:
  setup v2 (the bench and the robot DRTC run), offline evaluation, the robot baselines, growing an eval, changing
  the plan.
- **[Part 1: from a new dataset to trained models](#part-1-from-a-new-dataset-to-trained-models)**: record, review,
  register, train.
- **[Part 2: robot evaluation](#part-2-robot-evaluation)**: official trials from the laptop or the Pi, results.
- **[Part 3: bench, offline evaluation, report and slides](#part-3-bench-offline-evaluation-report-and-slides)**.
- **[Mistakes and fixes](#mistakes-and-fixes)**, **[latency](#what-to-know-about-latency)**.

**What each stage is and why** (offline vs robot eval vs bench, explained with a new dataset):
[docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md). This file has the commands.

Related: the plan and its status [PLAN.md](PLAN.md); what every number means [METRICS.md](METRICS.md); every result
for the report and the slides [REPORT.md](REPORT.md); details on recording [docs/DATA.md](docs/DATA.md), training
[docs/TRAIN.md](docs/TRAIN.md), the GPU servers [docs/SERVERS.md](docs/SERVERS.md), how results are stored
[docs/EVALUATION.md](docs/EVALUATION.md).

# Machines

| Machine | Used for | Log in | Repo | Update the code |
|---|---|---|---|---|
| **Laptop** (RTX 4060, 8 GB) | recording; the robot computer for every official experiment; serving small models | (you are on it) | `~/Projects/drtc-major` | `git pull` |
| **A100** (80 GB, shared) | training; serving models for robot sessions; bench; offline evaluation | `ssh b230968ee@192.168.5.13` | `~/drtc_major` | `git pull`; after a change of `uv.lock`: `lab/bin/a100-setup`. **Never a plain `uv sync`** there: it installs CUDA 13 builds the driver can't run |
| **H200** (141 GB, shared, professor's account) | training; serving; bench; offline evaluation | Jupyter terminal, or `ssh jayaraj@192.168.41.118` then `docker exec -it drtc_lab bash` | `/workspace/drtc_major` inside the `drtc_lab` container | **inside the container**: `cd /workspace/drtc_major && . ./.server.env && git pull`, then after a change of `uv.lock`: `uv sync --frozen --extra gpu --extra sim --python /usr/bin/python3`. A `git pull` on the host fails (files there are root-owned) |
| **Raspberry Pi 5** (8 GB, no GPU) | the edge robot computer (tested; not used for official results yet) | `ssh rasp@raspberrypi.local` (from the laptop, on its hotspot) | `~/drtc_major` | `git pull`; after a change of `uv.lock`: `lab/bin/pi-setup` |

**More on the A100 and the H200** (the container, updating, the rules, which server for what):
[docs/SERVERS.md](docs/SERVERS.md).

**A second robot laptop** (a teammate's, the same arms and accounts): set it up with
[docs/SETUP-LAPTOP.md](docs/SETUP-LAPTOP.md); it is `laptop2` in `lab/hosts.yaml` and serves on its own GPU with
`--server local`.

Rules on the servers: work only in the repo folder; never delete anything outside it; on the H200 only inside
`drtc_major` (nothing outside, no system changes); on the A100 stop only your own processes. Long jobs run in tmux:
`tmux attach -t <name>` to look, Ctrl-b d to leave it running, `tmux ls` to list.

The environment on every machine: `source .venv/bin/activate` once per terminal, then `lab ...`; or write
`.venv/bin/lab ...` (as in the server commands below). On the A100 and H200, `lab` reads `.server.env` by itself.

# The whole path at a glance

```
 A. RECORD (laptop)            B. TRAIN (H200 / A100)          C. EVALUATE (server + robot computer)     D. REPORT
 ───────────────────           ──────────────────────          ─────────────────────────────────────     ──────────
 plan the dataset              lab/models/<task>.yaml          lab/evals/<eval>.yaml                     lab eval report --pull
 lab/bin/session record        lab check, --dry-run            lab eval check / freeze                   cd report && make
 lab record <name> ...         commit + push                   commit + push                             (Claude updates
 lab data show / edit          server: pull, prepare, train    server: lab eval serve <eval>              REPORT.md and slides)
 commit recorded.yaml          → models on the Hub             robot: lab eval run <eval> --server ...
```

Each step ends with a **commit and push**: that is how the other machines learn about it (a server learns about a new
dataset from `lab/datasets/recorded.yaml`, every machine learns about an eval from `lab/evals/<eval>.yaml`).

**Models are only compared inside one eval** (METRICS.md rule 5): trials of all its models and conditions run
interleaved in the same sessions. A frozen eval that has trials is never changed. So:

| Situation | Make this eval | In it |
|---|---|---|
| A task's baseline (e.g. `so101_cubebox_v1`) | `cubebox_v1_baseline` | every trained model of the task; the one condition `baseline: v2_robot/upstream_patient__none` |
| A model trained later | the same baseline eval, grown ([Growing an eval](#growing-an-eval-more-trials-more-models)) | add it under `models:` and freeze: only it is frozen |
| A new version of a model, or more data for a task | `<task>_baseline_v2` | the new model(s) **plus** the ones to beat, run again side by side |
| A method compared with the baseline | like `sort_one_drtc_v2` (setup v2, E4) | the chosen models; conditions from `v2_robot/` (made by `experiments/v2/_make.py`) and the eval's `comparisons:` |

Only `lab eval run` makes official results (`results/evals/`). Anything else you try (`lab run`, `--test`, `--replay`)
goes to `results/scratch/` and never into the results.

---

# The plan, step by step

The remaining steps of [PLAN.md](PLAN.md) ("Order of work") with the exact commands (2026-11-22). Parts 1-3 below
explain each command in general; this part says which ones to run for the current plan, in which order.

| Plan step | What | Status (2026-11-22) | Here |
|---|---|---|---|
| 4-7, 9 | **Setup v2: the DRTC and smoothing experiments** (bench E1-E3, robot E4) | built; V0 (freeze the bench evals) is next | [Setup v2](#setup-v2-the-drtc-and-smoothing-experiments) |
| 1 | Trainings | you; π0.5 x2 and MolmoAct 2 cubebox v1 left | [Part 1](#part-1-from-a-new-dataset-to-trained-models), steps 7-9 |
| 2 | Offline M9/M10 for every trained model | 10 done; GR00T N1.7 x2 ready now | [Step 6](#step-6-offline-evaluation-of-each-new-model) |
| 8 | Robot baselines: sort_one and cubebox v1 (5 models now, 2 later); cubebox v0 optional | ready to freeze | [Step 5](#step-5-the-robot-baselines) |
| 10 | Growth when there is time (more trials, models, the cubebox v1 DRTC run) | after the core | [Growing an eval](#growing-an-eval-more-trials-more-models), [Step 8](#step-8-optional-the-drtc-experiment-on-cubebox-v1) |
| 11 | Report and slides | Claude, after each result | [Part 3](#the-report-and-the-slides) |

To change any of it (another model, more trials, a model dropped): [Changing the plan](#changing-the-plan) and
[Growing an eval](#growing-an-eval-more-trials-more-models).

## Setup v2: the DRTC and smoothing experiments

The plan: [PLAN.md](PLAN.md), "DRTC + SMOOTHING EXPERIMENT SETUP v2" (locked 2026-11-22). The conditions are
generated files in [experiments/v2/](experiments/v2/) (bench) and [experiments/v2_robot/](experiments/v2_robot/)
(robot), one per system and fault: `<system>__<fault>.yaml`, written by `experiments/v2/_make.py`.

| Step | Eval(s) | Where | Runs | Time (about) |
|---|---|---|---|---|
| V1 | E1 + E2: `bench_v2_drtc_sort_one`, `bench_v2_drtc_cubebox_v1` | a GPU server (you choose) | 2,080 each | ~10 h each |
| V2 | E3a: `bench_v2_smooth_<dataset>` and `bench_v2_smooth_rtc_<dataset>` | a GPU server | 280 + 32 each | ~1.5 h each |
| V3 | the final stack is chosen (the rule from the eval setup) and E3b + E4 are written | Claude | – | – |
| V4 | E3b: `bench_v2_stack_<dataset>` | a GPU server | ~240 each | ~1.2 h each |
| V5 | E4: `sort_one_drtc_v2` (SmolVLA + GR00T N1.5, 4 conditions, 10 trials) | the laptop + a GPU server | 80 | ~1.5 h |
| (5) | the baselines (5 models now) can run any time | the laptop + a GPU server | 50 + 50 | ~1 h each |

V1 and V2 need nothing from each other: run them in either order, or on two servers at once (one dataset each).

**What the names mean.** Systems: `upstream_patient` (LeRobot's scheduler, starvation grace and request deadline = the
whole trial, the baseline), `upstream_shipped` (the same with LeRobot's 1 s grace and 5 s request deadline), `drtc` (our DRTC), `a1_no_cooldown`,
`a2_upstream_estimator`, `a3_upstream_trigger`, `a4_half_chunk_cap`, `a5_smin_half`, `a5_smin_x1_5` (DRTC minus or
changed in one part), `s_<method>` (DRTC + one smoothing method). Faults: `none`, `spike` (one reply +1 s), `drops`
(every reply lost for 1 s), `outage` (1.5 s, both directions), `mix` (spike, drops, outage), `rise` (+400 ms on
every reply for 5 s), `loss` (10% of the replies lost). Every timed fault is moved by a random 0-4 s per trial
number: trial 3 meets its faults at the same times in every condition (`fault_shift_applied_s` in the trial's
`config.yaml`).

### V0. Freeze the bench evals (once, laptop)

```bash
cd ~/Projects/drtc-major && source .venv/bin/activate && git pull
for e in bench_v2_drtc_sort_one bench_v2_drtc_cubebox_v1 bench_v2_smooth_sort_one bench_v2_smooth_cubebox_v1 \
         bench_v2_smooth_rtc_sort_one bench_v2_smooth_rtc_cubebox_v1; do
  lab eval check $e && lab eval freeze $e
done
git add lab/evals/bench_v2_*.yaml && git commit -m "bench v2 evals: frozen" && git push
```

`check` prints `ok (5 models x 52 conditions x 8 trials = 2080; not frozen yet)` for the E1 + E2 evals; `freeze`
lists each model's Hub commit.

### V1, V2, V4. Run a bench eval (GPU server)

The same two commands for every bench eval, in tmux so it keeps running when you disconnect. On the A100
(`ssh b230968ee@192.168.5.13`, `cd ~/drtc_major && . ./.server.env`) or in the H200 container
(`cd /workspace/drtc_major && . ./.server.env`):

```bash
git pull
nvidia-smi                                                 # room for the 5 models: ~15 GB
tmux new -s bench                                          # window 1: the model servers
.venv/bin/lab eval serve bench_v2_drtc_sort_one            # wait for "all 5 ready"
# Ctrl-b c (window 2): the runs
.venv/bin/lab eval run bench_v2_drtc_sort_one --server local
# Ctrl-b d detaches; `tmux attach -t bench` comes back
```

What you see: `round 1, 17/260: ...` lines, one per run (14 s each), in rounds (every model x condition once per
round, 8 rounds). **Stopping and continuing:** to stop cleanly, start it with `--max-trials N` (it stops after N
runs and pushes the session). Ctrl-C stops at once: the run in progress is not counted (it is run again later) and
the session is not pushed, so push it yourself: `.venv/bin/lab eval push bench_v2_drtc_sort_one`. Either way the
same `lab eval run` command later continues where it stopped (an unfinished round is finished first). When the eval
is done: Ctrl-C the servers in window 1, then the next eval.

**RTC evals (`bench_v2_smooth_rtc_<dataset>`, and E3b/E4 if the stack has RTC): one test round first.** The first
real RTC request after a server starts compiles for over 5 s, and the server then drops that run. So after `lab eval
serve` and before the official run, in window 2:

```bash
.venv/bin/lab eval run bench_v2_smooth_rtc_sort_one --server local --test --max-trials 4   # results/scratch, not counted
.venv/bin/lab eval run bench_v2_smooth_rtc_sort_one --server local                          # the official one
```

**Suggested order on one server** (each `serve` + `run` pair as above): `bench_v2_smooth_rtc_sort_one`,
`bench_v2_smooth_rtc_cubebox_v1` (~10 min each) and `bench_v2_smooth_sort_one`, `bench_v2_smooth_cubebox_v1`
(~1.5 h each) first, so V3 (the stack) can be done while the long `bench_v2_drtc_sort_one` and
`bench_v2_drtc_cubebox_v1` (~10 h each) run.

Back on the laptop:

```bash
lab eval status bench_v2_drtc_sort_one                     # runs done per condition and model
lab eval report bench_v2_drtc_sort_one --pull              # results/bench/bench_v2_drtc_sort_one/report.md
```

What the report holds (metrics v3, [METRICS.md](METRICS.md)): success and timing tables, **Recovery (M11)** per
fault, **Primary comparisons** (DRTC against upstream (patient) per fault: tested, Holm-corrected, and how many models
DRTC is better in), then every other comparison of the eval, marked exploratory.

Tell Claude when an eval is done: the results go into REPORT.md and the report.

### V3. The final smoothing stack (after V2)

The rule was fixed before V2 ran (METRICS.md v3, rule 9; `src/drtc/evals/stack.py`). On the laptop, once the four
V2 evals are done and pulled:

```bash
for e in bench_v2_smooth_sort_one bench_v2_smooth_cubebox_v1 bench_v2_smooth_rtc_sort_one bench_v2_smooth_rtc_cubebox_v1; do
  lab eval report $e --pull
done
.venv/bin/python -m drtc.evals.stack
```

It prints, per model and dataset, each candidate's jerk relative to DRTC alone (or why a guard ruled it out), the
pairs each one won, and the final stack with the next command, e.g.
`.venv/bin/python lab/experiments/v2/_make.py --stack adaptive,rtc,zero_lag,ramp`. Run that command **with
`--evals`** added: it writes the stack's conditions, the E3b evals (`bench_v2_stack_<dataset>`, and
`bench_v2_stack_nortc_<dataset>` for ACT, Diffusion and GR00T N1.7 when the stack has RTC) and E4
(`sort_one_drtc_v2`). Then `lab eval check` each, commit, and V0 (freeze), V4 (bench) and V5 (robot). Claude can do
V3 for you: send it the output of the stack command.

### V5. The robot run (E4) and the baselines

Like [Step 5c](#5c-each-session) with the eval's name: the model server on a GPU server (`lab eval serve
sort_one_drtc_v2`), then on the laptop `lab eval servers sort_one_drtc_v2 --server <server>` and
`lab eval run sort_one_drtc_v2 --server <server>`. One round = 8 trials (2 models x 4 conditions); 10 rounds.

### Notes from the local check (2026-11-22, ACT and SmolVLA on the laptop's replay robot)

- Upstream never re-sends a request, so a lost reply (the drops, the outage, loss) ended `upstream_patient` runs
  after LeRobot's 5 s request deadline (`action_timeout`), whatever the starvation grace. Since 2026-11-22 (the
  user) upstream (patient) and DRTC both have `request_deadline: trial` (the whole trial): patient upstream then
  holds the arm and waits to the end of the run (checked: 55% stalled, longest stall 7.5 s under the mix), which
  shows up in the stall metrics instead of ending the run. Replacing a lost reply is what DRTC's cooldown adds
  (ablation A1). `upstream_shipped` keeps LeRobot's 1 s grace and 5 s deadline.
- GR00T N1.7 has no RTC cell: our N1.7 models are trained with relative actions, and LeRobot turns the RTC prefix
  off for those.
- **RTC evals: warm up first.** The first real RTC request after a server starts compiles the RTC path and took
  over 5 s on the laptop; the server then drops the session (`Policy operation deadline exceeded`) and that trial
  ends as a fault. The next ones ran normally (0.23 s per chunk). So for `bench_v2_smooth_rtc_<dataset>` (and any
  eval with an RTC condition), right after `lab eval serve`, run one test round first, then the official session:
  `.venv/bin/lab eval run bench_v2_smooth_rtc_sort_one --server local --test --max-trials 4`.

## Step 6: offline evaluation of each new model

No robot, a few minutes per model, on the H200 or the A100: the held-out action error (M9) and the inference time
and GPU memory (M10). Run it for each model once its training has finished and pushed to the Hub. Not on a server
that is serving a robot session at that moment (it would slow the session's model down).

**Done:** SmolVLA and GR00T N1.5 on all three datasets, ACT and Diffusion on sort_one and cubebox v1 (10 models).
**Now: the two GR00T N1.7 models.** On the H200 (from the host shell: `docker exec -it drtc_lab bash`):

```bash
cd /workspace/drtc_major && . ./.server.env && git pull
nvidia-smi                                   # shared GPU: GR00T needs ~6 GB free
tmux new -s offline '.venv/bin/lab eval offline --models grootn17_cubebox_v1_full,grootn17_sort_one_v1_full; bash'
```

or on the A100:

```bash
ssh b230968ee@192.168.5.13
cd ~/drtc_major && . ./.server.env && git pull
nvidia-smi
tmux new -s offline '.venv/bin/lab eval offline --models grootn17_cubebox_v1_full,grootn17_sort_one_v1_full; bash'
```

Each model prints its numbers, then `-> results/offline/<dataset>/<model>.json` and `pushed to the results store`.
The window stays open at the shell prompt afterwards (`exit` closes it). The dataset must be prepared on that server:
all three are on the H200; if it stops because the dataset is missing (e.g. cubebox v1 on the A100), run
`.venv/bin/lab prepare so101_cubebox_v1` there once, then the offline command again.

**Later, as each training finishes** (same commands, other `--models`; several at once with commas):

| Model(s) | When |
|---|---|
| `molmoact2_sort_one_v1_lora` | the MolmoAct 2 run has finished (A100) |
| `pi05_cubebox_v1_full`, `pi05_sort_one_v1_full`, `molmoact2_cubebox_v1_lora` | after their trainings |

**On the laptop**, to get the results (for the report):

```bash
cd ~/Projects/drtc-major && source .venv/bin/activate && git pull
lab eval report cubebox_v1_baseline --pull    # pulls every offline result of cubebox v1 (works before any trial)
lab eval report sort_one_baseline --pull      # the same for sort_one
```

Then tell Claude, who adds them to REPORT.md, the report and the slides.

## Step 5: the robot baselines

Setup v2 (2026-11-22): the condition is `v2_robot/upstream_patient__none` (upstream LeRobot with the starvation
grace raised to the whole trial, normal network: the same reference as E4), 10 trials per model, the laptop as the
robot computer. The evals start with the 5 trained models; π0.5 and MolmoAct 2 are added when they are trained
([Growing an eval](#growing-an-eval-more-trials-more-models)).

| Eval | Models | Trials | Ready? |
|---|---|---|---|
| `sort_one_baseline` | SmolVLA, GR00T N1.5, GR00T N1.7, ACT, Diffusion (+ π0.5, MolmoAct 2 later) | 5 x 10 = 50 (+20) | **yes**: freeze (5a) |
| `cubebox_v1_baseline` | the same 5 on cubebox v1 (+2 later) | 50 (+20) | **yes**: freeze (5a) |
| `cubebox_v0_baseline` | SmolVLA, GR00T N1.5 (old condition `main/baseline`, 1 s grace) | 2 x 10 = 20 | optional; frozen 2026-11-20 |

SmolVLA and GR00T N1.5 run here next to the other models: models are only compared inside one eval (METRICS.md
rule 5). Their own reference condition in E4 (`sort_one_drtc_v2`) belongs to the DRTC experiment.

### 5a. Freeze (once per eval, laptop)

When every model of the eval is trained and on the Hub:

```bash
cd ~/Projects/drtc-major && source .venv/bin/activate && git pull
lab eval check sort_one_baseline       # "ok (5 models x 1 conditions x 10 trials = 50; not frozen yet)"
lab eval freeze sort_one_baseline      # pins each model to its Hub commit; lists the 5 commits
git add lab/evals/sort_one_baseline.yaml && git commit -m "eval sort_one_baseline: frozen" && git push
```

`freeze` stops with `cannot read orangemoon599/<model> on the Hub ... is it trained and pushed?` while a model is
missing: list only trained models under `models:`, and add the others later
([Growing an eval](#growing-an-eval-more-trials-more-models)).

### 5b. Check the new models once, without counting (before the first official session)

GR00T N1.7 (and later π0.5 and MolmoAct 2) have not been served to the robot yet. Serve the eval (5c, the server
part), then on the laptop:

```bash
lab eval servers sort_one_baseline --server h200            # all 5 "ready"
lab eval run sort_one_baseline --server h200 --replay       # one round on the replay robot: the arm doesn't move
lab eval run sort_one_baseline --server h200 --test         # real trials on the arm, kept in results/scratch;
                                                            # one round (5 trials) is enough, then q
```

If one fails (a traceback in its server log `logs/serve/<eval>/`, or `not served`), paste it to Claude before the
official session.

### 5c. Each session

**Which server.** The whole eval on one server where possible: every trial records its server, and the report
splits the timing metrics (M2-M6) per server when there are several.

| Server | Can serve |
|---|---|
| H200 (141 GB) | every model at once. The first choice for the baselines |
| A100 (80 GB) | all 5 (~15 GB) if `nvidia-smi` shows room next to any running training |
| laptop GPU (8 GB) | ACT, Diffusion, SmolVLA only (`--models` on both commands) |

GPU memory per model when serving (M10 peaks so far): ACT 0.3 GB, SmolVLA 1.0 GB, Diffusion 1.3 GB, GR00T N1.5
5.7 GB; GR00T N1.7, π0.5 and MolmoAct 2: their offline evaluation (step 6) measures it.

If you have to split an eval over two servers (e.g. the small models on the laptop, the rest on the H200), keep each
model on the same server for all its trials, and run the two groups with `--models` on both `serve` and `run`.

**1. The model server.** On the H200, in the container:

```bash
cd /workspace/drtc_major && . ./.server.env && git pull
nvidia-smi
tmux new -s serve '.venv/bin/lab eval serve sort_one_baseline; bash'
# wait for "all 5 ready" (each model loads in turn: several minutes), then Ctrl-b d
```

**2. The robot computer (laptop).**

```bash
cd ~/Projects/drtc-major && source .venv/bin/activate && git pull
lab/bin/session record                                   # phone camera + rig check; every line ok
lab eval servers sort_one_baseline --server h200         # type the H200 password once; all 5 "ready"
lab eval run sort_one_baseline --server h200             # the official session
```

During the session (keys and rules: [Part 2, step 4](#4-run-the-session)):

- **Rounds:** one round = 5 trials (one per model, in a random order). 10 rounds = 50 trials, roughly 1 h of
  trials plus resets; split it over several sessions as you like (`q` ends one, the next `lab eval run`
  continues; `--max-trials N` stops after N).
- **Layouts:** one layout per round, set before its first trial. cubebox: the cube and the box on their taped marks.
  sort_one: the 4 combinations of the recording (red or blue cube x the two squares swapped), in a fixed order over
  the 10 rounds (e.g. 1-2-3-4-1-2-3-4-1-2); write it down and tick off the rounds.
- `s` = success by the eval's rule (printed at the start), `f` = failure, `v` after a trial = a setup mistake only.
- The session is pushed to the results store at its end.

**3. After the session.** On the H200: `tmux attach -t serve`, Ctrl-C. On the laptop:

```bash
lab eval status sort_one_baseline              # trials done per model, the next round
lab eval report sort_one_baseline --pull       # results/evals/sort_one_baseline/report.md
```

When an eval has all its trials, tell Claude (REPORT.md, the report and the slides are updated from it).

The same for the other two: `cubebox_v0_baseline` (2 models: the laptop GPU fits SmolVLA only, so serve it from the
H200 or the A100) and `cubebox_v1_baseline`.

## Step 8 (optional): the DRTC experiment on cubebox v1

Only if time remains (PLAN.md, setup v2, robot evals, item 2): E4's four conditions with SmolVLA and GR00T N1.5 on
cubebox v1, 2 x 4 x 10 = 80 trials. Make the eval from E4 once it exists (setup v2, V3):

```bash
cp lab/evals/sort_one_drtc_v2.yaml lab/evals/cubebox_v1_drtc_v2.yaml
```

Edit `lab/evals/cubebox_v1_drtc_v2.yaml`:

- `description:` `DRTC v2, cubebox v1 - upstream patient, + mix, DRTC + mix, DRTC + stack + mix`
- `success:` `The green cube ends inside the box and the gripper is open, before the time limit.`
- `models:` `smolvla_cubebox_v1_full` and `groot_cubebox_v1_full`
- **delete the whole `frozen:` block** at the end (and its comment line): those are the sort_one models' commits.
- keep `conditions:`, `trials: 10`, `time_limit_s: 30` as they are (the same experiment as on sort_one).

Then `lab eval check`, `lab eval freeze`, commit and push (5a), and sessions as in 5c with `cubebox_v1_drtc_v2`
(one round = 8 trials).

## Growing an eval: more trials, more models

Since 2026-11-22 an eval with trials may **grow** (and only grow): `lab eval run` accepts a plan whose `trials` went
up or that has new models, as long as everything else is unchanged and every frozen model keeps its commit. Trials
run in rounds (every model x condition once per round, in a seeded order), so stop at the end of a round and every
condition has the same number of trials.

**More trials** (e.g. E4 from 10 to 20 when there is time):

```bash
# edit lab/evals/sort_one_drtc_v2.yaml: trials: 10 -> trials: 20
lab eval check sort_one_drtc_v2                 # "... x 20 trials"
git add lab/evals/sort_one_drtc_v2.yaml && git commit -m "eval sort_one_drtc_v2: 20 trials" && git push
lab eval run sort_one_drtc_v2 --server h200     # continues with round 11
```

**More models** (e.g. π0.5 and MolmoAct 2 in the baselines once trained):

```bash
# edit lab/evals/sort_one_baseline.yaml: add pi05_sort_one_v1_full and molmoact2_sort_one_v1_lora under models:
lab eval check sort_one_baseline
lab eval freeze sort_one_baseline     # "has 50 trials: freezing only the added models (...)"; the others keep theirs
git add lab/evals/sort_one_baseline.yaml && git commit -m "eval sort_one_baseline: + pi05, molmoact2" && git push
lab eval serve sort_one_baseline      # on the server: now 7 models
lab eval run sort_one_baseline --server h200     # the next rounds hold only the two new models until they have
                                                  # 10 trials each; with trials: 20 later, all 7 run together again
```

The new models run on another day than the first ones: comparing them with the older models also compares the two
days (light, cube, arm). Comparisons inside one model are not affected.

**Anything else** (a condition, the time limit, the success rule, a model's version): a new eval, below.

What to add first when there is time (PLAN.md, setup v2, robot evals): E4 to 20 trials; the DRTC robot run on
cubebox v1; E4 on a third model; the baselines to 20 trials; a robot run of a bench fault that surprised us.

## Changing the plan

The rule: **an eval can be edited freely until its first official trial; after that it only grows** (more trials,
added models: above), and any other change means a new eval (`<eval>_v2`). `lab eval run` refuses a plan that
differs from its earlier sessions in anything else, so a mistake can't slip in.

| You want to | Do | Allowed |
|---|---|---|
| Drop a model (not trained in time) | delete its line under `models:` | before the first trial; if the eval is frozen, also delete its line under `frozen:` |
| Add a model | add its name under `models:` (it must be in `lab/models/*.yaml`), then `lab eval freeze` again | before the first trial |
| Retrained a model before any trial | `lab eval freeze <eval>` again (pins the new Hub commit) | before the first trial |
| More or fewer trials, another time limit or success rule | edit `trials:`, `time_limit_s:`, `success:` | before the first trial |
| More trials or another model after trials exist | grow the eval ([Growing an eval](#growing-an-eval-more-trials-more-models)) | always |
| Anything else after trials exist | a new eval: copy the file to `<eval>_v2.yaml`, delete its `frozen:` block, change it, freeze it; include the models to compare with (rule 5) | always |
| Another condition (other DRTC or fault settings) | ask Claude: it is a new system or fault in `lab/experiments/v2/_make.py` (which writes the condition files) and a change to the locked setup. Never edit an experiment file an eval with trials uses | – |
| Another task's baseline | a new eval file like `sort_one_baseline.yaml`, models of that task | – |

After any edit, check, commit and push, then pull on the server and the robot computer before the next session
(`lab eval run` refuses uncommitted changes):

```bash
lab eval check sort_one_baseline
lab eval freeze sort_one_baseline              # only if models were added or retrained, or it was frozen before the edit
git add lab/evals/sort_one_baseline.yaml
git commit -m "eval sort_one_baseline: without pi05 (not trained in time)"
git push
```

If the change is to the plan itself (a model left out of the study, a step dropped), also change the tables in
[PLAN.md](PLAN.md) and commit them the same way (`git add lab/PLAN.md`), so the report says the same.

---

# Part 1: from a new dataset to trained models

The example in this part is a new dataset, sorting B (`so101_sort_two_v1`). It is only an example: the datasets of
the locked scope are recorded already ([PLAN.md](PLAN.md)). Details: [docs/DATA.md](docs/DATA.md),
[docs/TRAIN.md](docs/TRAIN.md).

```
laptop:  plan → set up the scene → record → review / edit → commit recorded.yaml → add the models → check → push
server:  pull → prepare → train (tmux) → watch in wandb → models on the Hub
```

## Cheat sheet

```bash
# --- laptop ---
cd ~/Projects/drtc-major && source .venv/bin/activate && git pull
lab/bin/session record                          # phone camera + checks, in tmux "robot"
lab record so101_sort_two_v1 --task "Put the red cube in the red square and the blue cube in the blue square." \
    --episodes 100 --held-out 10 --episode-time 45 --reset-time 15
lab data show so101_sort_two_v1
git add lab/datasets/recorded.yaml && git commit -m "so101_sort_two_v1: 110 episodes (100 + 10)" && git push
# write lab/models/sort_two_v1.yaml (step 6)
lab check && lab train act_sort_two_v1_full --dry-run
git add lab/models/sort_two_v1.yaml && git commit -m "models: sort_two_v1" && git push

# --- H200, in the drtc_lab container ---
cd /workspace/drtc_major && . ./.server.env && git pull && uv sync --frozen --extra gpu --extra sim --python /usr/bin/python3
.venv/bin/lab prepare so101_sort_two_v1
nvidia-smi
tmux new -d -s sort2 -n act '.venv/bin/lab train act_sort_two_v1_full; bash'
sleep 60
tmux new-window -t sort2 -n smolvla '.venv/bin/lab train smolvla_sort_two_v1_full; bash'
```

---

## 1. Plan the dataset

Decide these before the session and write them down:

| What | Rule | Sorting B |
|---|---|---|
| Name | `so101_<task>_v<N>`; a new version for a new setup, never reuse a name | `so101_sort_two_v1` |
| Instruction | one sentence, identical for every episode (copy it, don't retype it) | "Put the red cube in the red square and the blue cube in the blue square." |
| Episodes | N training + K held out (K about 10% of N); records N + K | 100 + 10 = 110 |
| Times | episode time long enough for a slow episode; reset time to put everything back on the marks | 45 s / 15 s |
| Rotation | every layout combination equally often over **all N + K** episodes, in a fixed order | 4 combinations x 27-28 |

The rotation is kept by hand (nothing records the layout). Write it on paper and tick off the episodes. With 110
episodes and 4 combinations it can't be perfectly even (27, 28, 27, 28): that's fine, keep the order fixed.

## 2. Set up the scene and the rig (laptop)

1. Objects on their taped marks; the scene camera (phone) and the wrist camera in place; both arms powered and
   connected over USB; the phone connected over USB with DroidCam open.
2. Start the session:

   ```bash
   cd ~/Projects/drtc-major && source .venv/bin/activate
   lab/bin/session record
   ```

   This opens tmux `robot`: the phone camera bridge in one window, the checks in the other. Re-running it
   re-attaches. Without it, start the bridge yourself with `lab/bin/phonecam` in its own terminal.
3. Check the arms, ports, calibration and cameras (the session does this; `lab record` checks again):

   ```bash
   lab check --rig
   ```

   Every line should say `ok`. Never recalibrate for a new dataset: the calibration stays the same for all of them
   (if you ever do, copy it to the Pi too: [the Pi section](#the-raspberry-pi-as-the-robot-computer)).

## 3. Record

```bash
lab record so101_sort_two_v1 --task "Put the red cube in the red square and the blue cube in the blue square." \
    --episodes 100 --held-out 10 --episode-time 45 --reset-time 15 --dry-run    # check what it will do
# then the same command without --dry-run
```

During the session:

| Key | Does |
|---|---|
| → | the episode is done: end it now and go on to the reset |
| ← | the episode went wrong: discard it and record it again |
| Esc | stop the session; what is recorded is kept, checked, trimmed and pushed |

- The spoken prompts say which episode is next ("Recording episode 3") and when to reset.
- A bad episode: press ← straight away rather than fixing it later. The rotation stays in step that way.
- Tired? Press Esc, and continue another time with the **same command plus `--resume`**. It prints how many
  episodes are recorded and how many are left; go on in the rotation from the next episode number.

At the end `lab record` checks the recording, builds the trimmed copy, pushes both to the Hub and writes their
entries in `lab/datasets/recorded.yaml`.

## 4. Review and edit

```bash
lab data show so101_sort_two_v1          # one line per episode: length, kept range, warnings (!)
lab data view so101_sort_two_v1 12       # look at an episode in Rerun (add --trimmed for the trimmed copy)
```

Look at the episodes marked `!` and very short or very long ones. If something is wrong:

```bash
lab data delete so101_sort_two_v1 12                      # a failed episode (the later ones are renumbered)
lab data trim so101_sort_two_v1 12 --start 1.5 --end 20   # cut by hand, in seconds of the raw episode
lab data undo so101_sort_two_v1                           # undo the last edit
lab data push so101_sort_two_v1                           # upload the edits, update recorded.yaml
```

Edits stay on the laptop until `lab data push`. Edit **before** training starts: deleting episodes later changes
which episodes are held out, so models trained before and after aren't comparable.

## 5. Commit the dataset entry

The servers learn about the dataset (its exact Hub revision and its held-out split) from `recorded.yaml`:

```bash
git status                                   # only lab/datasets/recorded.yaml should be changed
git add lab/datasets/recorded.yaml
git commit -m "so101_sort_two_v1: 110 episodes (100 training + 10 held out)"
git push
```

Do this after every `lab record` and every `lab data push`.

## 6. Add the models

Create `lab/models/sort_two_v1.yaml` (copy `lab/models/sort_one_v1.yaml` and change the names):

```yaml
# Sorting B (so101_sort_two_v1): original recipes.
act_sort_two_v1_full: {recipe: act_original, dataset: so101_sort_two_v1, status: planned,
  notes: 'ACT on so101_sort_two_v1; original recipe'}
smolvla_sort_two_v1_full: {recipe: smolvla_authors, dataset: so101_sort_two_v1, status: planned,
  notes: 'SmolVLA on so101_sort_two_v1; original recipe (action expert only)'}
```

- Name: `<policy>_<task>_<version>_<full|lora>`; the dataset is the **trimmed** name (no `_raw`). GR00T N1.7 models
  are named `grootn17_...`, π0.5 `pi05_...`, MolmoAct 2 `molmoact2_..._lora`.
- The seven recipes (fixed; never edit them; a new recipe or model is a question for Claude):

  | Policy | Recipe | Steps x batch | Time alone on the H200 (measured) |
  |---|---|---|---|
  | ACT | `act_original` | 100k x 8 | ~3 h (~5 h with other runs) |
  | SmolVLA | `smolvla_authors` | 20k x 64 | ~1.1 h (~4 h with other runs) |
  | GR00T N1.5 | `groot_n15_nvidia` | 10k x 32 | ~1.4 h (~3.6 h with other runs) |
  | Diffusion Policy | `diffusion_original` | 600 epochs x 64 | ~11 h for 110 episodes, with other runs |
  | GR00T N1.7 | `groot_n17_nvidia` | 20k x 64 | not measured yet |
  | π0.5 | `pi05_openpi` | 20k x 64 | ~27 h, ~77 GB of GPU memory |
  | MolmoAct 2 | `molmoact2_lora` | 50k x 16 | ~5 days on the A100 (~9-10 s per step) |

- `status` is a note only (`planned` → `trained` once it's on the Hub); nothing depends on it.

Check, then commit and push:

```bash
lab check                                        # the registry is consistent
lab train act_sort_two_v1_full --dry-run         # dataset revision, held-out episodes, full command; downloads nothing
lab train smolvla_sort_two_v1_full --dry-run
git add lab/models/sort_two_v1.yaml
git commit -m "models: ACT, SmolVLA on so101_sort_two_v1"
git push
```

The dry run should show the held-out count you recorded (`held out: [...]`, 10 episodes) and
`episodes: 100 for training`.

## 7. Train

**On the H200**, in the `drtc_lab` container (login and update: [Machines](#machines)):

```bash
cd /workspace/drtc_major && . ./.server.env && git pull && uv sync --frozen --extra gpu --extra sim --python /usr/bin/python3
.venv/bin/lab check --logging                  # wandb and HF keys present (once per setup is enough)
.venv/bin/lab prepare so101_sort_two_v1        # download + prepare, once per new dataset revision
nvidia-smi                                     # the GPU is shared: check there is room
df -h /workspace                               # and the disk (each GR00T checkpoint is ~5.5 GB)
tmux new -d -s sort2 -n act     '.venv/bin/lab train act_sort_two_v1_full; bash'
sleep 60
tmux new-window -t sort2 -n smolvla '.venv/bin/lab train smolvla_sort_two_v1_full; bash'
tmux attach -t sort2          # Ctrl-b n: next window; Ctrl-b d: detach (the runs go on)
```

Start ACT first: the CPU limits in `lab/hosts.yaml` apply to runs started while another one is training, and ACT
needs the least. Use a new tmux session name per dataset (`sort2`, `stack`, ...) so it doesn't clash with a running
one (`tmux ls`). Closing the browser or logging out doesn't stop the runs.

**On the A100** (the same commands, on the host, no container):

```bash
ssh b230968ee@192.168.5.13
cd ~/drtc_major && . ./.server.env && git pull          # . ./.server.env BEFORE starting: video decoding needs it
nvidia-smi && df -h ~                                    # shared GPU; /home is nearly full
tmux new -d -s train -n act '.venv/bin/lab train act_sort_two_v1_full; bash'
```

## 8. Watch

- **wandb**, project `drtc-lab`: training loss, learning rate and held-out loss, from any machine.
- The tmux windows: progress bar with the time left.
- Normal noise at the start (not errors): transformers deprecation lines, `min/max_dynamic_tiles`, and for GR00T a
  "tokenizer ... incorrect regex pattern ... `fix_mistral_regex=True`" warning (its tokenizer is Qwen, not Mistral).
- A real problem: a traceback, `CUDA out of memory`, or the window back at the shell prompt before the end.

Stopping and continuing:

```bash
.venv/bin/lab stop act_sort_two_v1_full        # clean stop: checkpoint, push, exit (or Ctrl-C in its window)
.venv/bin/lab train act_sort_two_v1_full --resume   # continues from the last saved checkpoint
```

A window closed by accident (the process killed) loses only the steps since the last checkpoint (`save_freq` in the
recipe): continue with `--resume` the same way.

## 9. Done

Each run pushes to `orangemoon599/<model>` at its end: the repo root is the best checkpoint by held-out loss (what
`lab eval serve` uses), `checkpoints/<step>/` the last one with its training state. On the laptop, set
`status: trained` in the model file, commit and push. Then (PLAN.md steps 2 and 8): the offline evaluation
([Part 3](#offline-evaluation-m9-m10)), and add it to its task's robot baseline and bench evals: they grow, so the
model is added under `models:` and frozen on its own ([Growing an eval](#growing-an-eval-more-trials-more-models)).
For a bench eval that is not frozen yet, add the model to `DATASETS` in `lab/experiments/v2/_make.py` and run it with
`--evals` instead.

## When to ask Claude

- Anything that deletes or replaces: `outputs/train/<model>` already exists, a stale dataset copy on a server,
  `lab data remove --hub`.
- A traceback, out-of-memory that persists after `nvidia-smi` shows the GPU free, a dry run that looks wrong.
- A new task with per-episode instructions, a new recipe or model, a new server.

Paste the last lines of the window (or the dry run's output) with the question.

---

# Part 2: robot evaluation

Official robot trials, from the plan to the report. Why it works this way: [docs/EVALUATION.md](docs/EVALUATION.md).
What every number means: [METRICS.md](METRICS.md). The example throughout is **`cubebox_v1_baseline`**: the 5
trained models on the cube-in-box v1 task, condition `baseline` (upstream patient, normal network), 10 trials each.

```
once per eval    laptop: write lab/evals/<eval>.yaml → lab eval check → lab eval freeze → commit + push
each session     model server (A100 / H200 / laptop GPU): lab eval serve <eval>               (leave it running)
                 robot computer (laptop, or the Pi): lab eval servers → lab eval run <eval> --server <host>
any time         lab eval status <eval>   ·   lab eval report <eval> --pull
```

**The robot computer for official results is the laptop** (the locked scope, PLAN.md; the report says so). The Pi
works the same way ([its section](#the-raspberry-pi-as-the-robot-computer)) and is for the edge tests.

## Cheat sheet

```bash
# --- once per eval (laptop) ---
cd ~/Projects/drtc-major && source .venv/bin/activate && git pull
lab eval check cubebox_v1_baseline
lab eval freeze cubebox_v1_baseline
git add lab/evals/cubebox_v1_baseline.yaml && git commit -m "eval cubebox_v1_baseline: frozen" && git push

# --- each session: 1. the model server (pick one) ---
# A100:   ssh b230968ee@192.168.5.13, then
cd ~/drtc_major && . ./.server.env && git pull && tmux new -s serve '.venv/bin/lab eval serve cubebox_v1_baseline; bash'
# H200:   in the drtc_lab container
cd /workspace/drtc_major && . ./.server.env && git pull
tmux new -s serve '.venv/bin/lab eval serve cubebox_v1_baseline; bash'
# laptop GPU (8 GB: small models only; add --for-pi when the Pi is the robot computer)
lab eval serve cubebox_v1_baseline --models smolvla_cubebox_v1_full

# --- each session: 2. the robot computer (laptop) ---
cd ~/Projects/drtc-major && source .venv/bin/activate && git pull
lab/bin/session record                                      # phone camera + rig check (tmux "robot")
lab eval servers cubebox_v1_baseline --server a100          # every model "ready"?
lab eval run cubebox_v1_baseline --server a100 --test       # optional: one test trial (scratch)
lab eval run cubebox_v1_baseline --server a100              # the official session

# --- afterwards (any machine) ---
lab eval status cubebox_v1_baseline
lab eval report cubebox_v1_baseline --pull                  # → results/evals/cubebox_v1_baseline/report.md
```

---

## 1. Write the eval (once per study)

One file per study in `lab/evals/`. `cubebox_v1_baseline.yaml` before freezing (setup v2):

```yaml
description: Baseline, cubebox_v1 - upstream LeRobot (patient), normal network
success: The green cube ends inside the box and the gripper is open, before the time limit.
time_limit_s: 30          # a trial that runs out is a failure
trials: 10                # valid trials per model per condition (METRICS.md rule 1); may grow later
seed: 0                   # the order inside each round
models:                   # the trained ones; more are added later (Growing an eval)
  - smolvla_cubebox_v1_full
  - groot_cubebox_v1_full
  - grootn17_cubebox_v1_full
  - act_cubebox_v1_full
  - diffusion_cubebox_v1_full
conditions:
  baseline: v2_robot/upstream_patient__none   # lab/experiments/v2_robot/upstream_patient__none.yaml
video: true
```

- **Decide everything before the first trial.** Once trials exist the plan only grows (more `trials`, added
  `models`); anything else means a new eval (e.g. `cubebox_v1_baseline_v2`), and `lab eval run` refuses it.
- **`success`** is the rule you apply when you press `s`. Write it so that close calls are rare.
- **`conditions`**: setup v2's conditions are generated in `lab/experiments/v2_robot/` (robot) and
  `lab/experiments/v2/` (bench) by `lab/experiments/v2/_make.py`, which also writes the v2 eval files with their
  `comparisons:` and `primary:` lists (METRICS.md v3, rules 5 and 7). The older conditions (`main/...`,
  `bench_main/...`) belong to the first DRTC run and its benches, kept as side results.

Then:

```bash
lab eval check cubebox_v1_baseline      # names exist, conditions point at real experiments, task found
lab eval freeze cubebox_v1_baseline     # pins every model to its current Hub commit, written into the file
git add lab/evals/cubebox_v1_baseline.yaml && git commit -m "eval cubebox_v1_baseline: frozen" && git push
```

Freezing makes "the same model" exact: the laptop, the Pi and every server use that commit, and a server serving any
other version simply can't be reached by the session. Freeze once every model in it is trained and on the Hub.

## 2. Start the model server

Pick whichever server is free that day. **The whole session uses one server**, and each trial records which one it was.

| Server | Where to type | Command |
|---|---|---|
| A100 | `ssh b230968ee@192.168.5.13` | `cd ~/drtc_major && . ./.server.env && git pull && tmux new -s serve '.venv/bin/lab eval serve cubebox_v1_baseline; bash'` |
| H200 | the `drtc_lab` container | `cd /workspace/drtc_major && . ./.server.env && git pull`, then `tmux new -s serve '.venv/bin/lab eval serve cubebox_v1_baseline; bash'` |
| laptop GPU | a laptop terminal | `lab eval serve cubebox_v1_baseline --models smolvla_cubebox_v1_full` (add `--for-pi` when the Pi is the robot computer) |

It starts the models one after the other and prints each one's status:

```
serving cubebox_v1_baseline (5 models) on 127.0.0.1; logs in logs/serve/cubebox_v1_baseline/
  smolvla_cubebox_v1_full          port 9720  starting (loading the model) ... ready in 540 s
  groot_cubebox_v1_full            port 9721  starting (loading the model) ... ready in 160 s
  ...
all 5 ready. Leave this running during the session; Ctrl-C stops every server.
```

- Wait for **`all N ready`**. SmolVLA compiles its model when it starts (about 9 minutes the first time on the A100).
- Detach with Ctrl-b d. At the end of the session, attach (`tmux attach -t serve`) and press Ctrl-C.
- The laptop's 8 GB fits the small models (ACT, Diffusion, SmolVLA); not GR00T or the large VLAs. Then run the
  session with the same `--models`.
- Each eval has its own ports, so two evals can be served on one machine at once if the GPU has room. If
  `lab eval serve` says `ports in use`, that eval is already being served there: `tmux ls`, attach, Ctrl-C.
- The A100 and H200 are shared: check `nvidia-smi` for room first. Training jobs may run alongside; the trial records
  the inference time, so that is visible in the results.

## 3. Prepare the robot (laptop)

```bash
cd ~/Projects/drtc-major && source .venv/bin/activate && git pull
lab/bin/session record          # phone camera bridge + rig check; every line ok
lab eval servers cubebox_v1_baseline --server a100
```

`lab eval servers` opens the way to the servers and asks each model to describe itself:

| `--server` | How it connects |
|---|---|
| `a100` | an SSH tunnel by key (no password) |
| `h200` | an SSH tunnel into the container: type the H200 password once per session |
| `laptop` | on the laptop: directly; on the Pi: to the laptop's hotspot address (the laptop serves with `--for-pi`) |
| `local` | servers on this same machine (the bench) |

```
cubebox_v1_baseline on a100:
  smolvla_cubebox_v1_full          ready   commit 8beffbfd  checkpoint sha256:...  tcp/127.0.0.1:19720
  groot_cubebox_v1_full            ready   commit 9b164e1e  checkpoint sha256:...  tcp/127.0.0.1:19721
  ...                              (one line per model)
```

Every model must say `ready` (through a tunnel the local port is the server's + 10000). Optional checks without
counting anything:

```bash
lab eval run cubebox_v1_baseline --server a100 --test     # one real trial on the arm, kept in results/scratch
lab eval run cubebox_v1_baseline --server a100 --replay   # one round on the replay robot: the arm doesn't move
```

## 4. Run the session

```bash
lab eval run cubebox_v1_baseline --server a100
```

It checks everything first and refuses to start when something is off: uncommitted changes in the repo, a changed
plan, a failed rig check, or a model that isn't served. Then it pulls the sessions other machines pushed, so the
counts include them.

It works in **rounds**. A round is one trial of every model under every condition, in a random order (the same on
every machine).

```
=== Round 1: 5 trials ===
Set this round's layout (the same for every trial of the round).

Round 1, trial 1/5: groot_cubebox_v1_full, t01. Put the cube(s) on the marks, press Enter to start ...
  [success] ending the trial
  t01: SUCCESS in 9.4 s (s; completed)
  Enter = next trial, v = void this trial (setup mistake), q = end the session:
```

| When | Key | Means |
|---|---|---|
| before a trial | Enter | start (objects on their marks) |
| before a trial | `q` | end the session |
| during a trial | `s` | **success**, right now: the task is done by the success rule. Ends the trial; the time is the time to success |
| during a trial | `f` | **failure**: it won't succeed. Ends the trial early |
| during a trial | (nothing) | the time limit ends it: failure |
| after a trial | Enter | next trial |
| after a trial | `v` | **void**: a setup mistake (object not on its mark, wrong layout, cable caught), with a reason. Kept, not counted, run again in a later round. **Never for a model's failure** |
| after a trial | `q` | end the session |

- **Layouts are yours:** one layout per round, all trials of the round use it; rotate the layouts over the rounds as
  in the recording (10 rounds for 10 trials).
- **A fault ends a trial** when the system stops the arm (the server stopped, the network dropped, the shipped client
  gave up): after the arm has started moving it is a **failure** (label `fault`, counted in M1f), whatever key you
  press while the arm returns home (your key is kept as `key`); before the first action it is a setup problem and
  the trial is run again in a later round (METRICS.md v3).
- Stop whenever you like (`q`). The next `lab eval run` continues where the eval stands, on any machine.
- `--max-trials N` ends the session after N trials.
- At the end the session is **pushed** to the results store automatically. Without internet use `--offline`, then
  `lab eval push cubebox_v1_baseline` later.

## 5. Results

```bash
lab eval status cubebox_v1_baseline           # trials done per model, the next round, unpushed sessions
lab eval report cubebox_v1_baseline --pull    # rebuild the report from every session (all machines)
```

`results/evals/cubebox_v1_baseline/`:

| File | What |
|---|---|
| `report.md` | success with its 95% interval, trials ended by a fault, time to success; stalls, latency, server time, jerk, chunk jumps, deviation from the plan (median and spread); recovery after each fault (M11); the primary comparisons (tested, Holm) and every other comparison (exploratory); trials not counted; sessions (METRICS.md v3) |
| `trials.csv` | one row per trial (also voided ones), with its folder |
| `metrics.csv` | one row per counted trial, every metric |
| `sessions/<date>-<machine>/trials/<condition>/<model>/tNN/` | the raw files of one trial: `ticks.csv` (with the model's plan, `plan.*`), `chunks.csv`, `faults.csv`, `recovery.csv`, `config.yaml` (with the fault shift), `trial.json` (label, key), `rollout.log`, `video/scene.mp4`, `video/wrist.mp4` |

The report is rebuilt every time, with the current metrics, so never edit it. Tell Claude when an eval is done: REPORT.md, the report and the
slides are updated from it.

## The Raspberry Pi as the robot computer

The Pi (Raspberry Pi 5, 8 GB, no GPU, hostname `raspberrypi`, on the laptop's hotspot) does exactly the laptop's
robot-side job: the arm, the cameras, `lab eval run`. Every session records which robot computer it was
(`results/evals/<eval>/sessions/<date>-pi/`), and reports never mix robot computers.

**Once** (and again after a change of `uv.lock`), from a laptop terminal:

```bash
ssh rasp@raspberrypi.local                                               # the laptop's SSH key is authorised
git clone -b major https://github.com/arshjeevs/DRTC.git ~/drtc_major    # first time only
cd ~/drtc_major && git pull && lab/bin/pi-setup
```

`pi-setup` installs Python 3.12 and the robot side with the **CPU build** of PyTorch (the Pi has no GPU), and checks
the system parts (the phone camera device `/dev/video10`, the clock synced to the laptop). It asks for the sudo
password only when one of those is missing. It ends with `pi-setup: done`.

**The arm's calibration is the laptop's** (with another file the joint angles would not match the recordings). Copy
it after every new calibration on the laptop:

```bash
# laptop
C=.cache/huggingface/lerobot/calibration
ssh rasp@raspberrypi.local "mkdir -p ~/$C/robots/so101_follower ~/$C/teleoperators/so101_leader"
scp ~/$C/robots/so101_follower/follower.json rasp@raspberrypi.local:$C/robots/so101_follower/
scp ~/$C/teleoperators/so101_leader/leader.json rasp@raspberrypi.local:$C/teleoperators/so101_leader/
```

**Each session:** plug the follower arm (with its power supply), the wrist camera and the phone into the Pi (the
leader arm is not needed; the Pi needs its 5 V / 5 A supply). The first time, the phone asks to allow USB debugging
for the Pi: tap Allow (and "always allow"). Then on the Pi:

```bash
cd ~/drtc_major && source .venv/bin/activate && git pull
lab/bin/session record                          # phone camera + rig check (tmux "robot"); every line ok
lab eval servers <eval> --server a100           # every model "ready"?
lab eval run <eval> --server a100 --test        # a test trial; drop --test for official trials
```

**Model servers for Pi sessions** (the eval must be frozen and pulled on both machines):

| Server | On the server | On the Pi |
|---|---|---|
| A100 | `lab eval serve <eval>` (in tmux) | `--server a100`: the Pi's own key opens the tunnel (authorised 2026-11-20) |
| H200 | `lab eval serve <eval>` in the drtc_lab container | `--server h200`: type the H200 password once per session |
| laptop GPU (small models) | `lab eval serve <eval> --for-pi`: listens on the hotspot address 10.42.0.1 | `--server laptop`: straight to the laptop, no tunnel |

With `--for-pi` the laptop's own sessions cannot reach those servers (they listen on the hotspot address only); serve
without it for laptop sessions.

First test (2026-11-20, `cubebox_v1_baseline --test`, A100): 29.4-29.9 Hz control, no video frames dropped; the round
trip is about twice the laptop's (SmolVLA 594 ms, GR00T N1.5 855 ms: WiFi to the laptop's hotspot, then to the A100),
so the shipped client (1 s give-up) can starve GR00T N1.5 even without faults.

---

# Part 3: bench, offline evaluation, report and slides

## The bench: replay robot

**The current benches are setup v2's**: commands in [Setup v2](#setup-v2-the-drtc-and-smoothing-experiments) (V0-V4).
This section explains the bench in general; the `bench_main_*` evals below are the first DRTC run's benches
(2026-11-20, done), kept as side results.

A bench eval (`robot: replay` in the eval file) needs no robot and no laptop: the model servers and the replay robot
run on the same GPU server, and the replay robot plays the held-out episodes as the camera input. It records M2-M8,
M11 and M12 (no task success: nothing moves). The network to the robot is simulated on every reply as measured on the robot,
**250 ± 20 ms** ([experiments/bench_main/](experiments/bench_main/)).

| Bench eval | Compares | Trials |
|---|---|---|
| `bench_main_spike` | the four conditions, the latency spike alone | 2 models x 4 x 8 = 64 |
| `bench_main_drops` | the four conditions, the dropped replies alone | 64 |
| `bench_main_outage` | the four conditions, the outage alone | 64 |
| `bench_main_mix` | the four conditions, the whole fault mix (checks the bench against the robot) | 64 |

On the A100 (or in the H200 container), one bench at a time, about 20 minutes each:

```bash
cd ~/drtc_major && git pull
tmux new -s bench                                          # window 1: the servers
.venv/bin/lab eval serve bench_main_spike                  # wait for "all 2 ready"
# Ctrl-b c (a second window):
.venv/bin/lab eval run bench_main_spike --server local     # runs every trial by itself; pushes at the end
# back in window 1 (Ctrl-b p): Ctrl-C stops the servers; then the next bench
```

`--server local` means the servers on this same machine. Detach with Ctrl-b d; the bench keeps running. The script
`~/drtc_major/logs/run_bench_main.sh` on the A100 runs all four in a row. On the laptop:
`lab eval report bench_main_spike --pull` → `results/bench/bench_main_spike/report.md`. The bench evals are done
(2026-11-20); run them again only after a change to the scheduler or the smoothing.

## Offline evaluation (M9, M10)

On a GPU server, no robot: each model predicts from the held-out episodes (M9, the action error) and is timed over
200 requests (M10, inference time and GPU memory). A few minutes per model. Run it for every newly trained model:

```bash
# A100 (in the H200 container: cd /workspace/drtc_major instead)
cd ~/drtc_major && . ./.server.env && git pull
tmux new -s offline '.venv/bin/lab eval offline --models grootn17_cubebox_v1_full,grootn17_sort_one_v1_full; bash'
```

Which models are left and when: [Step 6](#step-6-offline-evaluation-of-each-new-model).

- `--models` scores the models at their current Hub commits; `lab eval offline <eval>` scores an eval's models at its
  frozen commits.
- The results go to `results/offline/<dataset>/<model>.json` and are pushed to the results store; on the laptop they
  come with `lab eval report <any eval of that dataset> --pull`.
- Not during a robot session on the same server (it would slow the session's model down).

## The report and the slides

- **Report (LaTeX)**, in `report/` on the laptop: every number, table and graph is made from the results.

  ```bash
  cd ~/Projects/drtc-major/report
  make          # results -> report/generated/ -> report/report.pdf
  make pdf      # only the PDF, after editing a chapter
  ```

  Pull the results first (`lab eval report <eval> --pull`). Title-page details: `report/config.tex`; photos:
  `report/assets/` (list in `report/assets/README.md`). How to add things: [report/README.md](../report/README.md).
- **Slides**: the deck https://claude.ai/artifact/JsUD9Vp56hRnTS5qU7VGcJ (its slide list and placeholders:
  [REPORT.md](REPORT.md), section 12). Claude updates it from the same figures.

---

## Mistakes and fixes

| Situation | Do |
|---|---|
| Realised after a trial that an object wasn't on its mark | `lab eval void cubebox_v1_baseline <model> t05 --reason "cube off its mark"` (add `--condition <name>` when the eval has several), then `lab eval push cubebox_v1_baseline` |
| Want to try something without it counting | `--test` (a real session, kept in `results/scratch/`) or `--replay` (no arm movement) |
| `the git tree has uncommitted changes` | commit and push first (official trials record the exact code), or ask Claude what changed |
| `... changed since session ...: <why>` | the eval file was changed in a way a plan with trials may not (only more `trials` and added `models` are allowed): undo the edit, or make a new eval |
| A run of an RTC eval ends with `Policy operation deadline exceeded` (server log) | the server's first RTC request compiles for over 5 s: run one `--test` round right after `lab eval serve`, then the official session ([Setup v2, V1](#v1-v2-v4-run-a-bench-eval-gpu-server)) |
| `lab eval freeze` says `has N trials: freezing only the added models` | expected when an eval grows: the new models are frozen, the others keep their commits |
| `... is not frozen` | `lab eval freeze <eval>` on the laptop, commit, push, then `git pull` on the server and the robot computer |
| `not served on a100: ...` | the server isn't running or is still loading: check its tmux window |
| `ports in use` | that eval is already being served there: `tmux ls`, attach, Ctrl-C |
| `cannot connect to a100` | the A100 isn't reachable from this network: use the H200 or the laptop GPU |
| `rig check failed` | arms or cameras not found: `lab check --rig`, check USB, power and the phone camera (DroidCam open, USB debugging allowed) |
| `WARNING: push failed` | the session stays on this machine: `lab eval push <eval>` when online |
| A download from the results store stopped (network error) | run the same `--pull` again: it finishes the interrupted sessions |
| H200: `git pull` says `Permission denied` | you ran it on the host: run it inside the container (`docker exec -it drtc_lab bash`) |
| A100: `NVIDIA driver too old` | someone ran a plain `uv sync`: run `lab/bin/a100-setup`, then start the job again |
| A training window closed by accident | `lab train <model> --resume` (it continues from the last checkpoint) |

## What to know about latency

Measured 2026-11-20 (robot runs, `sort_one_drtc_main`): the round trip from the laptop to the A100 is **311 ms** for
SmolVLA (53 ms of it on the GPU) and **452 ms** for GR00T N1.5 (194 ms). About **260 ms is the network**, mostly the two
camera images sent uncompressed (2 x 800 x 600 x 3 bytes = 2.9 MB per request); on the laptop's own GPU the round
trip is tens of ms. From the Pi the round trip roughly doubles (it goes over WiFi to the laptop's hotspot first). It's
recorded in every trial (M4, M6), and reports never mix servers or robot computers.

Models with short chunks have the least room for the delay: GR00T N1.5 (16 actions = 0.53 s) stalled 59% of the time
on the normal network with the shipped client, while SmolVLA (50 actions) did not stall. Expect GR00T to pause often
when served from the A100 or H200; the adaptive playback of condition 4 is what fixes that (24% stalls).
