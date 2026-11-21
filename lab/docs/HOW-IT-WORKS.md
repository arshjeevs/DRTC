# How the whole project works

*Written 2026-11-21. This file explains **what** each stage is and **why** it is done that way, following one new
dataset from recording to the report. The exact commands, with every option, are in [../WORKFLOW.md](../WORKFLOW.md);
this file links to the right part of it at each stage. The plan and what is done: [../PLAN.md](../PLAN.md). What every
number means: [../METRICS.md](../METRICS.md).*

The example throughout: a new task, **sorting B** (both cubes into their squares), dataset `so101_sort_two_v1`.

---

## 1. The big picture

```
  A. RECORD               B. TRAIN                 C. EVALUATE (three different kinds)              D. REPORT
  ─────────               ────────                 ───────────────────────────────────              ─────────
  you + the leader arm    a GPU server             C1 offline    no robot, no you, minutes          numbers, tables,
  teach the task          learns a model           C2 robot      the real arm, you judge, hours     graphs, slides
  110 episodes            per policy               C3 bench      no robot, no you, the network      (Claude)
       │                       │                                 faults one at a time
       ▼                       ▼                         ▼
  dataset on the Hub  ──▶ models on the Hub   ──▶  results store (orangemoon599/drtc-results)  ──▶  report/
```

Everything is passed on through two places, never by copying files by hand:

- **The Hugging Face Hub**: the datasets (`orangemoon599/so101_...`), the trained models (`orangemoon599/<model>`),
  and the results store (`orangemoon599/drtc-results`, private).
- **git** (this repo): the small text files that say what exists and what to do: `lab/datasets/recorded.yaml`
  (datasets), `lab/models/*.yaml` (models), `lab/evals/*.yaml` (evaluations). Every machine `git pull`s them. That
  is why every stage ends with **commit and push**.

## 2. Words used everywhere

| Word | Means | Example |
|---|---|---|
| **episode** | one recorded demonstration of the task, from start to end | episode 12: you moved both cubes once |
| **held-out episodes** | episodes recorded like the others but **never used for training**, kept aside to test the model on situations it has not seen | 10 of the 110 |
| **policy** | a kind of model (an architecture) | ACT, SmolVLA, GR00T N1.5, π0.5 |
| **recipe** | the fixed training settings of a policy, copied from its authors (`lab/recipes/`) | `act_original`: 100k steps, batch 8 |
| **model** | one policy trained on one dataset with its recipe; it has a name and a Hub repo | `act_sort_two_v1_full` |
| **chunk** | a model does not answer with one action but with a list of the next ones (e.g. 50 actions = 1.7 s at 30 Hz) | SmolVLA: 50, GR00T N1.5: 16 |
| **model server** | a GPU machine that runs the model: it receives camera images + joint positions, answers with a chunk | the A100, the H200, the laptop GPU |
| **robot computer** | the computer with the arm and cameras attached; it asks the server for chunks and plays them on the arm | the laptop (the Pi for edge tests) |
| **condition** | one way of running the robot side and the network (`lab/experiments/main/`) | `baseline`, `drtc_faults` |
| **eval** | one study: which models, which conditions, how many trials, the success rule (`lab/evals/<eval>.yaml`) | `sort_one_baseline` |
| **frozen** | the eval pins every model to one exact version on the Hub, so every trial uses the same weights | `lab eval freeze` |
| **trial** | one attempt: start, the model runs, it ends in success or failure | `t03` |
| **round** | one trial of every model under every condition, in a random order, with one layout | 7 trials in a 7-model baseline |
| **session** | one sitting of trials on one robot computer with one server; any number of rounds | `2026-11-22-laptop` |

## 3. Stage A: collect a new dataset

**What happens.** You move the leader arm; the follower arm copies it; the laptop records the follower's joint
positions (the actions) and the two cameras (scene + wrist) 30 times per second. Each episode is one demonstration.

**Why each rule exists:**

| Rule | Why |
|---|---|
| Decide name, instruction, episode count, times **before** recording | the instruction text goes into every episode; a model trained on mixed texts learns noise |
| Record **N + K** episodes (100 + 10) | the K held-out ones are recorded in the same conditions, so testing on them is fair (C1 and C3 use them) |
| Rotate the layouts evenly, in a fixed order, over all N + K | otherwise the model can learn "the red square is always on the left" instead of looking |
| A new version for a new setup (`_v2`), never reuse a name | results always say exactly which data a model saw |
| Edit (delete, trim) **before** training | deleting episodes renumbers them and changes which are held out; models trained before and after would not be comparable |

**What you run** ([WORKFLOW Part 1, steps 1-5](../WORKFLOW.md#part-1-from-a-new-dataset-to-trained-models)):

```bash
lab/bin/session record                    # phone camera bridge + rig check
lab record so101_sort_two_v1 --task "Put the red cube in the red square and the blue cube in the blue square." \
    --episodes 100 --held-out 10 --episode-time 45 --reset-time 15
lab data show so101_sort_two_v1           # look for warnings (!), very short or long episodes
git add lab/datasets/recorded.yaml && git commit -m "so101_sort_two_v1: 110 episodes" && git push
```

**What you get.** Two datasets on the Hub: `so101_sort_two_v1_raw` (as recorded) and `so101_sort_two_v1` (idle
start and end trimmed automatically; this one is used for training). `recorded.yaml` records the exact Hub version
and which 10 episodes are held out.

## 4. Stage B: train the models

**What happens.** A GPU server downloads the dataset, prepares it once (e.g. relative-action copies for π0.5 and
GR00T N1.7), and trains: it shows the model the 100 training episodes over and over, and checks its error on the
10 held-out ones from time to time. At the end it pushes the model to the Hub.

**Why the recipes are fixed.** Every policy is trained the way its authors trained it. If a model does badly, it is
the model, not our tuning; and the comparison between policies is fair.

**What you run** ([WORKFLOW Part 1, steps 6-9](../WORKFLOW.md#6-add-the-models)):

1. On the laptop: add the models to `lab/models/sort_two_v1.yaml` (one line per policy), `lab check`,
   `lab train <model> --dry-run` (shows the dataset version and the held-out episodes), commit, push.
2. On the H200 or the A100: `git pull`, `lab prepare so101_sort_two_v1`, then `lab train <model>` in tmux, one
   window per model.
3. Watch the loss in wandb (project `drtc-lab`).

**What you get.** `orangemoon599/act_sort_two_v1_full` etc. on the Hub: the repo root is the **best checkpoint by
held-out loss** (this is what gets served), `checkpoints/<step>/` the last one(s). Training takes from ~1 h (SmolVLA)
to ~5 days (MolmoAct 2).

## 5. Stage C: three kinds of evaluation

This is the part that is easy to mix up. All three test a trained model, but they answer different questions.

| | C1 Offline | C2 Robot eval | C3 Bench |
|---|---|---|---|
| **Question** | Did the model learn the task's movements? How fast and big is it? | Does the robot **do the task**, under each condition? | What does **each network fault** do to the motion, and how well does DRTC handle it? |
| **Real arm moves?** | no | **yes** | no |
| **Needs you?** | no | **yes** (you set up, you judge success) | no |
| **Network** | none (model on the GPU directly) | the real one (laptop → WiFi → server) | simulated on one machine (250 ± 20 ms, as measured on the real one) |
| **Input to the model** | frames of the held-out episodes | the live cameras | frames of the held-out episodes, played as if live |
| **Measures** | M9 action error, M10 inference time + GPU memory | **M1 success**, M1f faults, M1t time, + M2-M8 | M2-M8 (stalls, latency, jerk, chunk jumps); **no success** |
| **Time** | minutes per model | ~1 min per trial; 70 trials ≈ 1.5-2 h | ~20 min per bench, runs by itself |
| **Command** | `lab eval offline` | `lab eval serve` + `lab eval run` | `lab eval serve` + `lab eval run --server local` |
| **Results** | `results/offline/<dataset>/<model>.json` | `results/evals/<eval>/` | `results/bench/<eval>/` |

### C1. Offline evaluation (M9, M10): "is the model sane?"

```
  held-out episode 49, frame 120:  camera images + joint positions  ──▶  model  ──▶  predicted next 15 actions
                                   what you actually did next (recorded 15 actions)  ──┘  compare: error in degrees
```

The model looks at a moment of a held-out episode it has never seen and predicts what to do next; that is compared
with what you did in the recording. Done from every 10th frame of every held-out episode. A low error means it learnt
the movements; it does **not** prove the task succeeds (small errors can add up on the real arm), which is why C2
exists. It also times 200 predictions and measures the GPU memory (M10), which tells you which server can serve it.

When: right after each training, before anything else. It is cheap, needs nobody, and catches a broken model before
you spend robot time on it.

```bash
# on the H200 (in the container) or the A100
.venv/bin/lab eval offline --models act_sort_two_v1_full,smolvla_sort_two_v1_full
```

### C2. Robot evaluation: "does it do the task?"

This is the main result. The real system, end to end:

```
   LAPTOP (robot computer)                                    GPU SERVER (model server)
   ┌───────────────────────────┐   camera images + joints     ┌────────────────────────┐
   │ follower arm, 2 cameras   │ ───────── over WiFi ───────▶ │ the model              │
   │ plays the actions at 30 Hz│ ◀──────── a chunk ────────── │ (lab eval serve)       │
   │ (lab eval run)            │   of the next actions        └────────────────────────┘
   └───────────────────────────┘
         ▲ you: set the layout, press Enter, then s (success) or f (failure)
```

A round trip takes ~300-450 ms, so the robot always asks for the next chunk **before** the current one runs out. How
it does that is exactly what the **conditions** change:

| Condition | Robot side | Network |
|---|---|---|
| `baseline` | upstream LeRobot as it ships; gives up after 1 s without actions | normal |
| `baseline_faults` | the same | **the fault mix**: +1 s spike at 2 s, every reply lost 4-5 s, full outage 6-7.5 s |
| `drtc_faults` | **DRTC**: estimates the latency, asks in time, holds the arm through gaps, recovers | the fault mix |
| `drtc_smooth_faults` | DRTC + the Butterworth filter + adaptive playback (smoothing) | the fault mix |

The claim of the project is read from comparing them: `baseline_faults` vs `baseline` (faults hurt),
`drtc_faults` vs `baseline_faults` (DRTC recovers), `drtc_smooth_faults` vs `drtc_faults` (smoothing helps).

**Two kinds of robot eval**, both plain eval files:

- a **baseline** eval per dataset: all models, the one condition `baseline`. Answers "which policy does this task
  best?" (e.g. `sort_one_baseline`, 7 models x 10 trials).
- a **DRTC experiment** eval: two chosen models, the four conditions (e.g. `sort_one_drtc_main`, 2 x 4 x 10).

**Why the eval file, the freezing and the rounds:**

| Rule | Why |
|---|---|
| Everything is written in the eval file before the first trial (models, conditions, trials, success rule) | you can't change the test after seeing results; a change = a new eval |
| Freeze: every model pinned to one Hub commit | all trials, on any day, any server, use exactly the same weights |
| Rounds: every model once per round, random order, one layout per round | the models see the same layouts, the same lighting, the same tiredness of the operator; nobody is favoured |
| The success rule is written down | you press `s` by the rule, not by feel; close calls stay rare |
| `v` (void) only for a setup mistake, never for a model's failure | a void trial is run again; using it for failures would hide them |
| Models are compared only inside one eval | trials of different evals were run on different days and setups |

**What you run** (each session; [WORKFLOW, step 5](../WORKFLOW.md#step-5-the-robot-baselines)):

```bash
# once per eval, laptop: write lab/evals/sort_two_baseline.yaml (copy sort_one_baseline.yaml), then
lab eval check sort_two_baseline && lab eval freeze sort_two_baseline
git add lab/evals/sort_two_baseline.yaml && git commit -m "eval sort_two_baseline: frozen" && git push

# each session: on the server (H200 container)
tmux new -s serve '.venv/bin/lab eval serve sort_two_baseline; bash'      # wait for "all N ready"
# on the laptop
lab/bin/session record
lab eval servers sort_two_baseline --server h200
lab eval run sort_two_baseline --server h200                              # rounds; s / f; q to stop
lab eval report sort_two_baseline --pull                                  # results/evals/sort_two_baseline/report.md
```

`--test` (real trials that don't count, kept in `results/scratch/`) and `--replay` (one round on the replay robot,
the arm doesn't move) are setup checks only.

### C3. Bench: "what does each fault do, exactly?"

The robot eval runs the faults as one mix and only 10 times per condition, with you in the loop. To see each fault
type on its own, many times, the same pipeline runs **without the robot**:

```
   ONE GPU SERVER (A100 or H200)
   ┌───────────────────────────────────────────────────────────────────────────────────────┐
   │ replay robot: plays held-out episode 49 as if it were the live cameras + joints       │
   │      │  ▲                                                                             │
   │      ▼  │  simulated network: 250 ± 20 ms per reply (what WiFi cost on the real robot) │
   │      │  │  + ONE fault: only the spike, or only the drops, or only the outage         │
   │  model server (same machine)                                                          │
   └───────────────────────────────────────────────────────────────────────────────────────┘
```

The robot side (baseline client or DRTC, with or without smoothing) runs exactly as on the laptop and records the
same timing and motion files. Nothing moves and the images don't react to the actions, so there is **no success**:
only M2-M8 (how often the arm would stall, how long, latency estimates, jerk, jumps between chunks). Each trial
replays the next held-out episode, so `trials:` = the number of held-out episodes (8 for sort_one).

| Bench eval | Fault | Why |
|---|---|---|
| `bench_main_spike` | the +1 s latency spike alone | how each condition handles a slow reply |
| `bench_main_drops` | the lost replies alone | how each condition handles lost messages |
| `bench_main_outage` | the full outage alone | how each condition handles a dead network |
| `bench_main_mix` | the whole mix, as on the robot | checks that the bench behaves like the robot |

When: after the robot DRTC experiment, for the same two models (the bench's 250 ms comes from that experiment's
measurements). It runs by itself in ~20 min per bench; run it again only if the DRTC code or the smoothing changes.

For a new dataset: copy the four `lab/evals/bench_main_*.yaml` to `bench_sort_two_*.yaml`, change the two `models:`,
set `trials:` to its number of held-out episodes, delete the `frozen:` block, then check, freeze, commit, push. The
condition files in `lab/experiments/bench_main/` are reused as they are (the `model:` inside them is replaced by
each eval model).

```bash
# on the A100 (or in the H200 container), one bench at a time
tmux new -s bench
.venv/bin/lab eval serve bench_sort_two_spike                  # window 1: wait for "all 2 ready"
.venv/bin/lab eval run bench_sort_two_spike --server local     # window 2 (Ctrl-b c): runs every trial, pushes
```

## 6. Stage D: results and the report

Every official trial is saved with all its raw files (actions per tick, chunks, faults, video) under
`results/evals/<eval>/sessions/<date>-<machine>/`, and pushed to the results store. `lab eval report <eval> --pull`
gathers every machine's sessions and rebuilds `report.md`, `trials.csv` and `metrics.csv` from the raw files; the
offline results of the eval's models are added in their own section. Never edit these by hand: they are rebuilt each
time.

Then tell Claude which eval or training finished: `lab/REPORT.md`, the LaTeX report (`report/`, `make`) and the
slides are updated from those files.

## 7. A new dataset from start to end: the checklist

| # | Stage | Where | Command(s) | Done when |
|---|---|---|---|---|
| 1 | Plan the dataset | paper | name, instruction, N + K, times, layout rotation | written down |
| 2 | Record | laptop | `lab/bin/session record`, `lab record ...` | it pushed both datasets |
| 3 | Review, edit | laptop | `lab data show / view / delete / trim`, `lab data push` | no bad episodes left |
| 4 | Commit the dataset | laptop | `git add lab/datasets/recorded.yaml`, commit, push | pushed |
| 5 | Add the models | laptop | `lab/models/sort_two_v1.yaml`, `lab check`, `--dry-run`, commit, push | dry runs show 100 + 10 |
| 6 | Train | H200 / A100 | `git pull`, `lab prepare`, `lab train <model>` (tmux) | each model on the Hub |
| 7 | Offline (C1) | H200 / A100 | `lab eval offline --models ...` | a json per model |
| 8 | Baseline eval file | laptop | copy `sort_one_baseline.yaml`, edit, `check`, `freeze`, commit, push | frozen |
| 9 | Robot baseline (C2) | server + laptop | `lab eval serve`, `lab eval run` (sessions until done) | all trials |
| 10 | DRTC experiment (C2), if wanted | server + laptop | an eval like `sort_one_drtc_main` with 2 models | all trials |
| 11 | Bench (C3), if wanted | A100 / H200 | 4 bench evals like `bench_main_*` | 4 x 64 trials |
| 12 | Report | laptop | `lab eval report <eval> --pull`, tell Claude | updated |

Steps 7 and 8 can run while other models are still training; step 8's `freeze` waits until every model of the eval
is on the Hub. How to change an eval later (drop a model, more trials): [WORKFLOW, Changing the
plan](../WORKFLOW.md#changing-the-plan).
