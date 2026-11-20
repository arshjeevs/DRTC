# Workflow

- **[The whole path at a glance](#the-whole-path-at-a-glance)** (below): new dataset → trained models → evaluated
  against the current baseline, on one page.
- **[Part 1: from a new dataset to trained models](#part-1-from-a-new-dataset-to-trained-models)**: record, review,
  register, train, with every command.
- **[Part 2: evaluation](#part-2-evaluation)**: official robot trials, results, reports, with every command.

# The whole path at a glance

```
 A. RECORD (laptop)              B. TRAIN (H200)                 C. EVALUATE (server + laptop)
 ─────────────────────           ─────────────────────           ─────────────────────────────────────
 plan the dataset                lab/models/<task>_v1.yaml       lab/evals/<eval>.yaml
 lab record <name> ...           lab check, --dry-run            lab eval check / freeze
 lab data show / edit            commit + push                   commit + push
 commit recorded.yaml            H200: pull, prepare, train      server: lab eval serve <eval>
                                 → models on the Hub             laptop: lab eval run <eval> --server ...
                                                                 lab eval report <eval> --pull
```

Each step ends with a **commit and push**. That's how the next machine learns about it: the GPU server learns about
the new dataset from `recorded.yaml`, and every machine learns about the eval from `lab/evals/<eval>.yaml`.

### A. Record a new dataset (laptop) → Part 1, steps 1-5

1. **Plan:**
   - the name `so101_<task>_v<N>`;
   - one fixed instruction;
   - N training + K held-out episodes, about 10% held out (`--episodes 100 --held-out 10` records 110);
   - the layout rotation, on paper.
2. **Record:** `lab/bin/session record`, then `lab record <name> --task "..." --episodes N --held-out K ...`.
   Continue later with the same command plus `--resume`.
3. **Review:** `lab data show <name>`; fix with `lab data delete/trim/undo`, then `lab data push`. Edit only
   **before** training.
4. **Commit `lab/datasets/recorded.yaml`** and push.

### B. Train (H200) → Part 1, steps 6-9

1. **Laptop:** write `lab/models/<task>_v1.yaml`, one line per model, e.g. `act_<task>_v1_full` with recipe
   `act_original`. Run `lab check` and `lab train <model> --dry-run`, then commit and push.
2. **H200 container:** `git pull && uv sync ...`, then `lab prepare <name>`, then one tmux window per model with
   `lab train <model>`, ACT first.
3. **Watch** in wandb (`drtc-lab`). At the end each model pushes itself to `orangemoon599/<model>`. Set
   `status: trained` in the model file, commit and push.

### C. Evaluate against the current baseline → Part 2

**The one rule to remember:** models are only compared **inside one eval**, run alternately in the same sessions
(METRICS.md rule 5). A frozen eval that has trials is never changed. So a new model gets compared by **making a
new eval that contains both the new models and the ones to beat.**

| Situation | Make this eval | Models in it |
|---|---|---|
| **New task** (e.g. sorting B, `so101_sort_two_v1`) | `sort_two_baseline` | the new models (ACT, SmolVLA, GR00T, Diffusion on sort_two). This *is* the task's baseline |
| **Same task, new model or more data** (e.g. a better SmolVLA on sorting A) | `sort_one_baseline_v2` | the new model(s) **plus** the current best from `sort_one_baseline`, re-run side by side |
| **Same models, a new method** (Phases 7-8: smoothing, faults) | `sort_one_smoothing`, `sort_one_faults` | the chosen models; the conditions are `baseline` + the methods. Each method is compared with `baseline` in the same eval |

Steps:
1. **Laptop:** copy an existing eval, e.g. `cp lab/evals/sort_one_baseline.yaml lab/evals/sort_two_baseline.yaml`.
   Edit `description`, `success` (the rule you'll apply when pressing `s`), `time_limit_s` and `models`. Then run
   `lab eval check <eval>` and `lab eval freeze <eval>`, commit and push.
2. **Model server**, the A100 or H200, whichever is free: `git pull`, then
   `tmux new -s serve '.venv/bin/lab eval serve <eval>; bash'`. Wait for `all N ready`.
3. **Laptop with the robot:**
   - `git pull` and `lab/bin/session record`;
   - `lab eval servers <eval> --server <a100|h200>`: all `ready`?
   - one `--test` trial to check the keys;
   - `lab eval run <eval> --server <a100|h200>`: 20 trials per model, in rounds. Set one layout per round. During a
     trial press `s` for success or `f` for failure; `q` stops, and the next run continues where it stopped.
4. **Results:** `lab eval report <eval> --pull` gives `results/evals/<eval>/report.md`. Then stop the servers
   (Ctrl-C in their tmux window) and tell Claude the eval is done.

What never changes: the recipes (`lab/recipes/`), the metrics (`lab/METRICS.md`), 20 trials per model, and the rule
that only `lab eval run` makes official results. Anything you try by hand is a test (`results/scratch/`).


# Part 1: from a new dataset to trained models

The whole path, in order, with every command. Details on each part: [DATA.md](DATA.md) (recording and editing),
[TRAIN.md](TRAIN.md) (training). The example throughout is sorting B, `so101_sort_two_v1`.

```
laptop:  plan → set up the scene → record → review / edit → commit recorded.yaml
         → add the models → check → commit + push
H200:    pull → prepare → train (tmux) → watch in wandb → models on the Hub
```

## Cheat sheet

```bash
# --- laptop ---
cd ~/Projects/drtc-major && source .venv/bin/activate
lab/bin/session record                          # phone camera + checks, in tmux "robot"
lab record so101_sort_two_v1 --task "Put the red cube in the red square and the blue cube in the blue square." \
    --episodes 100 --held-out 10 --episode-time 45 --reset-time 15
lab data show so101_sort_two_v1
git add lab/datasets/recorded.yaml && git commit -m "so101_sort_two_v1: 110 episodes (100 + 10)" && git push
# write lab/models/sort_two_v1.yaml (step 6)
lab check && lab train act_sort_two_v1_full --dry-run
git add lab/models/sort_two_v1.yaml && git commit -m "models: sort_two_v1" && git push

# --- H200, in the drtc_lab container ---
cd /workspace/drtc_major && . ./.server.env && git pull && uv sync --frozen --extra gpu --python /usr/bin/python3
.venv/bin/lab prepare so101_sort_two_v1
nvidia-smi
tmux new -d -s sort2 -n act '.venv/bin/lab train act_sort_two_v1_full; bash'
sleep 60
tmux new-window -t sort2 -n groot   '.venv/bin/lab train groot_sort_two_v1_full; bash'
tmux new-window -t sort2 -n smolvla '.venv/bin/lab train smolvla_sort_two_v1_full; bash'
```

---

## 1. Plan the dataset

Decide these before the session and write them down (the plan for each task is in [PLAN.md](PLAN.md), Phase 4):

| What | Rule | Sorting B |
|---|---|---|
| Name | `so101_<task>_v<N>`; a new version for a new setup, never reuse a name | `so101_sort_two_v1` |
| Instruction | one sentence, identical for every episode (copy it, don't retype it) | "Put the red cube in the red square and the blue cube in the blue square." |
| Episodes | N training + K held out (K about 10% of N); records N + K | 100 + 10 = 110 |
| Times | episode time long enough for a slow episode; reset time to put everything back on the marks | 45 s / 15 s |
| Rotation | every layout combination equally often over **all N + K** episodes, in a fixed order | 4 combinations x 27-28 |

The rotation is kept by hand (nothing records the layout). Write it on paper and tick off the episodes. With 110
episodes and 4 combinations, it can't be perfectly even (27, 28, 27, 28): that's fine, keep the order fixed.

## 2. Set up the scene and the rig (laptop)

1. Squares, cubes and spots on their taped marks; the scene camera (phone) and the wrist camera in place; both arms
   powered and connected over USB; the phone connected over USB with DroidCam open.
2. Start the session:

   ```bash
   cd ~/Projects/drtc-major && source .venv/bin/activate
   lab/bin/session record
   ```

   This opens tmux `robot` with the phone camera bridge in one window and the checks in the other. Re-running it
   re-attaches. Without the session, start the bridge yourself with `lab/bin/phonecam` in its own terminal.
3. Check the arms, ports, calibration and cameras (the session already does this; `lab record` checks again):

   ```bash
   lab check --rig
   ```

   Every line should say `ok`. Never recalibrate for a new dataset: the calibration stays the same for all of them.

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
- Tired? Press Esc, and continue another time with the **same command plus `--resume`**:

  ```bash
  lab record so101_sort_two_v1 --task "Put the red cube in the red square and the blue cube in the blue square." \
      --episodes 100 --held-out 10 --episode-time 45 --reset-time 15 --resume
  ```

  It prints how many episodes are recorded and how many are left. Go on in the rotation from the next episode number.

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

The GPU servers learn about the dataset (its exact Hub revision and its held-out split) from `recorded.yaml`:

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
# Sorting B (so101_sort_two_v1): original recipes, full fine-tunes.
act_sort_two_v1_full: {recipe: act_original, dataset: so101_sort_two_v1, status: planned,
  notes: 'ACT on so101_sort_two_v1; original recipe'}
smolvla_sort_two_v1_full: {recipe: smolvla_authors, dataset: so101_sort_two_v1, status: planned,
  notes: 'SmolVLA on so101_sort_two_v1; original recipe (action expert only)'}
groot_sort_two_v1_full: {recipe: groot_n15_nvidia, dataset: so101_sort_two_v1, status: planned,
  notes: 'GR00T N1.5 on so101_sort_two_v1; original recipe'}
```

- Name: `<policy>_<task>_<version>_full`; the dataset is the **trimmed** name (no `_raw`).
- Recipes: `act_original`, `smolvla_authors`, `groot_n15_nvidia` (the three in use). Never edit a recipe in
  `lab/recipes/`: they're fixed so every dataset is trained the same way. A new recipe or model is a question for
  Claude.
- `status` is a note only (`planned` → `trained` once it's on the Hub); nothing depends on it.

Check, then commit and push:

```bash
lab check                                        # the registry is consistent
lab train act_sort_two_v1_full --dry-run         # dataset revision, held-out episodes, full command; downloads nothing
lab train groot_sort_two_v1_full --dry-run
lab train smolvla_sort_two_v1_full --dry-run
git add lab/models/sort_two_v1.yaml
git commit -m "models: ACT, SmolVLA, GR00T N1.5 on so101_sort_two_v1"
git push
```

The dry run should show the held-out count you recorded (`held out: [...]`, 10 episodes) and
`episodes: 100 for training`.

## 7. Train on the H200

In the `drtc_lab` container (Jupyter terminal, or `docker exec -it drtc_lab bash`; first-time setup is in
[H200.md](H200.md)):

```bash
cd /workspace/drtc_major && . ./.server.env && git pull && uv sync --frozen --extra gpu --python /usr/bin/python3
.venv/bin/lab check --logging                  # wandb and HF keys present (once per setup is enough)
.venv/bin/lab prepare so101_sort_two_v1        # download + prepare, once per new dataset revision
nvidia-smi                                     # the GPU is shared: check there is room
```

Start the three runs, ACT first (the CPU limits in `lab/hosts.yaml` apply to runs started while another one
is training, and ACT needs the least):

```bash
tmux new -d -s sort2 -n act     '.venv/bin/lab train act_sort_two_v1_full; bash'
sleep 60
tmux new-window -t sort2 -n groot   '.venv/bin/lab train groot_sort_two_v1_full; bash'
tmux new-window -t sort2 -n smolvla '.venv/bin/lab train smolvla_sort_two_v1_full; bash'
tmux attach -t sort2          # Ctrl-b n: next window; Ctrl-b d: detach (the runs go on)
```

Use a new tmux session name per dataset (`sort2`, `stack`, ...) so it doesn't clash with a running one. Closing the
browser or logging out doesn't stop the runs.

Approximate lengths, one run alone on the H200: ACT ~3 h, SmolVLA ~1.1 h, GR00T N1.5 ~1.4 h. Three at once take
longer each.

## 8. Watch

- **wandb**, project `drtc-lab`: training loss, learning rate and held-out loss, from any machine.
- The tmux windows: progress bar with the time left.
- Normal noise at the start (not errors): transformers deprecation lines, `min/max_dynamic_tiles`, and for GR00T a
  "tokenizer ... incorrect regex pattern ... `fix_mistral_regex=True`" warning (its tokenizer is Qwen, not Mistral).
- A real problem: a traceback, `CUDA out of memory`, or the window back at the shell prompt before the end.

Stopping and continuing:

```bash
.venv/bin/lab stop act_sort_two_v1_full        # clean stop: checkpoint, push, exit (or Ctrl-C in its window)
.venv/bin/lab train act_sort_two_v1_full --resume
```

## 9. Done

Each run pushes to `orangemoon599/<model>` at its end: the repo root is the best checkpoint by held-out loss (what
`lab serve` uses), `checkpoints/<step>/` the last one with its training state. On the laptop, set `status: trained`
in the model file, commit and push. Then tell Claude: the offline evaluation and robot trials come next (Phase 5-6
in [PLAN.md](PLAN.md)).

## When to ask Claude

- Anything that deletes or replaces: `outputs/train/<model>` already exists, a stale dataset copy on a server,
  `lab data remove --hub`.
- A traceback, out-of-memory that persists after `nvidia-smi` shows the GPU free, a dry run that looks wrong.
- A new task with per-episode instructions (sorting C), a new recipe or model, a new server.

Paste the last lines of the window (or the dry run's output) with the question.

---

# Part 2: evaluation

Official robot trials, from the plan to the report. Why it works this way: [EVALUATION.md](EVALUATION.md). What
every number means: [METRICS.md](METRICS.md). The example throughout is `sort_one_baseline` (Phase 6: ACT,
SmolVLA, GR00T N1.5 and Diffusion on sorting A, 20 trials each).

```
once per eval    laptop: write lab/evals/<eval>.yaml → lab eval check → lab eval freeze → commit + push
each session     model server (A100 / H200 / laptop GPU): lab eval serve <eval>        (leave it running)
                 laptop (robot):  lab eval servers → lab eval run <eval> --server <host>  (s / f per trial)
any time         lab eval status <eval>   ·   lab eval report <eval> --pull
```

Only `lab eval run` makes official results. Anything else you try (`lab run`, `--test`, `--replay`) goes to
`results/scratch/` and never into the results.

## Cheat sheet

```bash
# --- once per eval (laptop) ---
cd ~/Projects/drtc-major && source .venv/bin/activate && git pull
lab eval check sort_one_baseline
lab eval freeze sort_one_baseline
git add lab/evals/sort_one_baseline.yaml && git commit -m "eval sort_one_baseline: frozen" && git push

# --- each session: 1. the model server (pick one) ---
# A100:   ssh b230968ee@192.168.5.13, then
cd ~/drtc_major && git pull && tmux new -s serve '.venv/bin/lab eval serve sort_one_baseline; bash'
# H200:   in the drtc_lab container
cd /workspace/drtc_major && . ./.server.env && git pull && uv sync --frozen --extra gpu --python /usr/bin/python3
tmux new -s serve '.venv/bin/lab eval serve sort_one_baseline; bash'
# laptop GPU (8 GB: the small models only)
lab eval serve sort_one_baseline --models act_sort_one_v1_full,diffusion_sort_one_v1_full

# --- each session: 2. the laptop with the robot ---
cd ~/Projects/drtc-major && source .venv/bin/activate && git pull
lab/bin/session record                                   # phone camera + rig check (tmux "robot")
lab eval servers sort_one_baseline --server a100         # every model "ready"?
lab eval run sort_one_baseline --server a100 --replay    # optional: 1 round on the replay robot, no arm moves
lab eval run sort_one_baseline --server a100             # the official session

# --- afterwards (any machine) ---
lab eval status sort_one_baseline
lab eval report sort_one_baseline --pull                 # → results/evals/sort_one_baseline/report.md
```

---

## 1. Write the eval (once per study)

One file per study in `lab/evals/`. Copy `lab/evals/sort_one_baseline.yaml` and change it:

```yaml
description: Phase 6 baseline, sorting A (every trained model, as trained)
success: The cube is fully inside the square of its colour and the gripper has let go of it, before the time limit.
time_limit_s: 30          # a trial that runs out is a failure
trials: 20                # per model per condition (the standard)
seed: 0                   # the order inside each round
models: [act_sort_one_v1_full, smolvla_sort_one_v1_full, groot_sort_one_v1_full, diffusion_sort_one_v1_full]
conditions:
  baseline: replication/baseline      # settings from lab/experiments
video: true
```

- **Decide everything before the first trial.** Once trials exist, the plan can't change. A change means a new eval
  (e.g. `sort_one_baseline_v2`), and `lab eval run` refuses a plan that differs from earlier sessions.
- **`success`** is the rule you apply when you press `s`. Write it so that close calls are rare.
- **`conditions`** are what you compare: just `baseline` in Phase 6. In Phases 7–8, e.g. `baseline`, `smoothing_a` and
  `spikes`, each pointing at an experiment file in `lab/experiments/`.

Then:

```bash
lab eval check sort_one_baseline      # names exist, conditions point at real experiments, task found
lab eval freeze sort_one_baseline     # pins every model to its current Hub commit, written into the file
git add lab/evals/sort_one_baseline.yaml && git commit -m "eval sort_one_baseline: frozen" && git push
```

Freezing is what makes "the same model" exact: the laptop, the Pi and every server use that commit, and a server
serving any other version simply can't be reached by the session.

## 2. Start the model server

Pick whichever server is free that day. **The whole session uses one server**, and each trial records which one it was.

| Server | Where to type | Command |
|---|---|---|
| A100 | `ssh b230968ee@192.168.5.13` | `cd ~/drtc_major && git pull && tmux new -s serve '.venv/bin/lab eval serve sort_one_baseline; bash'` |
| H200 | the `drtc_lab` container ([H200.md](H200.md)) | `cd /workspace/drtc_major && . ./.server.env && git pull && uv sync --frozen --extra gpu --python /usr/bin/python3`, then `tmux new -s serve '.venv/bin/lab eval serve sort_one_baseline; bash'` |
| laptop GPU | a laptop terminal | `lab eval serve sort_one_baseline --models act_sort_one_v1_full,diffusion_sort_one_v1_full` |

It starts the models one after the other and prints each one's status:

```
  act_sort_one_v1_full             port 8840  starting (loading the model) ... ready in 30 s
  smolvla_sort_one_v1_full         port 8841  starting (loading the model) ... ready in 547 s
  groot_sort_one_v1_full           port 8842  starting (loading the model) ... ready in 159 s
  diffusion_sort_one_v1_full       port 8843  starting (loading the model) ... ready in 34 s
all 4 ready. Leave this running during the session; Ctrl-C stops every server.
```

- Wait for **`all N ready`**. SmolVLA compiles its model when it starts (about 9 minutes the first time on the A100).
- Detach with Ctrl-b d. At the end of the session, attach (`tmux attach -t serve`) and press Ctrl-C.
- The laptop's 8 GB fits ACT and Diffusion, not GR00T. Then run the session with the same `--models`.
- Each eval has its own ports, so two evals can be served on one machine at once if the GPU has room. If
  `lab eval serve` says `ports in use`, that eval is already being served there: `tmux ls`, attach, Ctrl-C.
- On the H200, the laptop connects through an SSH tunnel into the container: type the server password when
  `lab eval servers` or `lab eval run` asks for it (once per session). **Not tested on the H200 yet:** if it fails the
  first time, paste the message to Claude.

## 3. Prepare the robot (laptop)

```bash
cd ~/Projects/drtc-major && source .venv/bin/activate && git pull
lab/bin/session record          # phone camera bridge + rig check; every line ok
lab eval servers sort_one_baseline --server a100
```

`lab eval servers` connects (A100: by key; H200: asks the password) and asks each model to describe itself:

```
  act_sort_one_v1_full             ready   commit 449b9c12  checkpoint sha256:5af906278031  tcp/127.0.0.1:18840
  ...
```

Every model must say `ready`. Optional setup check without moving the arm: one round on the replay robot against the
same servers (a test, in scratch):

```bash
lab eval run sort_one_baseline --server a100 --replay
```

## 4. Run the session

```bash
lab eval run sort_one_baseline --server a100
```

It checks everything first and refuses to start when something is off: uncommitted changes, a changed plan, a
failed rig check, or a model that isn't served. Then it pulls the sessions other machines pushed, so the counts
include them.

It works in **rounds**. A round is one trial of every model, in a random order (the same order on every machine).

```
=== Round 1: 4 trials ===
Set this round's layout (the same for every trial of the round).

Round 1, trial 1/4: groot_sort_one_v1_full, t01. Put the cube(s) on the marks, press Enter to start ...
  [success] ending the trial
  t01: SUCCESS in 9.4 s (s; completed)
  Enter = next trial, v = void this trial (setup mistake), q = end the session:
```

| When | Key | Means |
|---|---|---|
| before a trial | Enter | start (cube on its mark) |
| before a trial | `q` | end the session |
| during a trial | `s` | **success**, right now: the task is done by the success rule. Ends the trial; the time is the time to success |
| during a trial | `f` | **failure**: it won't succeed. Ends the trial early |
| during a trial | (nothing) | the time limit ends it: failure |
| after a trial | Enter | next trial |
| after a trial | `v` | **void**: a setup mistake (cube not on its mark, wrong layout, cable caught), with a reason. Kept, not counted, run again in a later round. **Never for a model's failure** |
| after a trial | `q` | end the session |

- **Layouts are yours:** set one layout per round, all trials of the round use it, and rotate the layouts over the
  rounds as in the recording (sorting A: the 4 combinations, 5 rounds each over 20 rounds).
- A trial that ends by itself with a fault (the server stopped, the network dropped) is **not counted** and comes back
  in a later round. It's listed in the report.
- Stop whenever you like (`q`). The next `lab eval run` continues where the eval stands, on any machine.
- At the end, the session is **pushed** to the results store automatically. Without internet, use `--offline`, then
  `lab eval push sort_one_baseline` later.

## 5. Results

```bash
lab eval status sort_one_baseline           # trials done per model, the next round, unpushed sessions
lab eval report sort_one_baseline --pull    # rebuild the report from every session (all machines)
```

`results/evals/sort_one_baseline/`:

| File | What |
|---|---|
| `report.md` | the tables: success rate with its 95% interval and time to success; stalls, latency, server time, jerk, chunk jumps (median and spread); comparisons with the baseline condition; trials not counted; sessions |
| `trials.csv` | one row per trial (also voided ones), with its folder |
| `metrics.csv` | one row per counted trial, every metric |
| `sessions/<date>-<machine>/trials/<condition>/<model>/tNN/` | the raw files of one trial, including `video/scene.mp4` and `video/wrist.mp4` |

The report is rebuilt every time, so never edit it. Tell Claude when an eval is done: the next step reads the report.

## Mistakes and fixes

| Situation | Do |
|---|---|
| Realised after a trial that the cube wasn't on its mark | `lab eval void sort_one_baseline <model> t05 --reason "cube off its mark"`, then `lab eval push sort_one_baseline` |
| Want to try something without it counting | `--test` (a real session, kept in `results/scratch/`) or `--replay` (no arm movement) |
| `the git tree has uncommitted changes` | commit and push first (official trials record the exact code), or ask Claude what changed |
| `... changed since session ...` | the eval file was edited after trials started: undo the edit, or make a new eval |
| `not served on a100: ...` | the server isn't running or is still loading: check its tmux window |
| `cannot connect to a100` | the A100 isn't reachable from this network: use the H200 or the laptop GPU |
| `rig check failed` | arms or cameras not found: `lab check --rig`, check USB and the phone camera |
| `WARNING: push failed` | the session stays on this machine: `lab eval push sort_one_baseline` when online |

## What to know about latency (2026-11-20)

Measured from the laptop over WiFi to the A100, through the SSH tunnel: about **390-450 ms** turnaround for ACT, of
which the model takes about 40 ms. Ping is 7 ms and the upload speed 42 Mbit/s, so it isn't simply bandwidth; the
cause is still being looked into (lab/PLAN.md). On the laptop's own GPU it is about 36 ms. It's recorded in every
trial (M4, M6), and reports never mix servers. Every model served from the same server gets the same network delay,
but models with short chunks have the least room for it: on the replay robot over this link GR00T (16 actions =
0.5 s) stalled 74% of the time, Diffusion 46%, while ACT (100) and SmolVLA (50) hardly stall. Expect GR00T to pause
often when served from the A100 or H200 from this laptop.
