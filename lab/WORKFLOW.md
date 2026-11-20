# Workflow

- **Part 1: from a new dataset to trained models** (below): record, review, register, train.
- **[Part 2: evaluation](#part-2-evaluation)**: official robot trials, results, reports.

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

**Status (2026-11-20): being built** (lab/PLAN.md, Phase 5 steps 1-8). Each step is added here once it is tested;
until a step is marked ready, ask Claude. How results are stored and why: [EVALUATION.md](EVALUATION.md). What
every number means: [METRICS.md](METRICS.md).

The path, once built:

```
lab/evals/<eval>.yaml  →  lab eval freeze  →  commit + push
model server (laptop GPU / A100 / H200): start the eval's servers
robot computer (laptop, later the Pi): lab eval run <eval> --server <host>   (20 trials per model, rounds)
lab eval report <eval>   (any machine)
```

| Step | Ready |
|---|---|
| Test runs go to `results/scratch/<date>/` (never mixed with official results): `lab run`, `lab results` | ready |
| Writing an eval file, `lab eval freeze`, `lab eval status` | not yet |
| Serving an eval on the laptop GPU, the A100 or the H200 | not yet |
| `lab eval run`: trials, keys `s` / `f`, resume, `lab eval void`, video, push | not yet |
| `lab eval report` | not yet |
| `lab eval offline` (held-out action error, M9) | not yet |

Rules that already hold:

- Only `lab eval run` makes official results. Anything you try by hand is a test run.
- The official copy of every result is the private Hub dataset `orangemoon599/drtc-results`. Nothing analysis needs
  is kept on the GPU servers.
- Old results (before 2026-11-20) are in `results/archive/`.
