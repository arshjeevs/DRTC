# Major project plan

*Working plan for the major phase, from 2026-11-18. It replaces §9–§10 of `lab/PROPOSAL.md` for the major
(the proposal stays as the research framing). Each phase starts only when the previous one is done, so
nothing has to be redone. Items marked **[open]** are decisions still to make.*

## LOCKED SCOPE (2026-11-20, with the user): this is the project from here on

Decided one item at a time; it replaces the evaluation plans further down wherever they differ. Today's earlier robot
evals and benches (sort_one_baseline_v2, sort_one_faults, the bench_* evals) are side results: valid, shown in the
report, not the main experiment.

**Archive (2026-11-20, the user: "start fresh with respect to the DRTC experiments"):** every result made before
this scope (robot evals and benches) moved unchanged to `results/archive/2026-11-20-before-scope/` (laptop and H200;
the results store keeps its copy), their eval files to `lab/evals/archive/`. Side results, usable in the report.

**Claim.** In client-server robot inference, network faults lower the task scores; **DRTC recovers them**; **smoothing**
then improves the motion.

**Datasets** (as they are, no new recording or edits): `so101_cubebox_v0` (50 = 45 + 5 held out),
`so101_cubebox_v1` (110 = 100 + 10), `so101_sort_one_v1` (88 = 80 + 8).

**Models and what each dataset gets** (the 7 shortlisted, each with its fixed original recipe: SmolVLA, GR00T N1.5,
GR00T N1.7, π0.5, MolmoAct 2, ACT, Diffusion):

| Dataset | Trained | Robot evaluation |
|---|---|---|
| `so101_cubebox_v0` (initial, temporary research) | SmolVLA, GR00T N1.5 | baseline (condition 1) only |
| `so101_cubebox_v1` | all 7 | baseline (condition 1) for all 7; the DRTC experiment only if time remains |
| `so101_sort_one_v1` | all 7 | baseline (condition 1) for all 7; **the DRTC experiment (conditions 1-4) for SmolVLA and GR00T N1.5** |

**Trainings** (✓ trained and on the Hub, ▶ running, · to do):

| Model | cubebox v0 | cubebox v1 | sort_one |
|---|---|---|---|
| SmolVLA | ✓ | ▶ H200 | ✓ |
| GR00T N1.5 | ▶ H200 | ✓ | ✓ |
| ACT | (✓, not needed) | ▶ H200 | ✓ |
| Diffusion | – | ▶ H200 | ✓ |
| GR00T N1.7 | – | · | · |
| π0.5 | – | · | · |
| MolmoAct 2 (LoRA, its original recipe) | – | · | · |

The 4 running ones started 2026-11-20 on the H200 (tmux `train` in the drtc_lab container; logs
`logs/train_<model>.log`). Still to do: 6 (GR00T N1.7, π0.5, MolmoAct 2 on cubebox v1 and sort_one); π0.5 takes ~27 h
each.

**Conditions** (per model, per task; the same order and rules everywhere):

| # | Condition | Network | Robot side |
|---|---|---|---|
| 1 | baseline | normal | upstream LeRobot remote inference, as it ships: no DRTC, no filter, and it ends the run after 1 s without actions (starvation grace 1.0 s) |
| 2 | baseline + faults | the fault mix | as 1 |
| 3 | DRTC + faults | the fault mix | DRTC, our version (the 2026-11-20 fixes: cooldown never shorter than the measured latency incl. the slowest of the last 10 replies, cold-start wait, request delay capped at the chunk); **no** Butterworth filter; holds the arm through stalls and carries on |
| 4 | DRTC + smoothing + faults | the fault mix | as 3 + the Butterworth filter + adaptive playback (method A) |

Claim 1 = 2 against 1 (faults hurt) and 3 against 2 (DRTC recovers); claim 2 = 4 against 3 (smoothing).

**Fault mix** (conditions 2-4, every task, times from the trial start): latency spike +1 s at 2 s; every reply dropped
4-5 s; full outage (both directions) 6-7.5 s. Each fault type is measured separately on the bench only.

**Rules.** 10 trials per condition, per model, per task (rounds, as before). A terminal fault after the arm has started
moving counts as a failure (M1 = 0, label `fault`); a fault before the first action is a setup problem and is
repeated. The report shows the fraction of trials ended by a fault. Metrics v2 ([METRICS.md](METRICS.md)).

**Robot computer:** the laptop for every experiment; the Raspberry Pi is set up only after everything is complete. The
presentation and report say the laptop was the robot computer.

**Order of work** (approved; the user decides when each step starts):

| # | What | Who | Status |
|---|---|---|---|
| 1 | This scope into PLAN.md, the rule changes into METRICS.md (v2); earlier results archived | Claude | ✓ done |
| 2 | Build the 4 conditions, the fault mix, the new rules (fault = failure, M1f) and the eval files; test on the replay robot | Claude | ✓ done (A100 check 2026-11-20: upstream gave up under the faults, DRTC got through, smoothing cut jerk 3-4x) |
| 3 | Trainings (table above) | Claude, A100 + H200 | ▶ 4 running; 6 to do |
| 4 | **DRTC experiment on sort_one:** SmolVLA + GR00T N1.5 x 4 conditions x 10 trials = 80 trials (~1 h): eval `sort_one_drtc_main`, served on the A100 | the user | ▶ ready |
| 5 | **Baselines:** condition 1 x 10 trials: cubebox v0 (2 models), cubebox v1 (7), sort_one (7), as the models finish | the user | · after 2, and 3 per model |
| 6 | Bench (each fault type separately, sort_one, SmolVLA + GR00T) and offline M9/M10 for every model | Claude, H200 | · |
| 7 | One report with all results | Claude | · |
| 8 | If time: the DRTC experiment on cubebox v1 | the user | · optional |
| 9 | The Raspberry Pi setup | later | · |

| Phase | What | Done when | Status (2026-11-20) |
|---|---|---|---|
| 1 | Move to LeRobot 0.7, our code as a separate layer on top | The new setup runs everywhere | Done; its robot-side items moved to 3 and 5 |
| 2 | Shortlist the models, each with its original recipe | Recipes fixed and sourced | Done; **recipes fixed for all training** |
| 3 | Faster training, and the setup on every machine | One measured speed setup per active model | **Done** (2026-11-19) |
| 4 | Data collection (the user) | New datasets finalised and on the Hub | Sorting A done (2026-11-19); B, stacking later |
| 5 | Evaluation framework | Official trials on any robot computer and server, stored in one place; metrics v1 in one report | **Designed 2026-11-20** ([EVALUATION.md](EVALUATION.md), [METRICS.md](METRICS.md)); building |
| 6 | Train and choose | Baseline of every trained model; the 2 models for 7-8 chosen | Trained: round 1 (cubebox), round 2 (sorting A; Diffusion running). Baseline eval after 5 |
| 7 | Smoothing: bench, baselines, methods A/B/C | Methods compared with metrics v1 (M7, M8) on the chosen models | After 6 |
| 8 | DRTC experiments: faults, estimators, vs upstream | Fault conditions measured (M1-M6) on the chosen models | After 6, 7 |
| Later | Edge fallback (ACT); the deferred models | — | Only if time is left |

**[open]** Major presentation date. It decides how much of Phases 7-8 fits in.

## Decisions (2026-11-20): evaluation, Phases 5-8
- **Phases 7 and 8 are the main part of the project**; Phase 6 is the baseline and the choice of models for them.
- **Metrics are fixed: metrics v1** ([METRICS.md](METRICS.md)). They are the ones the final presentation shows;
  changes only as a new version, with every trial recomputed from its raw files.
- **Storage** ([EVALUATION.md](EVALUATION.md)): official trials only through `lab eval run`, into
  `results/evals/<eval>/sessions/<date>-<machine>/`; everything else into `results/scratch/`. The one official copy
  is the private Hub dataset `orangemoon599/drtc-results` (pull before a session, push after). Model servers store
  nothing for analysis (their timings come back in every reply). The old mixed `results/` moved to
  `results/archive/`.
- **Robot computer:** the laptop (usual) or the Raspberry Pi (edge proof, its own eval, set up after everything works
  on the laptop). **Model server:** the laptop GPU, the A100 or the H200, chosen per session by availability; every
  trial records both, and reports never average across them.
- **Trials:** 20 per model per condition, for every task. The user sets the layouts (one per round). Per trial: the
  operator's key (`s` success / `f` failure, which also gives the time to success), and video. No outcome levels,
  failure codes or blinding.
- **Choice for 7-8 (A):** the top 2 by success rate on sorting A, ideally one small model (ACT or Diffusion) and one
  VLA.
- **Phase 8 robot time (B):** all fault conditions on the replay robot (M2-M8); 4 headline conditions on the real
  robot for success (spikes, drops, disconnect, mixture).
- **[open]** The presentation date (decides Phase 8 step 4).

## Decisions (2026-11-19)
- **Recipes are fixed** for every training from now on: the confirmed originals in `lab/recipes/` (Phase 2
  "Recipes"). Speed settings (frame caches, workers, compile) are not recipe changes.
- **Active models: SmolVLA, ACT, GR00T N1.5.** Deferred: π0.5, GR00T N1.7, MolmoAct 2, Diffusion (recipes, plugins,
  caches kept; MolmoAct 2's flow-step decision waits with it).
- **Machines:** the A100 (192.168.5.13; cleaned 2026-11-19; Claude has SSH access) trains now and serves for the
  robot trials; the H200 tests everything during data collection, afterwards it trains the active models; the
  laptop records and runs the robot.
- **A100 and Hugging Face:** a first test showed ~2 KB/s, a second one 23.5 MB/s for a large file and 0.4-0.6 s for API
  calls: the Hub works from the A100, faster than copying from the laptop (1.8 MB/s over the LAN); only the SmolVLA
  base was copied, the rest downloads there.
- **New datasets (Phase 4):** a new task, the same scene positions and rig as v1 (`so101_v2`: camera1 wrist,
  camera2 scene, 800x600 at 30 fps), names `so101_cubebox_v2`, `v3`, ...; size (episodes, lengths, held-out)
  decided per task. Joints in **degrees** (LeRobot 0.7's default): old and new datasets are not merged, and units are
  set per model at run time (`lab run` sets `robot.use_degrees` from the model's dataset; the server declares the
  units and refuses a mismatched robot), so the old -100..100 models keep working. **The arm is not recalibrated**
  (both unit systems come from the same calibration files).

- **Training logs (decided 2026-11-19): wandb and Trackio for every run on every machine.** wandb (project
  `drtc-lab`) is the shared online view; Trackio keeps a local database per machine (its shared Space needs a
  Hugging Face PRO subscription). `lab train` mirrors LeRobot's Trackio logger to wandb (`drtc.compat.wandb_mirror`)
  and stops when wandb or its key is missing (`--no-wandb` only on purpose); each host's tokens come from its
  `secrets` file in `lab/hosts.yaml`; `lab check --logging` checks a machine.

## Plan by phase (2026-11-19)
**Phase 3 (with 4):**
- [x] A100 setup (`lab/bin/a100-setup`, 2026-11-19): torch 2.14.1+cu126, FFmpeg from conda-forge, uv's Python.
- [x] A100 round 1 trainings started (2026-11-19 17:50, wandb project `drtc-lab`), all three at once:
  - First start: the 24 CPUs were oversubscribed (load 28: each job's PyTorch uses all threads, plus 8 data
    workers each); GR00T N1.5 spent 3.8 s of a 4.9 s step in its CPU preprocessing. Restarted with a CPU budget:
    ACT 2 threads / 2 workers, SmolVLA 4 / 4, GR00T N1.5 12 / 2 (`OMP_NUM_THREADS`, `--num_workers`).
  - Then: GR00T N1.5 2.3 s per step (preprocessing 1.14 s), ~6.3 h; ACT 6.5 steps/s, ~4.1 h; load ~6.
  - SmolVLA (compile, max-autotune) retunes its kernels once at the first epoch's smaller last batch (step ~170
    on v0); a one-time cost.
- [x] Per-machine CPU budget (hosts.yaml `cpu_budget`, threads and workers per policy): `lab train` applies it when
  another `lab train` already runs on that machine; a job alone gets the whole machine. A100 values as measured.
- [x] GR00T N1.5 on the A100 restarted with the cache fix (2026-11-19 18:20; the first run decoded full-size videos):
  preprocessing 1.14 -> 0.23 s, step 2.65 -> 1.79 s (~4.9 h); loss at step 200 0.322 in both runs (same input).
- [x] Old models on LeRobot 0.7 (A100, 2026-11-19, measured with three trainings on the GPU, so the times are high):
  both admitted in chunk and rtc_guided mode. GR00T 379 ms chunk / 466 ms RTC, SmolVLA 571 / 928 ms.
  [x] On a free H200: GR00T 109 / 137 ms (4 / 5 steps, margin 8: fits); SmolVLA 272 / 445 ms (9 / 14, margin 30:
  fits). Both are usable references for the robot trials.
- [x] H200: FFmpeg, the frame-cache checks, the full comparison round, one speed setup per active model (Phase 3
  section, 2026-11-19).
- [ ] Later, while models train (not now): **easy serving on either GPU server.** Today serving on the A100 or the
  H200 means logging in, starting the container/environment and `lab serve <model>` by hand, before the robot
  client connects. Idea to work out: the client asks for a model and the chosen server starts it on demand (a small
  always-on launcher per server, or the laptop starting it over SSH), so `lab run` works the same against either
  machine.

**Phase 4 (the user, with 3):**
- [x] Robot-side checks (rig, calibration files present; no recalibration) and a 2-episode `lab record` test
  (2026-11-19: 29.9 Hz, no late ticks, degrees, AV1; fixes: Rerun viewer on PATH, `dataset.no_stamp`).
- [ ] Recording sessions; each finished dataset is trimmed, pushed, registered and prepared automatically.
- **Tasks (decided with the user, 2026-11-19).** Printed 7x7 cm colour squares as targets; one dataset per task, in
  this order. The squares, cubes and cube spots sit on taped marks, so every layout is exactly repeatable (and the
  robot trials use the same marks). Each layout combination is recorded equally often, in a fixed rotation, so no
  shortcut ("always go right") fits the data. Fixed order of moves within an episode (no mixed strategies).

  | Dataset | Setup | Instruction | Training + held out | Episode / reset |
  |---|---|---|---|---|
  | `so101_sort_one_v1` (A) | One cube (red or blue) on one fixed spot; the red and blue squares swap between two fixed spots | "Put the cube in the square of the same color." | 80 + 8 (done) | 30 / 10 s |
  | `so101_sort_two_v1` (B) | Both cubes on two fixed spots, swapping; the squares swap as in A; red first, then blue | "Put the red cube in the red square and the blue cube in the blue square." | 100 + 10 | 45 / 15 s |
  | `so101_stack_v1` | Red cube onto the blue cube (layout decided at its session) | "Put the red cube on top of the blue cube." | 100 + 10 | 30 / 10 s |

  - A's rotation (x20): red cube / blue square left; blue cube / blue left; red / blue right; blue / blue right.
  - B's rotation (x25): red cube left / blue square left; red left / blue right; red right / blue left; red right /
    blue right.
  - Later: C (both cubes, the instruction names one; language-conditioned, so not for ACT; needs per-episode tasks
    in `lab record`); A/B with varied cube positions; stacking with three cubes.
  - **Dataset operations (2026-11-19): `lab data show|view|delete|trim|undo|push|remove <name>`.** The raw
    recording is the only copy that is edited; the trimmed copy is rebuilt from it after every change (autotrim plus
    manual trims, kept in the raw copy's `meta/drtc.json`). Edits are local until `lab data push`; one undo level
    (hardlink snapshot) until the next push or recording. Pushing does not prepare (each training host prepares at
    its first `lab train`). Per-episode layout tracking was built and removed again at the user's request; the user
    keeps the rotation by hand. `lab record` speaks its prompts by default (`--no-sounds` to mute).
  - First session: `lab record so101_sort_one_v1 --task "Put the cube in the square of the same color." --episodes 80 --held-out 8`
    (`--resume` adds episodes in a later session).

**Phase 5:** designed 2026-11-20; see "Phase 5: Evaluation framework" below.

**Recording standard (2026-11-19, the user): held-out episodes come on top.** `lab record <name> --episodes N
--held-out K` records N + K episodes; N train, K are held out (seeded split, never trained on). `--resume` continues
up to N + K. `so101_sort_one_v1` was recorded before this (80 = 72 + 8); it becomes 80 + 8 with 8 more episodes
(`--episodes 80 --held-out 8 --resume`).

**Phase 6 round 2, sorting A (2026-11-19):** `so101_sort_one_v1` finished (88 episodes: 80 training + 8 held out;
trimmed 20,935 frames). The user chose ACT, SmolVLA and GR00T N1.5 with the fixed recipes (`lab/models/sort_one_v1.yaml`), trained
on the H200 at once (started by the user 2026-11-19) (`lab prepare` first, so the three runs do not prepare the same copy together; H200 `cpu_budget`
scaled from the A100's, not yet measured). Deferred models stay for later.

**Phase 6:** round 1 (A100): SmolVLA and ACT on trimmed v0, GR00T N1.5 on trimmed v1, then offline ranking and
robot trials against the kept minor models (GR00T 80%, SmolVLA 90%). Round 2 (after Phase 4): the user's training
order on the new datasets (A100, H200; laptop for ACT), each run evaluated with Phase 5.

---

## Status at the end of the 2026-11-19 session (handover)
**Done:** Phases 1-3. Phase 3's outcome: one speed setup per active model (recipes' `speed:` and `data:`; hosts'
`cpu_budget`): SmolVLA compile (H200 1.8 -> 1.1 h), GR00T N1.5 frame cache (H200 3.0 -> 1.4 h), ACT shares a GPU.
Logging standard: wandb (`drtc-lab`) + Trackio local. Video: AV1 recordings, torchcodec decoding everywhere.

**Round 1 crash (2026-11-19 18:30):** checkpoint pruning crashed `lab train` at each run's 2nd checkpoint (a
relative vs resolved path in a log line; fixed in a02cc21f, with a test). The checkpoints were saved first: ACT was
resumed from 10000, SmolVLA and GR00T N1.5 from 2000 (by `logs/resume_after_crash.sh` in A100 tmux window `watch`).
Held-out losses now persist in `outputs/train/<model>/eval_losses.json` (seeded from the logs for the resumes), and
`--resume` keeps the CPU threads.

**Running (Phase 6 round 1, A100, started 2026-11-19 ~18:00):** `act_cubebox_v0_full` (~3 h), `smolvla_cubebox_v0_full`
(~5 h), `groot_cubebox_v1_full` (~4.9 h, restarted 18:20 with the cache fix), with the fixed recipes. A100 `tmux`
session `drtc`, windows act / smolvla / groot; logs `~/drtc_major/logs/train_<model>.log`; wandb project `drtc-lab`.
Each pushes to `orangemoon599/<model>` at the end (best checkpoint by held-out loss at the repo root). Claude
reaches the A100 over SSH (`b230968ee@192.168.5.13`, key login); the user runs everything on the H200.

**Ready:** Phase 4 recording (`lab record so101_cubebox_v2 --task "..." --episodes N`; tested 2026-11-19). The user
records; task instruction and size per dataset are the user's.

**Next session, in order:**
1. Check round 1 finished and pushed; note the held-out losses.
2. Phase 5, the evaluation framework: offline first (held-out action error per checkpoint, one table), so round 1
   can be ranked; then the robot protocol (ask the user for N trials per model and the cube-position grid), DRTC
   metrics per trial, results report; Zenoh between the laptop and the A100/H200 (Phase 1 D7).
3. Round 1 robot trials against the kept minor models (both fit DRTC's margin on a free H200).

**Small open items:** H200 `lab check --logging` (after `git pull`, `uv sync`, `LAB_HOST=h200`, PATH with
`.venv/bin` in its `.server.env`); Diffusion's 0.162 vs 0.103 s (deferred); later: easy serving on either GPU server,
the deferred models, the presentation date.

## Phase 1: LeRobot upgrade (option c: LeRobot as a dependency, our code on top)

### Where we are
- The repo is a fork of `jackvial/drtc`, built on LeRobot from 2025-12-31 (version 0.4.3).
- Upstream is at **v0.7.0** (tagged 2026-11-05). `main` reads 0.7.1 and is 788 commits ahead of our base.
- Our changes since that base:
  - 267 commits
  - about 100 files outside `lab/`
  - about 19k lines, mostly the DRTC client/server and our policy ports

### Target layout
LeRobot is no longer copied into the repo; it is installed as a pinned package. A future upgrade means
bumping the pin and fixing what broke in our layer.

```
pyproject.toml              lerobot[...] == 0.7.0 (pinned), plus our packages
src/drtc/                   our code (one Python package; the `lab` command is drtc.cli)
  control/                  remote control on upstream remote_inference with Jack Vial's DRTC ideas
                            (freshest-wins merging, latency estimation, trigger cooldown, RTC), fault
                            injection, smoothing A/B/C
  registry/                 loads and checks lab/ configs, expands recipes and sweeps, lineage
  jobs/                     job files, the per-host worker, submission over SSH
  logging/                  wandb runs and artifacts, Hub mirror of results
  compat/                   loader for old LoRA adapters, importers for the old models.yaml / rig.yaml / index.csv
src/lerobot_policy_dot/     plugins, found automatically by LeRobot (lerobot_policy_* prefix)
src/lerobot_policy_sfp/
src/lerobot_policy_groot_n15/   GR00T N1.5 kept as a plugin (decision 1b)
lab/                        configs and docs (the registry, see "Lab system" below); bin/ keeps the old
                            command names as aliases for `lab ...`
tests/
```

Git and folders:
- The end-of-minor state is tag `minor-final`: the `drtc` branch plus the uncommitted rig edits. It is the code
  the minor results and checkpoints came from.
- Untracked files and the exact venv package lists are in `~/Projects/drtc-snapshots/minor-final-2026-11-18/`.
- New work is on branch `major`, in a separate worktree `~/Projects/drtc-major` with its own venv.
  `~/Projects/drtc` (branch `drtc`) and its venvs stay as they are and keep working for the robot until Phase 1
  passes its checks. Then `major` is merged into `drtc`.
- `lab/` history is kept.

### Lab system (accepted 2026-11-18)
Replaces the old `lab/bin` + `lab/tools` + single `models.yaml`/`rig.yaml`. It is built in Phase 1 in place of
porting the old tools.

```
  laptop (control + robot side)                      compute hosts (H200 now; A100 / more later)
  ┌───────────────────────────────┐                 ┌──────────────────────────────────────┐
  │ lab CLI: record, submit,      │   ssh           │ lab worker  (user starts it, tmux)   │
  │          run, status          │ ──────────────▶ │  ├─ job queue (local file)           │
  │ registry (git): rigs, hosts,  │                 │  ├─ checks out the job's git commit  │
  │   datasets, recipes, models,  │                 │  ├─ runs train / eval / bench / serve│
  │   protocols                   │                 │  └─ waits for free GPU memory        │
  │ Zenoh router  ◀───────────────┼── policy server ┤                                      │
  └──────────────┬────────────────┘                 └──────────────────────────────────────┘
                 │ Zenoh                 all hosts: datasets + checkpoints via HF Hub,
        Pi / laptop robot client         logs + lineage to wandb, results mirrored to a Hub repo
```

| Part | Design |
|---|---|
| Registry | YAML files per type in `lab/`: `rigs/`, `hosts.yaml`, `datasets/`, `recipes/`, `models/`, `protocols/`, `experiments/`. A checker validates references and names. Lineage (rig → dataset → derived dataset → model → evaluation) is recorded in wandb artifacts by every run |
| Recipes | One file per policy recipe, with its source (who made it work, on what robot). A model = dataset × recipe × base checkpoint + overrides. A sweep entry expands to N models. Our own expander on draccus, not Hydra |
| Rig | Versioned rig profiles. Every dataset records its rig version and calibration fingerprint; a run refuses a model trained for other cameras |
| Hosts and compute | `hosts.yaml`: roles (record, client, serve, train), GPUs, SSH/container, cache paths. Adding compute = one entry + `lab worker` there |
| Jobs | A job file holds the kind (train, eval, bench, serve), model, GPU memory and count needed, git commit and env lock. The worker checks out that commit, runs jobs when enough GPU memory is free (several side by side), multi-GPU via upstream `--parallelism.*`. After training: best checkpoint by validation, push, offline eval, benchmark. Other backends (SLURM, cloud) can replace the worker later with the same job files |
| H200 rule | The user starts workers and submits jobs (`lab submit <job> --host h200`). Claude writes job files and never submits them |
| Serving | Policy servers on any host connect to a Zenoh router on the laptop; the client asks for a deployment by name. Replaces `lab/bin/tunnel`; several servers at once |
| Collection | Leader-arm recording; every robot trial also saved as a dataset. After recording, automatically: checks (dropped frames, camera rate, joint limits), autotrim, stats, per-episode quality report, Hub push with a dataset card. DAgger after Phase 4 |
| Testing | Tiers: code tests → offline evaluation → replay bench → robot protocol (fixed start positions in random order, alternating conditions, set trial count, success and failure reason per trial) |
| Logging | wandb for everything (training, offline eval, robot trials, per-tick logs), mirrored to a private Hub dataset repo |
| CLI | One `lab` command, installed as a package, the same on laptop, Pi, H200 and Windows. `lab status`: jobs, latest models, what each still needs |
| Compatibility | Old `lab/bin/*` names stay as aliases. An import tool converts `models.yaml`, `rig.yaml`, `results/index.csv` |

### What happens to each of our changes

| Our change | Upstream 0.7 | Plan |
|---|---|---|
| DRTC client/server, LWW registers, latency estimation, RTC guidance, fault injection, metrics, simulation (`async_inference/`, ~8k lines) | `async_inference` removed; new Zenoh-based `remote_inference` with chunk alignment, blending, RTC, refill timing, position hold | Rebuild on upstream `remote_inference` with DRTC's ideas (decision 1a). The original DRTC stays runnable from `minor-final` for comparison |
| gRPC proto changes (`transport/`) | `transport/services.proto` still exists | Not needed: upstream's transport is used (Zenoh) |
| Training: held-out validation, LoRA, clean stop, progress line, uint8 frames in shared memory | Has held-out eval split, PEFT, accelerate (bf16, FSDP2/DDP), GPU-compiled image transforms | Use upstream for validation, LoRA, precision and parallelism. Keep only clean stop, progress line and `val_action_mse`, if still needed, in a thin wrapper around `lerobot-train`, not a copy of it |
| Our own LoRA (`policies/lora.py`) | PEFT built in | New training uses upstream PEFT (`--peft.*`). Keep our loader in `src/drtc/` so old adapters (e.g. the working GR00T LoRA) still load |
| π0-FAST port | In-tree | Drop ours |
| π0/π0.5 `embed_tokens` tie fix | Fixed (copies `lm_head` into `embed_tokens` when loading) | Drop |
| DOT, SFP | Not in LeRobot | Plugins `lerobot_policy_dot`, `lerobot_policy_sfp` |
| GR00T N1.5 (with our RTC for method B) | Only N1.7 (Qwen3-VL, 40-action chunks, RTC) | Plugin `lerobot_policy_groot_n15`, ported from our code (decision 1b) |
| ACT, Diffusion small edits (validation targets, DRTC server queue, frame resize) | Diffusion is not supported by upstream remote inference (needs observation history) | Drop; redo only if ACT/Diffusion are shortlisted |
| Jack's camera, record and teleop speed patches | Camera has a background reading thread with `async_read` and `read_latest` | Drop; measure the camera rate in 1.8, and patch only if it is too slow |
| Dataset batched video encoding (Jack) | New `dataset_writer` with batched encoding; dataset format still v3.0 | Drop; check that v1/v1_trim load and that recording works in 1.8 |

### Steps
- [x] 1.1 Snapshot: tag `minor-final`, archive and venv package lists in `~/Projects/drtc-snapshots/` (2026-11-18). To do: push the tag (`git push origin drtc minor-final`)
- [x] 1.2 Branch `major` in worktree `~/Projects/drtc-major` (created 2026-11-18); `pyproject.toml` pinned to lerobot 0.7.0. Check how many venvs we need now that every policy uses transformers 5.22 (today: separate venvs per model family)
  - Done 2026-11-18. The vendored fork was removed from `major`. `pyproject.toml` (package `drtc-lab`) pins
    `lerobot==0.7.0`; `uv.lock` is committed.
  - Base install is the robot side (Pi); extra `gpu` is training and serving.
  - One venv covers SmolVLA, π0/π0.5/π0-FAST, X-VLA and GR00T N1.7 (transformers 5.22.2, torch 2.14.1+cu130). It
    works on the 4060. Still open: whether GR00T N1.5's Eagle code runs on transformers 5 (step 1.5).
  - **The H200 cannot be changed** (no driver or system changes, not even cleaning caches). Its driver 580
    supports CUDA up to 13.0, so torch is pinned to **2.14.1+cu130** (the newest CUDA 13.0 build) for Linux x86_64.
    It is tested on the laptop (GPU, bf16, video decoding, all tests). The new setup uses its own dataset cache
    folder on the servers, so old caches there are never read.
- [x] 1.3 Finish the change table above (2026-11-18)
- [ ] 1.4 Build `src/drtc/` on upstream `remote_inference` (DRTC's scheduling ideas, our fault injection and logging) and run it against a test policy on the laptop
  - [x] D1: replay robot plugin `lerobot_robot_replay` (`--robot.type=replay`). It plays a recorded episode
    (real frames, 18 ms per observation); its state follows commands, it logs them, and it declares position hold.
  - [x] D2: `lab serve <model>`: the deployment comes from the registry. Joint semantics follow the dataset's units
    (`so101-range100-…` for v0/v1, `so101-degrees-…` for new data); camera semantics follow the rig's roles.
    Upstream's server and client ran end to end on the laptop: 29.8 Hz, 0 starved ticks.
  - [x] D3: `lab run <experiment> [--model] [--robot rig|replay]` plus `lab results`.
    - Per run: `results/<exp>/<model>/<run_id>/` with config.yaml, ticks.csv (per control tick), faults.csv,
      run.json (outcome and fault reason), rollout.log and the robot's commands; plus `results/index.csv`.
    - Failed runs are recorded too.
  - [x] D4 (option a): `--inference.type=drtc`. It extends upstream's remote engine and changes only the request
    threshold (DRTC: queue ≤ H − s_min) and the latency estimate (jk / max_last_10 / fixed / upstream).
    Replay test: the DRTC scheduler requested 32 chunks (queue ≥ 79), upstream 8 (queue down to 14); 0 stalls each.
  - [x] D5: fault injection at the transport (observations published, replies received), for all 10 replication
    conditions. **Replay results with the test ACT** (3.3 s chunks, ~45 ms turnaround, DRTC trigger):

    | Condition | Outcome on upstream's runtime |
    |---|---|
    | baseline, dup_action, spikes_jk, spikes_max_last_10 | 25 s, 0 stalls (duplicate replies rejected; 0.5 s and 1 s spikes absorbed) |
    | reorder_action | 25 s, 5 short stalls |
    | drop_obs, drop_action, disconnect, reorder_obs, mixture | **terminal fault after 6–10 s**: "action starvation grace expired" |
    | dup_obs | **terminal fault after 3 s**: the server rejects the second copy as "Duplicate or completed request", and the client treats it as a protocol error |

    **Why the lost-message runs end:** with one request in flight, a lost message blocks requests until its 5 s
    deadline. The queue runs out, and upstream's starvation recovery sends one fresh request with 1 s of grace,
    which falls in the next fault window. **This is the measured case for option (b)** (DRTC's cooldown
    re-request).
  - [x] D4b (decided 2026-11-18, built): the cooldown re-request. A request unanswered for its cooldown (latency
    estimate in steps, capped at H/2, + ε; s_min + ε before the first measurement) is given up, and a fresh
    observation is sent. Upstream's server rules are kept:
    - a `BUSY` reply backs off one action interval;
    - a `STALE` "duplicate" reply for the request in flight is ignored;
    - the aligned-mode anchor is cleared after a give-up.

    Every recovery is logged (`recovery.csv`). **Validated on the quiet laptop (2026-11-18): all 11
    replication conditions complete their full 25 s** (replay robot, test ACT; at most one 0.03 s stall). Without
    (b), 6 of them ended in a terminal fault.

    | Condition | Upstream runtime alone | With (b) |
    |---|---|---|
    | drop_obs / drop_action / disconnect | terminal after ~6 s | 25 s, 1 stall of 0.03 s, 64–70 re-sends |
    | reorder_obs | terminal after ~10 s | 25 s, 0 stalls (12 re-sends, 1 BUSY back-off) |
    | dup_obs | terminal after 3 s (protocol error) | 25 s, 0 stalls (9 duplicates ignored) |
    | mixture | terminal after ~10 s | 25 s, 0 stalls (37 re-sends, 2 duplicates ignored) |
    | baseline, dup_action, reorder_action, spikes ×2 | 25 s | 25 s, 0 stalls |
  - [x] D6 (decided 2026-11-18: only this for now): the Butterworth action filter, ported unchanged (order 2,
    3 Hz, gain 1.4; identical to the old filter to 1e-12, `tests/test_action_filter.py`). It is applied to every
    action before the robot and reset when motion is invalidated. Smoothing A/B/C: decided later.
- [x] Serving the kept models (`drtc.compat.serve`, used by `lab serve`, 2026-11-18). An old-format checkpoint is
  converted to a local copy (upgraded config, mapped processor step names). Upstream's server builds the policy with
  our old-LoRA loader. `smolvla_cubebox_v0_lora_raw` was served and run on the replay robot: 15 s, 0 stalls,
  ~280 ms turnaround on the 4060. GR00T N1.5 uses the same path; its serving test is on the H200 (D7).
- [ ] 1.5 Policy plugins (DOT, SFP, GR00T N1.5)
  - [x] GR00T N1.5 plugin `lerobot_policy_groot_n15` (type `groot_n15`, steps `groot_n15_*`), ported to
    transformers 5. Changes:
    - kw-only config dataclass
    - `post_init`
    - lazy Beta distribution (meta-device init)
    - image-processor imports moved
    - its own Eagle cache folder, so the old setup's stays untouched
  - [x] Loading the kept models (`drtc.compat`): our old LoRA adapter format, config upgrade, N1.5 processor step
    names mapped
  - [x] **Old-vs-new parity of both kept models (2026-11-18)** (`tests/parity`; results in
    `~/Projects/drtc-snapshots/parity/`):
    - **Method:** same inputs and the same recorded random draws through the old setup (`minor-final` venvs)
      and the new one.
    - **Preprocessing:** bit-identical for both, including GR00T's Eagle pixel values and tokens.
    - **Actions:** they differ only within the old setup's own GPU-to-GPU noise. That noise was measured as the
      same old code on the laptop vs the H200 (`compare_refs.py`): GR00T 0.31°, SmolVLA 2.84° (largest at the
      rest pose; 50-step chunks amplify rounding). Old vs new:
      - GR00T: 0.29° on the H200
      - SmolVLA: 1.86° on the H200, 1.06° on the laptop
    - **Laptop GR00T N1.5 inference:** 613 ms median (636 mean, 765 p90), 7.27 GB peak. Too slow to serve the
      robot from the laptop (a 16-action chunk is 533 ms), so it is served from the H200. This is the baseline for
      the efficiency work.
  - [ ] DOT, SFP plugins (only if shortlisted in Phase 2)
- [ ] 1.6 Training wrapper (clean stop, progress line), recipes → 0.7 `lerobot-train` flags
  - [x] `lab train <model>` (`drtc.train`, 2026-11-18). It builds the 0.7 `lerobot-train` command from the
    registry:
    - the dataset pinned to its revision; a stale local copy is refused;
    - our seeded held-out split, mapped onto 0.7's eval split (training episodes first, held-out last);
    - epochs → steps, and ~20 held-out evals per run;
    - recipe flags checked against the policy config;
    - camera roles → `rename_map`.

    It adds a clean stop (Ctrl-C / SIGTERM / `lab stop`: checkpoint at the next step, pushed with training
    state, `--resume` anywhere) and pushes the final or a chosen checkpoint (`--push <step>`). 0.7's own
    progress bar shows the time left.
  - [x] Tested on the laptop with ACT: 200 steps with held-out eval; stop at 55, resume to 200 (the same losses
    as an uninterrupted run).
  - [x] `lab submit` / `lab worker` / `lab jobs` (`drtc.jobs`). A job pins the pushed commit, `uv.lock` and its
    GPU memory need; the worker starts it only then, one at a time by default. Tested locally.
  - [ ] Dry runs: the commands build for all 9 trainable planned models. DOT and SFP need plugins (only if
    shortlisted).
  - [ ] **[open]** Logging: 0.7 has no wandb, only Trackio. Choose a Trackio Space, wandb through our wrapper,
    or both. Until then runs log to local Trackio, project `drtc-lab`.
  - [x] **All 8 planned recipes train on LeRobot 0.7 on the H200** (2026-11-18, `tests/smoke/train.sh`: 30
    steps + one held-out eval each, no checkpoints):

    | Model | Held-out loss | Peak GPU | Time |
    |---|---|---|---|
    | ACT | 0.81 | 44.6 GB | 195 s |
    | Diffusion | 1.01 | 15.8 GB | 175 s |
    | SmolVLA | 0.46 | 14.7 GB | 162 s |
    | X-VLA | 0.94 | 32.0 GB | 170 s |
    | π0-FAST | 7.38 (token loss) | 37.6 GB | 390 s |
    | π0 | 0.24 | 43.8 GB | 466 s |
    | π0.5 | 0.26 | 45.5 GB | 344 s |
    | GR00T N1.5 (plugin) | 0.22 | 20.5 GB | 250 s |

    - The old base-revision pins of π0/π0.5/π0-FAST work on 0.7.
    - The worker's GPU estimates are these peaks plus ~10%.
    - The GR00T plugin had to become its own package (`plugins/lerobot_policy_groot_n15`), because LeRobot finds
      plugins by distribution name.
    - **For Phase 3:** in the H200 container, torchcodec cannot decode the videos and LeRobot falls back to PyAV,
      so data loading is slower.
  - [ ] After training (later): pick the best checkpoint by held-out loss, then offline eval and benchmark.
- [ ] 1.7 Lab system (see above), in this order:
  - [x] registry + checker + import of the old `models.yaml` / `rig.yaml` (2026-11-18)
    - `src/drtc/registry/`, files in `lab/rigs`, `lab/hosts.yaml`, `lab/datasets`, `lab/recipes`, `lab/models`
    - The import produced 2 rigs, 4 hosts, 8 datasets, 34 recipes and 46 models
    - Parity tests (`tests/test_registry_parity.py`, 82 tests):
      - every old `models.yaml` entry comes out identical
      - every experiment's composed run config comes out identical (Linux and Windows rigs)
      - frozen old files are in `tests/fixtures/legacy/`
    - Recipes take camera roles (wrist/scene), so `rename_map` follows the dataset's rig
    - `lab list` and `lab check [--rig] [--data]` work: registry, devices, calibration (warns about LeRobot's
      default folder), episode counts
  - [x] clean start: renames, pinned datasets, registry reset, `lab check --data --online` (2026-11-18)
  - [x] `lab trim` (`drtc.data.trim`, 2026-11-18): the minor's autotrim detection, unchanged, with the crops
    applied on LeRobot 0.7 into a new folder.
    - **Method:** no re-encoding (each episode's video window moves), mp4 files hardlinked, numeric stats
      recomputed with LeRobot's functions.
    - **Parity:** on `so101_cubebox_v1_raw` (`tests/test_trim_parity.py`) it gives the same noise floor, the same
      110 crops, the same data rows and episode windows as the old setup. The per-episode stats are exact; the old
      editor's std was float32-imprecise, up to ~10% on the wrist roll.
    - **Generic:** on `so101_cubebox_v0_raw` it measures its own noise (0.0058) and cuts 19%.
    - **Dataset quantiles (decided 2026-11-18): LeRobot's current convention** (the installed LeRobot's
      `aggregate_stats`, an envelope: q01 −50.6 on joint 1 of v1). LeRobot 0.4.3 used the frame-weighted mean of
      the episodes' quantiles (−22.9, which the current v1 keeps); the true value is −46.5. `--quantiles
      weighted_mean` only reproduces old datasets.
  - [x] `lab record` (`drtc.data.record`): records `<name>_raw` with 0.7's `lerobot-record` (its terminal key
    listener works on Wayland/SSH), checks it, trims into `<name>`, pushes both and writes pinned registry entries.
    0.7's RecordConfig parses the generated command, with the right calibration folders and joint units. Not yet
    run with the robot.
  - **Joint units (decided 2026-11-18): LeRobot's current convention, degrees, for new recordings.** LeRobot
    0.7's SO101 default is `use_degrees: true`.
    - Each dataset entry states its own units: v1, v1_raw and v0_raw are −100..100.
    - A model runs the robot in its dataset's units (control system, D).
    - The user accepts retraining for new data; v2+ in degrees is not mixable with v1.
  - [x] `lab` CLI with the old names as aliases (2026-11-18).
    - **Aliases:** list, train, stop, run, record, preflight → `lab check --rig`, analyze → `lab results`,
      calibrate → `lab calibrate` (explicit calibration file, backup), server → `lab serve <model>`.
    - **Kept as is:** phonecam and session (they read the rig through `lab rig <key>`).
    - **Not ported (they say so):** bench, eval-offline, logstats, ablate-cameras, tunnel, edit-dataset, venv,
      h200-setup.
    - **Removed from `major`:** the old `lab/tools`, `models.yaml` and `rig.yaml`. They stay in `minor-final` and
      the test fixtures.
    - Hosts can declare their `hostname`, so `lab` knows where it runs (the laptop is `arshjeevs`).
  - [x] `lab import-results`: the minor's 9 robot runs plus its offline/benchmark folders were copied to
    `results/minor/` under the new model names.
  - [ ] jobs + worker + `lab submit` (train, eval, bench, serve)
  - [ ] logging to the Trackio Space for robot runs and a Hub mirror of results (training already logs to Trackio); the results import is done
  - [ ] Zenoh serving across hosts (replaces `tunnel`), Pi and H200 setup
    - 2026-11-18, for the parity check, done by Claude with the user's permission (no deletions):
      - `~/drtc_major` was cloned on the H200, with `.server.env` and `.venv` from `uv.lock` (6.8 GB, `uv` 0.10.4
        in the container).
      - torch 2.14.1+cu130 runs on the H200 NVL with driver 580.
      - The Hugging Face model cache is shared with `drtc_lab` (files only added).
      - `/home` is 97% full: 94 GB free afterwards.
      - Once the migration is done, the user will delete the whole DRTC space there and set up fresh.
    - H200 (cannot be changed; see `lab/H200.md` for the existing setup): the new setup lives in `~/drtc_major`
      (branch `major`), next to the old `~/drtc_lab`, inside the same `drtc_lab` container (the guide's image, which
      already mounts all of `/home/jayaraj`). It has its own `.venv` from `uv.lock`.
      - Shared with the old setup: the Hugging Face model cache (`drtc_lab/.cache/huggingface`, so no second copy
        of large checkpoints; `/home` is ~88% full) and `drtc_lab/.secrets.env`.
      - Its own dataset folder (`HF_LEROBOT_HOME=drtc_major/.cache/lerobot`), so the stale untrimmed v1 cached by
        the old setup is never read.
      - Nothing in `~/drtc_lab` is modified.
- [ ] 1.8 Checks:
  - `so101_cubebox_v1_trim` loads
  - a short training per kept policy on the laptop and the H200
  - a recorded test episode
  - the robot runs through DRTC (laptop, then Pi)
  - everything in the compatibility list below

### Clean start (2026-11-18)
The new setup starts clean instead of carrying the whole old one. Hub and local renames were run by Claude with
the user's approval; the log is in `~/Projects/drtc-snapshots/minor-final-2026-11-18/rename_hub.log`, and the
name table is in `lab/renames.yaml`.

**Datasets.** The trimmed copy is now v1. Untrimmed recordings keep a `_raw` name, so a name without `_raw` is
always trimmed.

| Old | New | |
|---|---|---|
| `so101_cubebox_v1_trim` | `so101_cubebox_v1` | 110 episodes, 25,067 frames |
| `so101_cubebox_v1` | `so101_cubebox_v1_raw` | untrimmed, 29,525 frames |
| `train_new_1` | `so101_cubebox_v0_raw` | earlier two-scene-camera setup, 50 episodes |
| `so101_cubebox_v1_scene` | deleted | regenerate from v1 when needed |

**Models.** Only the two that worked are kept: `groot_cubebox_v1_lora_raw` (was `groot_cubebox_v1_lora`) and
`smolvla_cubebox_v0_lora_raw` (was `smolvla_train_new_1_lora`). Jack's `jack_smolvla_cube` stays as an external
model for the replication experiments.
- Old models on raw v1 were renamed `<name>_raw` on the Hub, which frees the v1 names. They are not in the
  registry.
- The three `*_cubebox_v1_full_trim` models are untouched on the Hub and not in the registry.
- New models are `<policy>_cubebox_v1_full` (planned).

**Pins.** Every dataset entry pins its Hub revision (the commit of the v3.0 tag), episode count and frame count.
`lab check --data [--online]` and every job refuse a copy that doesn't match. That catches stale caches under
reused names, such as the untrimmed v1 still cached on the servers.

**wandb project.** `drtc-lab` (`lab/lab.yaml`). `drtc` stays as the minor's project.

**Recording from now on.** `lab record` pipeline:
1. Record into `<name>_raw`.
2. Run the checks.
3. Autotrim, saved as `<name>` (the trim report and parameters go in the dataset card and registry).
4. Push both, public.

Autotrim is rewritten on LeRobot 0.7's dataset tools, without the dataset editor. A test checks it makes the same
cuts as the old version on `so101_cubebox_v1_raw`. It must be done before the next recording.

### Compatibility (required)
Everything carried over must work as in `minor-final`:
- **Kept models:** the two kept models load and give identical outputs.
  - Their entries equal the old ones apart from the renamed repos (tests pass).
  - The old-vs-new output check runs in 1.8.
- **Experiments:** every experiment composes the same run config (tests pass, Linux and Windows).
  - The 10 fault conditions exist on the new system.
- **Calibration:** the files work without recalibrating.
- **Rig:** the same rig (cameras, ports, Pi and laptop).
- **Commands:** the old `lab/bin/*` names stay as aliases.
- **Results:** `results/index.csv` imports with the renames applied.

Findings so far (2026-11-18):
- **Calibration:** LeRobot 0.7 looks for calibration in `calibration/robots/so_follower/` and
  `teleoperators/so_leader/`, not `so101_*`. Both folders already exist on the laptop, but hold **old calibrations
  from 2026-11-05**; the ones in use are `so101_*` from 2026-11-10. Relying on the default would silently use the
  wrong calibration. The rig profile therefore names the calibration files explicitly, the lab layer always
  passes `calibration_dir`, and a fingerprint check catches mismatches.
- **Datasets:** `so101_cubebox_v1` (29,525 frames) and `so101_cubebox_v1_trim` (25,067 frames, trim saved) load in
  0.7, video decoding included.

### Decisions (2026-11-18)
- **1a: our own system on upstream `remote_inference`**, using Jack Vial's DRTC ideas (ii). We are building our own
  rather than replicating DRTC exactly. The original DRTC (`minor-final`) stays runnable, so the two can be
  compared and the better one kept.
- **1b: GR00T N1.5 kept** as a plugin (i).
- **1c: full fine-tunes on trimmed data wait** until Phase 1 is done (i).

## Phase 2: Shortlist
**Started 2026-11-18, paused to finish Phase 1. Findings so far:**
- **ArmnetBench v0.1** (arXiv 2607.24481), the strongest evidence.
  - **Setup:** SO-101 arms, 3 cameras, 20 fps, **50 teleop demos per task**, trained per task, 12 tasks, 2,518
    real rollouts.
  - **Overall strict success:** π0.5 **47.6%**, π0 35.1%, GR00T N1.7 29.4%, Diffusion 26.7%, ACT 19.2%,
    MolmoAct 2 18.9%, SmolVLA 15.0%.
  - **Closest tasks to ours:** transfer_cube (π0.5 86%, π0 47%, GR00T N1.7 47%, ACT 0%) and eye_drops_to_basket
    (π0 70%, π0.5 67%, GR00T N1.7 67%, ACT 63%).
  - **Recipes:** read on 2026-11-18, see "ArmnetBench recipes" below.
- **Project-IRA π0.5 on SO-101** (Hub model cards): `lerobot/pi05_base`, batch 32, 30k steps, checkpoints every 2k.
  The best checkpoints were at **8k–10k steps**; after ~20k they were unusable on the robot (overfitting, like
  our minor's validation curves).
- **SmolVLA:** the LeRobot docs recommend ~50 episodes and 20k steps. Community results are weak: 0–30% with
  ~20–60 episodes; 89% *grasping* with 340 episodes.
- **GR00T:** N1.5 and N1.7 SO-101 community cards (e.g. raymondlo84-nvidia, 10k steps, batch 8, lr 1e-4) have no
  success rates. NVIDIA's N1.7 path: 16-step absolute action chunks, ≥40 GB to fine-tune.
- **MolmoAct 2:** 56.7% on SO-100/101 in its paper (pretrained on 38k community SO-100/101 episodes), but 18.9% in
  ArmnetBench.
- **X-VLA, EVO1, LaWAM:** no SO-101 success rates found.
- **SO-101 failure and recovery benchmark** (arXiv 2606.08881): 100 demos per task, 4 tasks, 20 trials each.
  - **Average success:** π0.5 **56.25%**, Wall-X 51.25%, ACT 33.75%, SmolVLA 32.5%.
  - **Multi-object packing into a box:** π0.5 55%, ACT 10%, SmolVLA 10%.
  - **Recovery after a failure:** π0.5 30.8%, ACT 6.5%.
  - **Failures:** execution failures (grasp instability, repetition loops) dominate for all models; π0.5 has the
    fewest state mismatches.
  - No recipes in the paper.
- **"Does validation loss predict real-robot success?"** (arXiv 2611.00283): SO-101, a green ball among
  distractors into a basket, 225 demos, budgets 12 to 200, LeRobot 0.6.2 default recipes. Recipes:
  - ACT: 100k steps;
  - Diffusion: 100k steps;
  - SmolVLA: 20k steps, from pretrained;
  - π0.5: action expert only, 6k steps.

  Findings:
  - **Success at 200 episodes:** π0.5 **73%**, SmolVLA 38% (71% of the runs where it started moving); ACT 37/40
    at 100 episodes.
  - **"Never started" (the arm stays at rest) is a major failure mode:** untrimmed Diffusion 96% of rollouts, **trimmed
    Diffusion 9%**; SmolVLA 47%. This is direct support for trimming every dataset.
  - **Validation loss is a weak guide:**
    - held-out *action error* tracks success within a policy, but no better than the number of episodes;
    - held-out *loss* rises during training for diffusion/flow-matching models while action error falls, so it is
      not a stopping rule for them;
    - final checkpoints were within 5.3% of the best;
    - **use held-out action error to decide when to stop, not to choose models; judge only on the robot**, and
      report "never started" separately from success.
  - **Plateaus:** ACT at 20k–40k steps, SmolVLA at 2k–10k, Diffusion at 10k–30k (with ≤50 episodes).
- **Tentative shortlist (for the user to decide):** π0.5, π0, GR00T N1.7 (and N1.5 for continuity), Diffusion as
  the strongest small model; ACT for the edge fallback. SmolVLA and MolmoAct 2 rank low in independent tests.

### ArmnetBench recipes (read 2026-11-18)
**Where they are.** The HF collection holds only the three datasets. The checkpoint links are in the dataset
README (`armnet/armnetbench_v01_lerobot_so101`, `..._bimanual_so101`), one checkpoint per (task, policy).
Each rollout's `policy_repo_id` and `success` are in `meta/episodes`. The formats differ by family:
- ACT and Diffusion: LeRobot `train_config.json`;
- GR00T N1.7: Isaac-GR00T `experiment_cfg/conf.yaml`;
- π0 and π0.5: openpi JAX checkpoints with no config; the configs are in GitHub `pravsels/openpi`, branch
  `task/v50-retrain`, `src/openpi/training/config.py` (`_make_v50`, `LeRobotSO101DataConfig`).

**Their data:** 50 demos per task, 20 fps, front/top 576×1024 and wrist 720×1280 cameras, ~9.1k frames per task.
Ours: 110 episodes, 30 fps, 25,067 frames.

**Success by embodiment,** recomputed from `meta/episodes`. Single-arm is our setting; transfer_cube is bimanual.

| | π0.5 | π0 | GR00T N1.7 | Diffusion | ACT | MolmoAct 2 | SmolVLA |
|---|---|---|---|---|---|---|---|
| Single-arm, 8 tasks | **45.4%** | 36.4% | 34.6% | 22.1% | 20.3% | 14.8% | 13.6% |
| Bimanual, 4 tasks | **52.1%** | 32.5% | 19.2% | 35.8% | 2.0% | 8.8% | 12.0% |
| eye_drops_to_basket (single) | 67% | 70% | 67% | 43% | 63% | 33% | 23% |

**Recipes:**

| | π0.5 / π0 (identical except base) | GR00T N1.7 | ACT | Diffusion |
|---|---|---|---|---|
| Trainer | openpi (JAX), fork with per-timestep action norm | Isaac-GR00T, 4× RTX 5090 DDP | LeRobot | LeRobot |
| Base | `pi05_base` / `pi0_base` | `nvidia/GR00T-N1.7-3B` | ResNet18 ImageNet | ResNet18 ImageNet |
| Trained | everything (openpi default, no LoRA in params) | projector + DiT + VLLN; LLM and vision frozen | everything | everything |
| Steps × batch | **25k × 32** (2 GPU × 16) | **20k × 32** | **20k × 16** (10k + 10k warm restart) | 20k × 32 on 5 single-arm tasks; 80k × 8 on 3; several resumed from an earlier checkpoint, so these are not totals |
| ≈ epochs of their data | ~88 | ~70 | ~35 | ~70 |
| LR | 2.5e-5 peak, 1k warmup, cosine to 2.5e-6 over 100k (so nearly flat over 25k); EMA 0.999 | 1e-4 cosine, 5% warmup, wd 1e-5, grad clip 1 | 1e-5 (backbone too), wd 1e-4, constant | 1e-4, diffusers schedule, 500 warmup, wd 1e-6 |
| Chunk | 30 predicted (1.5 s), executed per their client | 16 used (0.8 s) of the 40-step head | 30 / 30 executed | horizon 64, 32 executed, 2 obs steps |
| Actions | **delta joints, absolute gripper**; per-timestep action normalization (per-step mean/std/q01/q99) | **relative arm, absolute gripper**; percentile stats | absolute, mean/std | absolute |
| Cameras | front + wrist (top dropped) | front, wrist, top | top, wrist, front at 480×640 | all three at full resolution, no crop, one encoder per camera |
| Other | default prompt per task | colour jitter, crop 244→224, state dropout 0.2 | image augmentation on (jitter, sharpness, small affine), KL 10 | DDPM 50 steps (DDIM 20 on 2 tasks), down_dims 512/1024/2048, no augmentation |

**Caveats:**
- The Diffusion recipe is not uniform. `cable_clip_remote_v2_diffusion`'s config names the block_stack dataset,
  which looks like an upload mistake.
- MolmoAct 2: the `quantile_norm_fix_25k` checkpoints are what the README links. The dataset has 60 rollouts per
  task because the earlier `_20k` (before the fix) was also evaluated, so the 14.8% mixes both. The fixed
  checkpoints alone: 19.6% single-arm, 63% on eye_drops_to_basket.

**The user's view (2026-11-18):** these are not the recipes to follow; they are reference only. The user is
looking for the right source. The training configs are confirmed after that research.

### Official recipes (read 2026-11-19)
Each model's own docs: paper, official repo, LeRobot page. Our data: 110 episodes, 30 fps, ~25k frames (≈780 steps per
epoch at batch 32). **User's leaning (2026-11-19):** SmolVLA full fine-tune, π0.5 full fine-tune, MolmoAct 2 LoRA.

| | What is trained | Steps × batch | LR, schedule | Chunk (executed) | Actions, norm | Other |
|---|---|---|---|---|---|---|
| **SmolVLA** (paper 2506.01844, LeRobot docs) | Action expert only, VLM frozen (paper). No full fine-tune documented; LeRobot has the flags (`freeze_vision_encoder`, `train_expert_only`, both `true` by default) | 20k × 64 (docs, "~4 h on one A100"); paper: 200k for real tasks, "much fewer works" | Fine-tune LR not stated; LeRobot default 1e-4, warmup 1k, cosine to 2.5e-6 over 30k | 50 (paper's sync eval runs all 50; LIBERO: 10–50 best) | Not stated | 512×512 images, ~50 episodes recommended (25 was "not enough"). Paper SO-100: 78% multi-task |
| **π0.5** (openpi `config.py`, LeRobot docs) | Full fine-tune in every openpi π0.5 config; LoRA only exists for π0/π0-FAST. LeRobot's main example is full too (expert-only "costs some success") | openpi: 20k × 32 (DROID), 20k × 64 (ALOHA), 30k × 256 (LIBERO). LeRobot LIBERO: 30k × 64 on one 80 GB GPU | AdamW b2 0.95, clip 1.0; 2.5e-5 peak, 1k warmup, cosine to 2.5e-6 over 30k (default) or flat 5e-5 (LIBERO); EMA 0.99 (0.999 LIBERO) | 50 (ALOHA default); 16 (DROID); 10 (LIBERO) | **Delta joints, absolute gripper** for joint-space robots (ALOHA, DROID joint pos); **quantile** norm (LeRobot example uses mean/std) | 224×224; full FT > 70 GB |
| **GR00T N1.5** (Isaac-GR00T n1.5-release) | Projector + DiT; LLM and vision frozen. LoRA possible (r 64, α 128) but full head recommended | 10k (script, blog) to 20k ("batch to the max") × 32 per GPU | 1e-4, AdamW (0.95, 0.999), wd 1e-5, warmup 5%, cosine | 16 | Absolute, **min-max** | crop 0.95 → 224×224, colour jitter (0.3/0.4/0.5/0.08); ~25 GB. LeRobot dropped N1.5 (our plugin) |
| **GR00T N1.7** (Isaac-GR00T main, LeRobot docs) | Projector + DiT + VLLN; LLM and vision frozen; no LoRA | NVIDIA: 10k × 64 global; LeRobot SO-101: **20k × 64** | 1e-4, warmup 5%, wd 1e-5 | **16 of the 40-step head** for SO-100 (LeRobot rollout executes 8, RTC horizon 8) | **Relative arm, absolute gripper**; percentile (q01/q99) | state dropout 0.2, colour jitter; ≥ 40 GB |
| **MolmoAct 2** (paper 2605.02881, allenai repo, LeRobot docs) | LeRobot docs: **LoRA (default) for < 200 demos**, "similar or even better than fft"; LoRA r 64 on the LLM + ViT, connector and action expert fully trained | Paper's real-robot eval fine-tunes: 50k × 16 (8 H100); SO-100 base itself: 100k × 64; LeRobot LIBERO 10k × 32 × 8 GPUs | LoRA: all groups 5e-5 (fft: VLM 1e-5, ViT/connector 5e-6, expert 5e-5); warmup 200, cosine, decay horizon 24k | 30 at 30 Hz | **Absolute joints**, quantile, gripper not normalized | Continuous flow matching, 8 flow times in training, 10 inference steps. Augmentation **on**: 95% crop, ±5° rotation, colour jitter, blur p 0.2 (LeRobot disables it by default) |
| **ACT** (paper 2304.13705, tonyzhaozh/act, LeRobot) | Everything | Original: batch 8, 2000 epochs sim, "≥ 5000 epochs real, or 3–4× after the loss plateaus"; LeRobot: 100k steps, batch 8 | 1e-5 (Adam) | 100; temporal ensembling at eval (our server refuses it) | Absolute joints | KL 10, 4/7 layers, dim 512/3200; best validation checkpoint; "train longer if jerky" |
| **Diffusion** (paper 2303.04137, real-stanford repo) | Everything; ResNet-18 from scratch, GroupNorm, spatial softmax | batch 64; ~600–3000 epochs | AdamW 1e-4 (0.95, 0.999), wd 1e-6, cosine, 500 warmup; EMA (power 0.75, max 0.9999) | To 2, Tp 16, Ta 8 **at 10 Hz** (1.6 s / 0.8 s; ≈ 48 / 24 at 30 fps) | Absolute position, min-max to [−1, 1] | 240×320 → crop 216×288 (random), 100 DDPM train steps, DDIM 10–16 at inference; robust to ≤ 4 steps latency |

**Where our recipes differ:**
- `smolvla_full_45ep` is named full but trains the action expert only (LeRobot's defaults). A full fine-tune needs
  `policy.freeze_vision_encoder: false` and `policy.train_expert_only: false`; that is not what the authors do.
- `pi05_full_openpi`: absolute actions, ~7.8k steps, no EMA. openpi: delta joints + absolute gripper, 20–30k steps,
  EMA, quantile norm.
- `groot_n17_nvidia`: chunk 40 and absolute actions. NVIDIA/LeRobot SO-100: chunk 16, relative arm, 20k × 64.
- `molmoact2_full`: `train_mode_vlm: fft`. The docs recommend LoRA for our 110 episodes; augmentation is off.
- `act_100ep`: batch 64, ~39k steps, LR default. Original: batch 8, LR 1e-5, much longer.
- `diffusion_full`: ImageNet backbone without GroupNorm; the original trains from scratch with GroupNorm and EMA.
  `n_obs_steps: 1` stays (the server refuses history; the original uses 2).

### Recipes (confirmed 2026-11-19)
**Rule (the user's, 2026-11-19): each model trains with its original recipe, the same way every time.** π0.5 and
GR00T N1.7 in particular always use their original relative actions. A newly recorded and finalised dataset must be
ready for all models with no manual step (see "Dataset preparation" below).

| Model | Recipe | Trained | Steps × batch | Chunk (executed) | Actions | Notes |
|---|---|---|---|---|---|---|
| SmolVLA | `smolvla_authors` | action expert, VLM frozen | 20k × 64 | 50 (50) | absolute | LR 1e-4, cosine to 2.5e-6; the user's full fine-tune idea is not the authors' recipe (a second run can be added) |
| π0.5 | `pi05_openpi` | everything | 20k × 64 | 50 (50) | **relative arm, absolute gripper** | openpi `pi05_aloha_pen_uncap`: LR 2.5e-5, wd 1e-10, EMA 0.99, quantile |
| GR00T N1.5 | `groot_n15_nvidia` | projector + DiT | 10k × 32 | 16 (16) | absolute | LR 1e-4, wd 1e-5, 5% warmup |
| GR00T N1.7 | `groot_n17_nvidia` | projector + DiT + VLLN | 20k × 64 | 16 (16) | **relative arm, absolute gripper** | LR 1e-4; **native RTC is off with relative actions** (modeling_groot.py), so chunk mode only |
| MolmoAct 2 | `molmoact2_lora` (model renamed `molmoact2_cubebox_v1_lora`) | LoRA r 64 on the VLM; connector + expert full | 50k × 16 | 30 (30) | absolute | LR 5e-5 all groups, augmentation on, compile on |
| ACT | `act_original` | everything | 100k × 8 | 100 (100) | absolute | LR 1e-5, KL 10 |
| Diffusion | `diffusion_original` | everything, ResNet-18 from scratch + GroupNorm | 600 epochs × 64 (214k steps) | 48 (24) | absolute | the paper's 16/8 at 10 Hz kept in seconds; DDIM 100 / 10; EMA warmup |

Every recipe pins its key values explicitly (not LeRobot's defaults, which can change), and each has a `source`.

**Checkpoints:** every 1k steps (ACT 5k, Diffusion 10k). `lab train` keeps every checkpoint's weights and the
training state only for the latest (`prune_training_state`; LeRobot 0.7 keeps everything, tens of GB each for π0.5).
The best checkpoint is chosen by held-out action error, as decided earlier.

**Found while checking the flags against LeRobot 0.7:**
- With `resize_shape` set, Diffusion ignores `crop_shape` and derives the crop from `crop_ratio`. The old
  `diffusion_full` therefore trained without its crop; `diffusion_original` uses `crop_ratio: 0.9` (216×288).
- `null` in a recipe reached lerobot-train as the string "None"; `flag_value` now writes `null`.

**Test run of the original recipes (H200, commit cc1ccc6e):**

| Model | Train (30 steps) | Peak train GPU | Serve check |
|---|---|---|---|
| ACT | ok, 87 s, eval 0.85 | 6.2 GB | chunk 13.2 ms (1 step) vs margin 80 |
| Diffusion | ok, 102 s, eval 1.00 | 10.0 GB | chunk 55.8 ms (2 steps) vs margin 16 (H 24) |
| SmolVLA | ok, 115 s, eval 0.46 | 14.7 GB | chunk 193 ms (6) / RTC 319 ms (10) vs margin 30 |
| π0.5 | **failed**: the pinned `pi05_base` processors have no `relative_actions_processor` step | — | — |
| GR00T N1.5 | ok, 298 s, eval 0.22 | 20.5 GB | chunk 77 ms (3) / RTC 98.5 ms (3) vs margin 8 |
| GR00T N1.7 | ok, 185 s, eval 1.14 (relative stats) | 40.2 GB | chunk 119.7 ms (4) vs margin 8. "rtc_guided" 119.8 ms: the same time, as expected with its native RTC off for relative actions |
| MolmoAct 2 | **failed**: "Error(s) in loading state_dict" (LoRA) | — | — |

Fixes:
- π0.5: base pin moved to `22c30e5b`. Hub diff against `a538eb27`: only README and the two processor files changed
  (upstream "Add relative action processor steps", 2026-06-03); the weights are identical.
- MolmoAct 2: 0.7 applies LoRA in `__init__` and then loads the checkpoint strictly, so a full-weight checkpoint
  (our base) does not fit the adapter-wrapped model. `drtc.compat.molmoact2` (installed by `lab train`) loads as
  fft, then applies the adapters exactly as `__init__` would, then compiles. LoRA checkpoints load unchanged.
- [x] Rerun (commit 1fd7db08), both ok:

  | Model | Train (30 steps) | Peak train GPU | Serve check |
  |---|---|---|---|
  | π0.5 | ok, 457 s, eval 0.087 (relative stats) | 76.5 GB | chunk 178.4 ms (6) / RTC 257.5 ms (8) vs margin 30: fits. The server's relative-chunk RTC path works |
  | MolmoAct 2 (LoRA) | ok, 391 s, eval 0.33 | 25.2 GB (55.8 GB full FT) | chunk 286.7 ms (9) fits margin 10 (barely); RTC 420.4 ms (13) **does not fit** |

  MolmoAct 2 serves 40% faster than the full fine-tune (475 / 604 ms). The likely reason: the checkpoint's config
  now has `compile_model: true`, so the server compiles the same blocks training does. LoRA itself only adds work
  (the adapters are not merged).
- [x] MolmoAct 2 latency, option A (serve settings, no retraining): `serve_check.py --set` times a copy of the
  checkpoint with config keys changed. Sweep (H200, 2026-11-19, commit 6d487db2), margin 10 (H 30, s_min 20):

  | Setting | chunk ms (steps) | RTC ms (p90) (steps) | RTC fits |
  |---|---|---|---|
  | 10 flow steps (default) | 292 (9) | 433 (448) (13) | no |
  | 8 | 257 (8) | 372 (377) (12) | no |
  | 6 | 212 (7) | 304 (322) (10) | no (needs < 10) |
  | 5 | 207 (7) | 279 (290) (9) | yes, 1 step spare |
  | 4 | 188 (6) | 247 (255) (8) | yes, 2 spare |
  | 10, `compile_model=false` | 507 (16) | 646 (657) (20) | no |

  - Compiling is what made the LoRA model faster (507 → 292 ms). Serving must keep `compile_model: true`.
  - Each flow step costs about 17 ms in chunk mode and about 31 ms in RTC mode (guided RTC runs a backward pass per
    step). About 120 ms per call is fixed: the VLM and preprocessing.
  - The check times only the server. Network time between robot and server comes on top, so 1 spare step is thin.
- [ ] MolmoAct 2 flow steps for serving (**decided 2026-11-19**, applied in Phase 4): after the real training,
  compare held-out action error at 10 / 6 / 5 / 4 flow steps and serve with the fewest whose error stays close to
  10 steps, preferring 4. If the error drops noticeably at 5 or fewer, fall back to option C: lower s_min for
  MolmoAct 2 (s_min 16 gives margin 14, and 10 flow steps then fits with RTC). `lab serve` gets the same config
  override as the check once the count is chosen.

### Dataset preparation (2026-11-19)
`drtc.data.prepare` makes what a recipe needs beyond the recorded dataset. Today that is relative-action stats:
LeRobot needs action stats computed in relative space over chunks of the policy's `chunk_size`, with the same joints
excluded, so π0.5 (chunk 50) and GR00T N1.7 (chunk 16) need different stats.
- Each need is a variant folder `$HF_LEROBOT_HOME/.drtc_variants/<repo>/<revision>_rel_c<chunk>_x<joints>/`:
  symlinks to the pinned dataset plus its own `meta/stats.json`. The dataset itself is never written.
- `lab train` makes the variant on first use on any host and passes it as `--dataset.root`; `lab prepare <dataset>`
  makes all of them ahead of time for every recipe in use; `lab record` runs it after pushing.
- A recipe with `use_relative_actions` must set `chunk_size` and `relative_exclude_joints` explicitly (checked).
- Made and checked on the laptop for `so101_cubebox_v1`: only the action stats differ; the dataset's stats.json is
  unchanged (checksum).

### Shortlist (decided 2026-11-18)
**Models:**
- SmolVLA
- GR00T N1.5
- GR00T N1.7
- π0.5
- MolmoAct 2
- ACT, also the edge fallback
- Diffusion Policy

**Retired:** π0, π0-FAST, X-VLA, DOT, SFP (`status: retired` in `lab/models/cubebox_v1.yaml`).

**DRTC compatibility.** DRTC's own parts work for any policy that LeRobot 0.7's policy server can serve as action
chunks:
- the trigger (request when the queue is ≤ H − s_min);
- the latency estimators;
- the cooldown re-request.

RTC (inpainting the next chunk onto the actions still queued) is extra. It needs a flow-matching policy that
declares `supports_rtc()`.

| Model | Chunk H (current recipe) | Served by the 0.7 server | RTC in 0.7 | To do |
|---|---|---|---|---|
| SmolVLA | 50 | yes (kept model run on the replay robot, ~280 ms on the 4060) | yes | — |
| GR00T N1.5 (our plugin) | 16 | yes (same path as the kept model) | **yes since 2026-11-18** (see below) | 613 ms on the laptop vs a 533 ms chunk, so serve from the H200. H − s_min = 8 steps is tight |
| GR00T N1.7 (upstream `groot`) | 40 | yes (smoke-trained) | yes (H200: RTC 121 ms) | — |
| π0.5 | 50 | yes (smoke-trained) | yes | — |
| MolmoAct 2 | 30 | not tested | only with `inference_action_mode: continuous` (its default) | smoke train + serve on the H200 |
| ACT | 100 | yes (serve check on the laptop: 27 ms per chunk) | no (not flow matching): chunk switching only | temporal ensembling must stay off (the server refuses it) |
| Diffusion | 16 executed | yes once trained with `n_obs_steps: 1` (now in `diffusion_full`) | no | — |

**Done on 2026-11-18 for the test runs:**
- **GR00T N1.5 plugin, guided RTC on 0.7.**
  - `rtc_config` config field; the server sets it for `rtc_guided`.
  - An `RTCProcessor` built at init, `supports_rtc()`, and the 0.7 `denoise_step` call.
  - The time and sign mapping is upstream's NOISE_AT_ZERO one.
  - Only the robot's 6 action dims are guided; GR00T's 26 padded dims keep their unguided velocity.
  - Tests in `tests/test_groot_n15_rtc.py`: the prefix is pulled in, nothing changes past the execution horizon,
    the padding is untouched, the server's RTC call interface is accepted, and the config round-trips.
  - The old setup's smoothness guidance (method C) is not ported; it belongs to Phase 5.
- **Recipes and registry:**
  - `groot_n17_nvidia` now uses `policy: groot` and `policy.base_model_path`.
  - `diffusion_full` now uses `n_obs_steps: 1` and `drop_n_last_frames: 16`.
  - GR00T N1.7 and MolmoAct 2 are `planned`, so the test run can reach them.
  - MolmoAct 2 is in the registry's RTC policies.
  - The naming rule accepts `grootn17_*` for `groot`. The kept N1.5 model still carries the old type `groot`.
- **`tests/smoke/train.sh`:** the 7 shortlisted models.
  - Each one trains 30 steps and saves the step-30 checkpoint.
  - Then `tests/smoke/serve_check.py` runs on that checkpoint, using upstream's `load_deployment`. That covers
    admission and warmup in `chunk` and, for RTC policies, `rtc_guided`.
  - It times each request and compares it with H − s_min at 30 fps.
  - Results: `logs/smoke/summary.txt` and `logs/smoke/serve_<model>.json`.
- **Serve check on the laptop (RTX 4060, s_min 20, 30 fps):**

  | Model | Checkpoint | Modes admitted | H | Time per chunk (median) | DRTC margin |
  |---|---|---|---|---|---|
  | ACT | the Phase 1 300-step test | chunk | 100 | chunk 27 ms (1 step) | 80 steps |
  | SmolVLA | 5 steps, batch 2 (removed after the check) | chunk + rtc_guided | 50 | chunk 157 ms (5 steps); RTC 229 ms (7 steps) | 30 steps |

  Peak GPU: ACT 0.3 GB, SmolVLA 1.5 GB. Guided RTC costs about +45% per chunk for SmolVLA (the autograd
  correction at each denoising step).

ACT and Diffusion get DRTC scheduling with the Butterworth filter and the Phase 5 smoothing (method A), but no
RTC. The other five can be compared with and without RTC.

**Next:**
- [x] Recipes: each model's original, confirmed 2026-11-19 (see "Recipes (confirmed 2026-11-19)" below).
- [x] Test runs of all 7 on the H200: `bash tests/smoke/train.sh` (the user runs it). All 7 train and serve; 6 fit the timing margin.
  - **First run (2026-11-18, commit c4d24663).** Serve times are the H200's; 30 fps, s_min 20:

    | Model | Train (30 steps) | Peak train GPU | Serve check |
    |---|---|---|---|
    | ACT | ok, 170 s, eval 0.81 | 44.6 GB | admitted chunk; 13.7 ms (1 step) vs margin 80 |
    | SmolVLA | ok, 170 s, eval 0.46 | 14.7 GB | admitted chunk + rtc_guided; 175 ms (6 steps) / RTC 301 ms (10 steps) vs margin 30 |
    | Diffusion | ok, 167 s, eval 1.00 | 9.6 GB | refused: no `chunk_size` (fix 1) |
    | π0.5 | ok, 585 s, eval 0.27 | 45.5 GB | refused: feature names ≠ checkpoint inputs (fix 2) |
    | GR00T N1.5 | ok, 261 s, eval 0.22 | 20.5 GB | refused: postprocessor returned one action, not the chunk (fix 3) |
    | GR00T N1.7 | failed: gated Hub repo (`nvidia/Cosmos-Reason2-2B`, the backbone) | — | — |
    | MolmoAct 2 | failed: the base config has 4 fields 0.7 dropped (fix 4) | — | — |

  - **Fixes:**
    1. `drtc.control.chunk_adapters` (installed by `lab serve` and the serve check) gives Diffusion a chunk spec
       (prediction = execution = `n_action_steps`; `n_obs_steps` must be 1). It also adds the observation-time
       axis to state: without it the server's (B, D) state fails in `generate_actions`, as a test showed.
       Tests: `tests/test_chunk_adapters.py`.
    2. Camera renames. `lab serve` declares the checkpoint's camera names (the recipe's `cameras`, the training
       `rename_map`), and `lab run` passes the same map as `--rename_map`. Camera semantics are kept.
    3. The GR00T N1.5 `action_unpack_unnormalize` step keeps every timestep of a (B, T, D) chunk. Upstream's N1.5
       port kept only the last one; single actions are unchanged. Test in `tests/test_groot_n15_rtc.py`.
    4. `lab train` passes a base with an outdated config as an upgraded local copy (`drtc.train.upgraded_base`:
       files linked, unknown fields dropped and printed). `molmoact2_full` now also sets
       `policy.train_mode_vlm: fft`; 0.7's default is LoRA on the VLM.
    5. GR00T N1.7: the user accepts the gate of `nvidia/Cosmos-Reason2-2B` (auto-approved) with the Hub account
       whose token is on the H200.
  - **Second run (2026-11-18, commit d30fef52), the 5 that failed:**

    | Model | Train | Peak train GPU | Serve check |
    |---|---|---|---|
    | Diffusion | ok, 177 s, eval 1.00 | 9.6 GB | **admitted** chunk; 54.9 ms (2 steps) vs margin 8 |
    | GR00T N1.5 | ok, 245 s, eval 0.22 | 20.5 GB | **admitted** chunk + rtc_guided; 77.7 ms (3 steps) / RTC 99.2 ms (p90 218) vs margin 8. The plugin's RTC works on the real model |
    | π0.5 | ok, 416 s, eval 0.27 | 45.5 GB | refused: feature names ≠ checkpoint inputs (fix 6) |
    | MolmoAct 2 | ok, 747 s, eval 0.23 (the base upgrade works) | 55.8 GB | refused: RGB shape ≠ checkpoint (fix 6) |
    | GR00T N1.7 | failed: gated repo again (fix 5 not done yet) | — | — |

  - **Fix 6.** A fine-tune keeps its base checkpoint's nominal input_features:
    - π0.5: 3 camera slots at 224×224 and a 32-d state;
    - MolmoAct 2: cam0/cam1 at 224×224.

    The 0.7 server demands an exact match. `drtc.control.chunk_adapters` overrides `validate_chunk_input_features`
    for both, as upstream does for VLA-JEPA:
    - any RGB resolution, since both policies resize images themselves;
    - π0.5: a subset of the camera slots (the missing one gets the same empty, masked image as in training) and a
      state no longer than max_state_dim (padded);
    - MolmoAct 2: exact names and state.

    Tests in `tests/test_chunk_adapters.py`.
  - **Third run (2026-11-18, commit a327e859)**, after the user's new HF token opened the GR00T N1.7 gate:

    | Model | Train | Peak train GPU | Serve check |
    |---|---|---|---|
    | π0.5 | ok, 377 s, eval 0.27 | 45.5 GB | **admitted** chunk + rtc_guided; 178.7 ms (6 steps) / RTC 256.0 ms (8 steps) vs margin 30; serve peak 10.3 GB |
    | MolmoAct 2 | ok, 603 s, eval 0.23 | 55.8 GB | refused in warmup: "Inference tensors cannot be saved for backward" (fix 7) |
    | GR00T N1.7 | ok, 178 s, eval 1.21 | 36.4 GB | **admitted** chunk + rtc_guided; 119.5 ms (4 steps) / RTC 121.2 ms (4 steps) vs margin 20; serve peak 15.3 GB |

  - **Fix 7.** MolmoAct 2's action expert caches its per-step timestep modulations across calls. The chunk-mode
    warmup creates them under `torch.inference_mode`, and the rtc_guided warmup then uses them in the RTC
    correction's autograd graph. `drtc.control.chunk_adapters` rebuilds a cache holding inference tensors once, when
    called outside inference mode; ordinary tensors work in both modes. Test in `tests/test_chunk_adapters.py`
    (fails without the fix with the H200's error).
  - **Fourth run (2026-11-18, commit 1f503991), MolmoAct 2:** train ok, 406 s, eval 0.23, 55.8 GB. Serve
    **admitted** chunk + rtc_guided (fix 7 works), but too slow: chunk 475.1 ms (15 steps), RTC 603.7 ms (p90 617.5,
    19 steps) vs margin 10 (H 30, s_min 20): **does not fit**. Serve peak 13.6 GB.
  - [ ] **[open]** MolmoAct 2's latency: options in the 2026-11-19 session (speed it up, lower s_min, retrain with
    fewer image tokens, or keep it for Phase 4 only).
- [x] GR00T N1.5 plugin: guided RTC on 0.7, with `supports_rtc()` (2026-11-18).

**What LeRobot 0.7 covers:**
- π0.5, π0 and GR00T: `use_relative_actions` with `relative_exclude_joints: [gripper]`.
- EMA for any policy: `--ema.type`.
- Not covered: openpi's per-timestep action normalization for π. It would need a processor step or plugin.

**How our current recipes differ:**
- `pi05_full_openpi` / `pi0_full_openpi`: absolute actions, chunk 50, 10 epochs (~7.8k steps), no EMA.
- `groot_n17_nvidia`: chunk 40, absolute actions.
- `act_100ep`: batch 64, ~39k steps, no augmentation, chunk 100 (default).
- `diffusion_full`: horizon 32 / 16 executed, 240×320 with crop, DDIM 10.

- [x] For each candidate, its recipe source: the model's own docs (2026-11-19, "Official recipes" above).
  - [x] Inference speed against DRTC's timing: the H200 test runs above (MolmoAct 2 does not fit yet).
- [x] Selection criteria: the user's choice (2026-11-18).
- [x] Candidates reviewed; shortlist of 7 (2026-11-18).
- [x] `lab/models/cubebox_v1.yaml`: the shortlist `planned`, the rest `retired`.
  - [x] Recipes confirmed (2026-11-19).

## Phase 3: Faster training
- [ ] **[open]** Hardware available (H200 only, or the A100 too; one GPU or several)
- [ ] Measure the time per step for each shortlisted model on 0.7 as-is: bf16, compiled image transforms,
      dataloader workers, video decoding
  - Baseline tool (2026-11-19): `tests/bench/train_speed.sh` runs each recipe for 200 steps (no eval, no
    checkpoint), averages LeRobot's logged step times after step 50 (data wait, preprocessing, update) with GPU
    utilization from nvidia-smi, and projects the full recipe run. `LABEL=... EXTRA=...` compares a setting.
  - The data share of a step says where to look: high = data loading / video decoding (workers, decoder), low with
    low GPU utilization = small kernels (compile, batch), high utilization = the model itself (multi-GPU, LoongForge).
- [x] Hardware (2026-11-19): the H200 now (141 GB, 48 CPUs); the A100 later (not available yet); the laptop (RTX
  4060 8 GB, 24 CPUs); more GPUs possible later.
- Data loading, measured on the laptop (2026-11-19), so101_cubebox_v1: 2 cameras, AV1 800×600, keyframe every 2
  frames (seeking is already cheap), 0.4 GB of video. One random frame decodes in 3.5 ms (torchcodec; pyav 8.7 ms).
  DataLoader at batch 64:

  | Images as | Workers | Samples/s |
  |---|---|---|
  | float32 (LeRobot 0.4.3, the minor phase) | 8 | 180 |
  | uint8 (LeRobot 0.7's training default) | 8 | 555 |
  | uint8 | 16 | 633 |

  0.7 already sends uint8 images (3× faster than the minor phase's loading). More workers help little. The rest
  of the cost is moving full 800×600 frames between processes; every model uses a smaller image (224 to 512 px).
  Whether data still limits a model depends on its step time (H200 baseline): it needs batch / step_s samples/s.
- [ ] Options to compare:
  - Data: more workers; MolmoAct 2's augmentation on the GPU (`dataset.image_transforms.backend: gpu`, same
    transforms) instead of in the workers at full resolution; a decoded or pre-resized frame cache as a dataset
    variant (`drtc.data.prepare`), only for models the baseline shows waiting on data (pre-resizing uses a
    different resize than serving does, so it needs checking).
  - Memory knobs that do not change the result: gradient checkpointing off where memory allows (π0.5, MolmoAct 2).
  - Compile: `policy.compile_model` (π0.5, SmolVLA, Diffusion; MolmoAct 2 already on).
  - One GPU, several jobs: ACT (6 GB) and Diffusion (10 GB) next to a VLA on the H200; ACT on the laptop.
  - bf16 mixed precision for ACT and Diffusion (`accelerator.mixed_precision`): changes the numbers, so it is a
    recipe change to confirm with the user.
  - upstream FSDP2/DDP (`--parallelism.*`), if there is more than one GPU
  - `torch.compile` of the policy
  - LoongForge (π0.5, X-VLA, GR00T N1.6/N1.7; its speedups were measured on 8 GPUs; DCP checkpoints need
    conversion)
- H200 baseline (2026-11-19, commits 5ac3fa7b / 6f6622f1; 200 steps, 8 workers). Machine: 2× Xeon 6505P, 48
  threads, no CPU quota, 251 GB RAM (about 150 GB free).

  | Model | Step | Data wait | Preprocessing | Model update | GPU | Full recipe run |
  |---|---|---|---|---|---|---|
  | MolmoAct 2 | 2.185 s | 0% | **0.467 s** | 1.708 s | 71% | **30.3 h** |
  | π0.5 | 4.901 s | 0% | 0.026 s | 4.864 s | 97% | **27.2 h** |
  | Diffusion | 0.317 s | **75%** | 0.003 s | 0.074 s | 21% | 18.8 h |
  | GR00T N1.7 | 0.600 s | 1% | 0.063 s | 0.529 s | 61% | 3.3 h |
  | GR00T N1.5 | 1.080 s | 0% | **0.773 s** | 0.302 s | 20% | 3.0 h |
  | ACT | 0.107 s | 2% | 0.001 s | 0.102 s | 45% | 3.0 h |
  | SmolVLA | 0.323 s | 2% | 0.008 s | 0.306 s | 69% | 1.8 h |

  One after another: about 87 h. The long runs are MolmoAct 2 and π0.5.
  - Diffusion workers (decoding): 16 → 0.174 s (data 49%), **24 → 0.149 s (data 7%, 8.9 h)**, 32 → 0.175 s. More
    workers also slow the update (0.074 → 0.155 s at 32): the main process competes with them for the CPUs.
  - π0.5 is compute-bound (97% GPU). LeRobot's π0.5 uses eager attention (softmax(QK^T)) on purpose, to keep the
    original numerics; switching to SDPA/flash would change them, so that is a recipe question.
  - GR00T N1.5 and MolmoAct 2 lose a quarter to three quarters of each step in their CPU preprocessing (HF image
    processors in the main process), not in the model.
  - GR00T N1.5's Eagle preprocessing, laptop, batch 32 × 2 cameras: 406 ms at 800×600 against 71 ms at 224×224
    (5.7×, random frames).
- [x] Frame cache for GR00T N1.5 (2026-11-19), recipe `data: {frames: [224, 224]}`, method `eagle`: frames stored
  at the Eagle processor's one 224×224 tile, made with its own bicubic resize on uint8. The float round trip in
  training (÷255, then the plugin back to uint8) changes no value, and the processor leaves a 224×224 frame
  unchanged, so the model input is **bit-identical**: eagle_pixel_values equal in 2/2 batches of 32 (laptop). Laptop:
  Eagle preprocessing per batch 954 → 82 ms; DataLoader 618 → 3918 samples/s; 7.1 GB. Expected on the H200: step
  ≈ 1.08 → ≈ 0.4 s.
- **The H200 container has no FFmpeg libraries** (2026-11-19): torchcodec cannot load there, so LeRobot decodes
  every training video with PyAV (the laptop: 3.5 ms per frame with torchcodec, 8.7 ms with PyAV). This is part of
  why Diffusion's data loading was slow on the H200. Caches are now decoded with the host's training decoder
  (recorded in the cache; rebuilt if it changes). On our data the two decoders give identical frames (20 random
  frames, laptop). Installing FFmpeg in the container (`apt-get install -y ffmpeg`) would let torchcodec load.
- [x] One video setup on every machine (2026-11-19, `drtc.data.video`): recordings are written with PyAV (LeRobot's
  only encoder) as AV1 / yuv420p / keyframe every 2 / CRF 30, passed explicitly by `lab record` (LeRobot's defaults,
  as so101_cubebox_v1 was recorded); training, caches and checks read with torchcodec. `lab train` checks that
  torchcodec loads and stops with instructions otherwise (`--dataset.video_backend=pyav` to use PyAV deliberately),
  so no machine falls back to PyAV silently again.
- Building a cache decodes in the training process, so `build_frames` closes LeRobot's decoders afterwards; forked
  DataLoader workers inheriting them fail with "Could not push packet to decoder".
- Laptop, ACT (2026-11-19, commit 3de1880e): step 0.500 s, data 4%, GPU 99.8%, peak 6.1 GB, full recipe ≈ 13.9 h
  (H200: 3.0 h at 45% GPU). The laptop GPU is the limit, so ACT fits there only as an overnight run; on the H200 it
  can share the GPU (round `together`).
- [x] Frame cache (2026-11-19), recipe `data: {frames: [240, 320]}`, on for Diffusion: `drtc.data.prepare` decodes
  every frame once with LeRobot's decoder, resizes it with Diffusion's own `Resize`, and stores uint8 arrays in the
  variant (11 GB, 97 s on the laptop); `drtc.compat.frame_cache` reads them in training. The model's Resize then
  leaves the frame unchanged. Laptop check (`tests/bench/check_frame_cache.py`): largest difference 0.5/255 (the
  uint8 rounding), mean 0.19/255; DataLoader 1167 → 2714 samples/s. Serving is unchanged.
- [x] Comparison rounds: `tests/bench/variants.yaml` (per model: flags, workers; `parallel` for jobs side by side),
  run with `tests/bench/round.py [variant ...]`; results next to the baseline.
- LoongForge (read 2026-11-19: homepage, π0.5 tutorial, README; v0.1.0, 2026-05):
  - Models: π0.5, GR00T N1.6 and N1.7, X-VLA and others. Not SmolVLA, MolmoAct 2, GR00T N1.5, ACT or Diffusion.
  - Reported speedups: π0.5 2.80×, GR00T N1.7 1.85× "over mainstream open-source baselines". GPU type, GPU count
    and any single-GPU figure are not stated; the π0.5 tutorial runs 8 GPUs.
  - How: bf16, fused AdamW, torch.compile (max-autotune-no-cudagraphs), activation checkpointing only on SigLIP,
    DDP tuning, FSDP2, optional FP8 communication, per-model I/O work. No breakdown by technique.
  - Recipe match: reads LeRobot v3.0 datasets, quantile (q99) normalization, π-style relative actions
    (`use_aloha_pi_delta_actions`, anchored on the chunk's start state). EMA is not mentioned; LoRA for VLAs is
    unverified. Checkpoints: "no conversion needed" for its own evaluation; loading them in LeRobot 0.7 for our
    server is unverified.
  - Install: a Docker image (`loongforge/loongforge`); our H200 is itself a container, so it would be a source
    install there, with its CUDA/PyTorch needs unstated.
  - Use for us: only π0.5 and GR00T N1.7. First take its single-GPU techniques inside LeRobot (compile, fused
    optimizer, checkpointing only on the vision tower), which our rounds measure. Try LoongForge itself only if
    those two stay the longest runs, or with several GPUs, and only after a check that its π0.5/N1.7 training
    matches our recipe (relative actions, EMA, stats) and that its checkpoints serve in LeRobot.
- H200 round 1 (2026-11-19, commit 73d175b8, FFmpeg installed, torchcodec). Cache checks on the H200: Diffusion
  max 0.5/255, DataLoader 346 -> 1922 samples/s; GR00T N1.5 eagle_pixel_values identical, Eagle preprocessing
  1325 -> 155 ms per batch of 32.

  | Model | Setting | Step (baseline) | Full run (baseline) |
  |---|---|---|---|
  | SmolVLA | compile | **0.205 s** (0.323), peak 4.6 GB (14.7) | **1.1 h** (1.8) |
  | GR00T N1.5 | frame cache | 1.032 s (1.080); preprocessing 0.690 s (0.773) | 2.9 h (3.0) |
  | ACT | with Diffusion and SmolVLA on the GPU | 0.162 s (0.107 alone); GPU 97% | 4.5 h (3.0) |
  | Diffusion | cache, 8 workers / **16 workers** | 0.239 / **0.103 s** (0.317) | 14.2 / **6.2 h** (18.8) |
  | Diffusion | compile (no cache) | 0.244 s | 14.5 h |
  | MolmoAct 2 | gradient checkpointing off | 1.636 s (2.185), peak **124 GB** | 22.7 h (30.3) |
  | MolmoAct 2 | augmentation on the GPU / 16 workers | 2.324 / 2.178 s | no gain |
  | π0.5 | checkpointing off / compile | **out of memory** (>140 GB) at batch 64 | stays 27.2 h |

  - Three jobs on the one GPU (ACT + Diffusion + SmolVLA) reach 97% GPU use; each is slower than alone, together
    faster than one after another.
  - **Correction (2026-11-19, later): the frame caches were not used in these training runs.** lerobot-train
    starts its DataLoader workers with `spawn`, and the cache was a patch of LeRobot's class in the training
    process, which spawned workers do not inherit: they decoded the full-size videos (the checks used forked
    workers, which do). So the "frames" rows above measure decoding, and Diffusion's gain at 16 workers came from
    the workers. Fixed: the dataset's reader is a `CachedDatasetReader`, whose class travels with the pickled
    dataset (checked: a spawned worker returns 224x224 frames). The model input was the same either way.
  - [x] GR00T N1.5 preprocessing profiled (laptop, real batches, `tests/bench/profile_preprocessing.py`): with the
    cache 118 ms per batch of 32 (Eagle encode 36 ms, collate 70 ms: tokenizer and Eagle's image processor), against
    0.69 s inside H200 training without it.
  - [x] Remeasured on the H200 with the fix (2026-11-19, round2): GR00T N1.5 **0.518 s** per step (1.080), **1.4 h**
    (3.0); Diffusion 0.162 s at 8 workers, 0.173 s at 16 (data 2%), 9.6 h (18.8). Open, for when Diffusion
    returns: slower than the 0.103 s measured earlier with 16 decoding workers although it no longer waits for data
    (other jobs on the shared H200 at one of the two times?).
- [x] One setup per active model (2026-11-19), in the recipes' `speed:` section (lerobot-train flags that change only
  the speed; `lab train` refuses one that is also a recipe flag, and prints them separately):
  - SmolVLA: `policy.compile_model: true`.
  - GR00T N1.5: the frame cache (`data: frames`).
  - ACT: none; train it next to another job.
  - Deferred, for when they return: Diffusion `num_workers: 16` with its cache (set); MolmoAct 2 checkpointing off
    only on the H200 (124 GB, more than the A100's 80); π0.5 no gain that fits one H200.

## Phase 5: Evaluation framework (designed 2026-11-20)
Design: [EVALUATION.md](EVALUATION.md) (roles, storage, rules); metrics: [METRICS.md](METRICS.md). Built step by
step, each step tested and added to [WORKFLOW.md](WORKFLOW.md) part 2 so the user can run it alone.
- [x] Old `results/` moved to `results/archive/`; private results store `orangemoon599/drtc-results` created
- [x] 1. `lab run` and every non-official run write to `results/scratch/<date>/` (`lab results` reads only scratch)
- [x] 2. Eval files (`lab/evals/`), `lab eval check|freeze|status|list`; rounds seeded per (seed, round); first eval
       `sort_one_baseline` frozen (ACT, SmolVLA, GR00T N1.5, Diffusion at their final checkpoints)
- [x] 3. Serving the same way on the laptop GPU, A100, H200: all models of an eval at once; the robot computer reaches
       them (A100 directly or tunnelled, H200 through an SSH tunnel to the container); servers report model version and
       GPU
- [x] 4. Logging for metrics v1: `chunks.csv` (turnaround, server inference time, estimate at send, chunk-switch
       disagreement), measured joints in `ticks.csv`
- [x] 5. `lab eval run`: rounds, seeded order, keys `s`/`f`, time limit, resume, `lab eval void`, video, session
       folders, pull before / push after
- [x] 6. `lab eval report`: `trials.csv`, `metrics.csv`, `report.md` (M1-M8, Wilson intervals, Fisher / Mann-Whitney)
- Findings while building (2026-11-20, replay robot, laptop on WiFi -> A100 through the SSH tunnel):
  - **Turnaround ~390-720 ms** (ACT 430, Diffusion 570, GR00T 715) against ~40-200 ms of server time; laptop GPU
    alone: 36 ms. Ping 7 ms and 42 Mbit/s upload, so not bandwidth; a tunnel with a terminal (TCP_NODELAY) gave 389 ms
    against 450. A direct Zenoh connection opens the session but gets no replies (tunnels stay the standard).
    **[open]** where the rest goes.
  - **Fixed: DRTC's cooldown at cold start.** Seeded with s_min + epsilon steps, it gave up every first request on a
    link slower than that, so no reply was ever measured. It now waits up to the startup timeout until the first reply.
  - **Fixed, option b (decided with the user): no H/2 cap on the cooldown.** With a latency above half a chunk
    (GR00T: 16 actions = 0.53 s) DRTC's cap made the cooldown shorter than every reply, so the arm never got actions.
    The cooldown is now never shorter than the measured latency; `cooldown_cap: half_chunk` keeps the original (Phase 8).
  - **Stalls hold the arm** (`action_starvation_grace_s` = the trial length) instead of upstream's terminal fault
    after 1 s, as DRTC does; stalls are measured (M2, M3) from the gaps between executed actions, since the control
    loop does not ask for actions while it holds.
  - **Fixed: "Invalid RTC delay"** (user's first official session, 12:29): upstream estimates the request delay as
    turnaround / step time, unbounded, and the server rejects a delay longer than the chunk; GR00T (16 steps) ended
    every trial whose turnaround passed 0.53 s. The delay is now capped at the server's prediction steps.
  - **Fixed: cooldown request storm** (same session, 12:34): with replies at 0.52-0.59 s, the cooldown from the
    smoothed estimate (0.6-0.69 s) gave up slow replies; the server, still computing them, delayed the next request
    further, until no reply got through and the start-up deadline ended the trial. The cooldown now also covers the
    slowest of the last 10 measured replies (option b, literally: never shorter than the measured latency).
  - GR00T over this link: 74% of the trial stalled (replay robot); Diffusion 46%. Their short chunks cannot cover
    the round trip: the central DRTC problem, now measured.
- [ ] 7. `lab eval offline <dataset>` (M9) and the benchmark (M10), on a GPU server, pushed to the results store
- [ ] 8. End-to-end test: replay robot against each server, then one short real trial; WORKFLOW.md part 2 complete
- [ ] Later: the Raspberry Pi as robot computer (edge eval)

## Phase 6: Train and choose
- [x] Trained (fixed recipes): round 1 `act_cubebox_v0_full`, `smolvla_cubebox_v0_full`, `groot_cubebox_v1_full`;
      round 2 `act/smolvla/groot_sort_one_v1_full`; `diffusion_sort_one_v1_full` training (2026-11-20)
- [ ] M9 offline table for every trained model
- [ ] Eval `sort_one_baseline`: ACT, SmolVLA, GR00T N1.5, Diffusion, 20 trials each (~1 h)
- [ ] Optional eval `cubebox_baseline`: round 1 models against the kept minor models (GR00T 80%, SmolVLA 90%)
- [x] `sort_one_baseline_v2` done (2026-11-20, 80/80, laptop -> A100): SmolVLA 16/20, Diffusion 16/20, ACT 15/20,
      GR00T 4/20 (stalls 58%: its 0.5 s chunks cannot cover the ~0.5 s round trip). `sort_one_baseline` (first 5
      rounds, before the GR00T fixes) is superseded.
- [x] Chosen for Phases 7-8 (the user, 2026-11-20): **SmolVLA, Diffusion and GR00T N1.5**. GR00T against the top-2
      rule on purpose: it is the model the network hurts most, so improving it is the strongest DRTC result.

## Phase 7: Smoothing
- [ ] Bench: the replay robot plays held-out episodes to the served model with simulated latency (normal and
      spikes); metrics v1 M2-M8 per run, in `results/bench/`; one command per method x model
- [ ] Baselines: no smoothing; upstream alignment and blending; upstream RTC; the Butterworth filter (current default)
- [ ] Methods ported from the minor: A adaptive playback, B RTC for GR00T, C smoothness guidance; new ones later
- [ ] Robot eval `sort_one_smoothing`: the best method(s) against the baseline on the 2 chosen models, normal network
      and latency spikes, 20 trials each (M1, M1t, M7, M8)

## Phase 8: DRTC experiments
- [ ] Bench: all fault conditions (`lab/experiments/replication/`) on the replay robot, the 2 chosen models (M2-M8)
- [ ] Robot eval `sort_one_faults`: 4 headline conditions (spikes, drops, disconnect, mixture) + baseline, 20 trials
      each (M1-M6) (decision B)
- [ ] Latency estimators (Jacobson-Karels, max of last 10, others), offline on the logged turnarounds (M5)
- [ ] DRTC against upstream remote inference: same model, normal network and spikes
- [ ] **[open]** Further items from the proposal (predictive triggering, adaptive chunk length, adaptive steps,
      caching, shared server), as time allows (presentation date)
- [ ] Edge eval `sort_one_edge`: the Raspberry Pi as robot computer (EVALUATION.md)
