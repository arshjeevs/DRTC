# Evaluation: how results are made and stored

Agreed with the user on 2026-11-20, updated 2026-11-22 for setup v2 (evals that grow, declared comparisons, metrics
v3). Metrics: [METRICS.md](../METRICS.md) (v3). Plan: [PLAN.md](../PLAN.md), "DRTC + SMOOTHING EXPERIMENT SETUP v2".
The step-by-step commands for the user: [WORKFLOW.md](../WORKFLOW.md) ("Setup v2", "Growing an eval", part 2).

## Roles

| Role | Machines | Stores |
|---|---|---|
| **Robot computer** (runs `lab eval run`, drives the arm) | laptop (usual) or Raspberry Pi (edge proof) | writes each session locally, then pushes it |
| **Model server** | laptop GPU, A100, H200 (chosen per session) | nothing for analysis: its timings and identity come back in every reply |
| **Results store**, the one official copy | Hub dataset repo `orangemoon599/drtc-results` (private) | everything, from every machine |
| **Analysis** (reports; Claude reads here) | any machine, normally the laptop | pulls from the results store |

```
 model server (A100 / H200 / laptop GPU)  — stores nothing for analysis
        ▲ images + state      │ actions + server timings + model version
        │                     ▼
 robot computer: laptop  OR  Raspberry Pi
   lab eval run → results/evals/<eval>/sessions/<date>-<machine>/   (written locally)
        │ end of session: push              ▲ start of session: pull (which trials are done)
        ▼                                   │
 Hub: orangemoon599/drtc-results   ← the one official copy (offline M9/M10 from the GPU servers too)
        │ pull
        ▼
 analysis (any machine): lab eval report <eval>
```

A model server keeps only its config (`outputs/serve/`) and log (`logs/serve/`) for debugging; nothing reads them for
results.

## Layout

```
results/
├── evals/<eval>/                     one study on the robot (e.g. sort_one_baseline, sort_one_drtc_v2)
│   ├── sessions/<YYYYMMDD-HHMM>-<machine>/          one sitting at the robot, self-contained
│   │   ├── session.json              robot computer, server + GPU, git commit, rig check, model versions served
│   │   └── trials/<condition>/<model>/t01 … t20/    one folder per trial, always these files:
│   │       ├── trial.json            identity, round, label, operator key + time (M1, M1t), void flag + reason
│   │       ├── config.yaml           the exact settings used, incl. the trial's fault shift (M11)
│   │       ├── ticks.csv             every tick: queue, stall, latency estimate, commanded + measured joints,
│   │       │                         the model's own action for the step (plan.*, M12; from 2026-11-22)
│   │       ├── chunks.csv            every chunk: request sent / received, turnaround, server inference time,
│   │       │                         latency estimate at send, chunk-switch disagreement
│   │       ├── faults.csv, recovery.csv   injected faults, DRTC recoveries
│   │       ├── rollout.log
│   │       └── video/wrist.mp4, video/scene.mp4
│   ├── trials.csv                    rebuilt from all sessions: one row per trial, with its session and folder
│   ├── metrics.csv                   rebuilt: the current metrics (v3), one row per trial
│   └── report.md                     rebuilt: the tables for the presentation
├── bench/<bench>/                    the bench on the replay robot (e.g. bench_v2_drtc_sort_one): the same structure
├── offline/<dataset>/<model>.json    M9 and M10 (computed on a GPU server, pushed to the results store)
├── scratch/<YYYYMMDD>/...            every run not started by `lab eval run`: never pushed, never read by reports
│                                     or by Claude, safe to delete
└── archive/                          results from before the locked scope of 2026-11-20 (side results)
```

## Rules

1. **The session is the unit.** A session folder belongs to one machine and one sitting, and no file is written by
   two machines. Results from the laptop and the Pi combine by putting session folders side by side; the eval-wide
   `trials.csv`, `metrics.csv` and `report.md` are always rebuilt from all sessions present.
2. **Raw files are never edited.** `trial.json`, `ticks.csv`, `chunks.csv` and the videos are written once, during
   the trial. Voiding a trial adds a flag and a reason to its `trial.json`; nothing is deleted. A metrics change
   never rewrites a trial either: e.g. the `fault` label of v3 is worked out from the run's own record (outcome and
   first action) when the report is built.
3. **Derived files are always rebuilt** by `lab eval report <eval>`. A metrics change (v2, v3) is one rerun.
4. **The plan is frozen in git.** `lab eval freeze <eval>` writes the exact model versions (Hub commits) into
   `lab/evals/<eval>.yaml`; it is committed. Every robot computer reads the same plan. An official trial refuses to
   start with a changed plan, a dirty git tree, a server serving another model version, or a failed rig check.
   **A plan with trials only grows** (2026-11-22): `trials` may go up and models may be added (`lab eval freeze`
   then freezes only the added ones; the frozen ones never change). Everything else (conditions, task, success rule,
   time limit, seed, robot, a model's version) needs a new eval. Trials run in rounds, so stopping at the end of a
   round keeps every condition at the same count.
5. **Pull before, push after.** `lab eval run` pulls the eval's sessions from the results store first (so it knows
   which trials are done, whichever machine did them) and pushes the new session at the end. Without internet the
   session stays local, marked not pushed; `lab eval status` lists it and `lab eval push` sends it later.
6. **Test runs cannot leak in.** Only `lab eval run` writes to `results/evals/`. `lab run` and everything else write
   to `results/scratch/`.
7. **Git holds definitions, not data:** `lab/evals/*.yaml`, `lab/METRICS.md`, this file. `results/` stays
   git-ignored.
8. **One eval per study.** Each study gets its own eval (the baselines, E4 on the robot, each bench of setup v2); the
   condition is part of every trial's path, so comparisons inside a study are like with like.
9. **The eval names its comparisons** (metrics v3): `comparisons:` lists the pairs the report shows, `primary:` the
   few that are tested (Holm-corrected); all others are exploratory (METRICS.md rules 5 and 7).

## Eval file

`lab/evals/<eval>.yaml`, written before the first trial:

```yaml
description: DRTC v2, sort_one - upstream patient, + mix, DRTC + mix, DRTC + stack + mix
success: The cube is fully inside the square of its colour and the gripper has let go of it, before the time limit.
time_limit_s: 30
trials: 10                            # valid trials per model per condition (METRICS.md rule 1); may grow
seed: 0                               # the order inside each round
models: [smolvla_sort_one_v1_full, groot_sort_one_v1_full]   # the task comes from their dataset; may grow
conditions:                           # name -> settings in lab/experiments (v2_robot/ is generated by v2/_make.py)
  upstream_patient__none: v2_robot/upstream_patient__none
  upstream_patient__mix: v2_robot/upstream_patient__mix
  drtc__mix: v2_robot/drtc__mix
  drtc_stack__mix: v2_robot/drtc_stack__mix
comparisons:                          # what the report compares: "<condition> vs <reference>"
  - upstream_patient__mix vs upstream_patient__none
  - drtc__mix vs upstream_patient__mix
  - drtc_stack__mix vs drtc__mix
primary: [...]                        # the tested ones (here all three)
video: true
robot: rig                            # rig (the arm) or replay (the bench: M2-M8, M11, M12)
frozen: {}                            # written by lab eval freeze: model -> Hub commit
```

The setup v2 evals are written by `lab/experiments/v2/_make.py --evals` (and `--stack ...` once the stack is chosen),
so their conditions and comparisons are the same everywhere.

## During a trial

| Key | Means |
|---|---|
| Enter | start the trial (after setting the layout of the round) |
| `s` | success: the task is done now (ends the trial; M1 = 1, M1t = now) |
| `f` | failure: it will not succeed (ends the trial early; M1 = 0) |
| time limit | failure (M1 = 0) |
| (the system ends the run) | `fault`: failure, counted in M1f, whatever key is pressed afterwards (the key is kept as `key`); before the first action: not counted, run again |

## The edge proof (Raspberry Pi)

Set up and tested 2026-11-20 (`lab/bin/pi-setup`, WORKFLOW part 2); official trials stay on the laptop. Its own eval (e.g. `sort_one_edge`): the same chosen model and server,
with the laptop and then the Pi as the robot computer. Each trial records its robot computer, and reports never
average across them. The Pi 5 has no hardware video encoder: video there is measured first and recorded at reduced
quality if it costs control time (the setting is in `trial.json`). Needs: the environment on the Pi, the Pi's arm
ports in the rig, its own tunnels to the A100 (SSH key) and H200.
