# Lab

Our layer on top of LeRobot 0.7 (pinned in `pyproject.toml`). The minor's setup is tag `minor-final` (branch `drtc`,
`~/Projects/drtc`).

**Start here** (the four files in this folder):

| File | What |
|---|---|
| [PLAN.md](PLAN.md) | the locked scope and the order of work, with status |
| [WORKFLOW.md](WORKFLOW.md) | the whole path with every command: dataset → trained models → evaluation |
| [METRICS.md](METRICS.md) | the fixed metrics (v2) every result uses |
| README.md | this page |

**Details** in [docs/](docs/): [EVALUATION.md](docs/EVALUATION.md) (how results are made and stored),
[DATA.md](docs/DATA.md) (recording and editing datasets), [TRAIN.md](docs/TRAIN.md) (training),
[H200.md](docs/H200.md) and [WINDOWS.md](docs/WINDOWS.md) (machine notes), [PROPOSAL.md](docs/PROPOSAL.md) (the
research proposal), [history/](docs/history/) (the plan of 2026-11-18 to 20).

```
lab/
├── PLAN.md, WORKFLOW.md, METRICS.md, README.md
├── docs/             the detail guides, machine notes, the proposal; docs/history/: the earlier plan
├── lab.yaml          Hub account, Trackio project, wandb, the results store
├── hosts.yaml        machines: roles, GPUs, SSH/container, hostname
├── rigs/             physical setups, versioned (cameras with roles, arms with ports per host, calibration files)
├── datasets/         datasets pinned to a Hub revision + episodes + frames; joint units; parents and derivation
├── recipes/          how each policy is trained (fixed: the original recipes)
├── models/           models = recipe x dataset (+ overrides); status
├── experiments/      conditions (_defaults.yaml < model < experiment): main/ (the DRTC experiment's four
│                     conditions), bench_main/ (the same on the replay robot), archive/ (before 2026-11-20)
├── evals/            eval plans (frozen before the first trial); archive/ (before 2026-11-20)
├── renames.yaml      the clean start of 2026-11-18 (old -> new Hub names)
└── bin/              session, phonecam, preflight, calibrate, server setup scripts; legacy/ (old command names)
src/drtc/             the `lab` command: registry, training, recording/trimming, robot control, evaluation
plugins/              LeRobot plugins: lerobot_policy_groot_n15 (GR00T N1.5), lerobot_robot_replay (replay robot)
```

Install: `uv sync --extra gpu --extra sim` (laptop; the H200 container adds `--frozen --python /usr/bin/python3`), **A100: `lab/bin/a100-setup` only** (a plain `uv sync` there installs CUDA 13 builds its driver cannot run), `uv sync` (Pi). Then `.venv/bin/lab ...`.

| Command | Does |
|---|---|
| `lab list`, `lab check [--rig] [--data [--online]]` | registry; devices and calibration; dataset pins |
| `lab record <name> --task "..."` | record `<name>_raw` with the leader arm, check, trim into `<name>`, push both, registry entries |
| `lab trim <raw dir> <out dir>` | trim the idle start/end of every episode into a new dataset |
| `lab train <model> [--steps N] [--resume] [--push CKPT]` | train a registry model with LeRobot 0.7 (clean stop: Ctrl-C or `lab stop <model>`) |
| `lab submit <model> --host h200`, `lab worker`, `lab jobs` | queue jobs pinned to a pushed commit; run them when the GPU has room |
| `lab serve <model>` | serve a model with LeRobot 0.7's policy server (deployment from the registry) |
| `lab eval list\|check\|freeze\|serve\|servers\|run\|void\|status\|push\|pull\|report\|offline` | official evaluations ([WORKFLOW.md](WORKFLOW.md) part 2, [docs/EVALUATION.md](docs/EVALUATION.md)) |
| `lab run <experiment> [--model M] [--robot rig\|replay]` | one robot run (drtc engine: DRTC scheduling, cooldown re-request, faults, filter) |
| `lab results [--experiment E]` | stalls, chunks, turnaround, faults, recoveries per run |
| `lab calibrate follower\|leader`, `lab rig <key>` | calibration with the rig's explicit file; rig values for scripts |
| `lab import-results <old results dir>` | copy the minor's results in, with the new model names |

Results: `results/<experiment>/<model>/<run_id>/` (config.yaml, ticks.csv, faults.csv, recovery.csv, run.json,
rollout.log, robot_commands.jsonl) and `results/index.csv`. The laptop keeps the phone camera bridge
(`lab/bin/phonecam`) and the tmux session (`lab/bin/session`) as before.
