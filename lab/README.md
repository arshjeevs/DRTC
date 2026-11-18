# Lab

Our layer on top of LeRobot 0.7 (pinned in `pyproject.toml`). Plan, decisions and progress: [PLAN.md](PLAN.md).
H200 server: [H200.md](H200.md). The minor's setup is tag `minor-final` (branch `drtc`, `~/Projects/drtc`).

```
lab/
├── lab.yaml          Hub account, Trackio project/Space
├── hosts.yaml        machines: roles, GPUs, SSH/container, hostname; network (Zenoh router, old gRPC server)
├── rigs/             physical setups, versioned (cameras with roles, arms with ports per host, calibration files,
│                     joint units for new recordings)
├── datasets/         datasets pinned to a Hub revision + episodes + frames; joint units; parents and derivation
├── recipes/          how each policy is trained (and where the recipe comes from)
├── models/           models = recipe x dataset (+ overrides); status
├── experiments/      robot-run conditions (_defaults.yaml < rig < model < experiment)
├── renames.yaml      the clean start of 2026-11-18 (old -> new Hub names)
└── bin/              old command names, now aliases for `lab ...`
src/drtc/             the `lab` command, registry, training, jobs, recording/trimming, robot control
plugins/              LeRobot plugins: lerobot_policy_groot_n15 (GR00T N1.5), lerobot_robot_replay (replay robot)
```

Install: `uv sync --extra gpu --extra sim` (laptop / GPU servers), `uv sync` (Pi). Then `.venv/bin/lab ...`.

| Command | Does |
|---|---|
| `lab list`, `lab check [--rig] [--data [--online]]` | registry; devices and calibration; dataset pins |
| `lab record <name> --task "..."` | record `<name>_raw` with the leader arm, check, trim into `<name>`, push both, registry entries |
| `lab trim <raw dir> <out dir>` | trim the idle start/end of every episode into a new dataset |
| `lab train <model> [--steps N] [--resume] [--push CKPT]` | train a registry model with LeRobot 0.7 (clean stop: Ctrl-C or `lab stop <model>`) |
| `lab submit <model> --host h200`, `lab worker`, `lab jobs` | queue jobs pinned to a pushed commit; run them when the GPU has room |
| `lab serve <model>` | serve a model with LeRobot 0.7's policy server (deployment from the registry) |
| `lab run <experiment> [--model M] [--robot rig\|replay]` | one robot run (drtc engine: DRTC scheduling, cooldown re-request, faults, filter) |
| `lab results [--experiment E]` | stalls, chunks, turnaround, faults, recoveries per run |
| `lab calibrate follower\|leader`, `lab rig <key>` | calibration with the rig's explicit file; rig values for scripts |
| `lab import-results <old results dir>` | copy the minor's results in, with the new model names |

Results: `results/<experiment>/<model>/<run_id>/` (config.yaml, ticks.csv, faults.csv, recovery.csv, run.json,
rollout.log, robot_commands.jsonl) and `results/index.csv`. The laptop keeps the phone camera bridge
(`lab/bin/phonecam`) and the tmux session (`lab/bin/session`) as before.
