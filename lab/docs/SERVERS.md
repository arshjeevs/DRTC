# The GPU servers: A100 and H200

*Written 2026-11-22 for the major setup (branch `major`, LeRobot 0.7). It replaces the minor's H200 notes
([history/H200-minor-setup.md](history/H200-minor-setup.md)). What runs on them, step by step:
[../WORKFLOW.md](../WORKFLOW.md) (training: Part 1; serving, bench and offline: Setup v2, Part 2, Part 3).*

Both servers are **shared** with other people and borrowed. The rules below come first.

## At a glance

| | A100 | H200 |
|---|---|---|
| GPU | A100 80 GB PCIe ("GPUserver") | H200 NVL 141 GB ("gpuh200"; may be MIG-sliced) |
| Account | `b230968ee` (ours) | `jayaraj` (the professor's, shared) |
| Log in | `ssh b230968ee@192.168.5.13` (key login from the laptop) | Jupyter terminal, or `ssh jayaraj@192.168.41.118` (password), then `docker exec -it drtc_lab bash` |
| Where the code is | `~/drtc_major` on the host | `/workspace/drtc_major` inside the `drtc_lab` container (= `/home/jayaraj/drtc_major` on the host) |
| Environment | `cd ~/drtc_major && . ./.server.env` | `cd /workspace/drtc_major && . ./.server.env` (inside the container) |
| Driver / CUDA | 570, CUDA <= 12.8: torch 2.14.1+**cu126** | 580, CUDA <= 13.0: torch 2.14.1+**cu130** (the lock) |
| Secrets (HF, wandb) | `~/.drtc.env` | `/workspace/drtc_lab/.secrets.env` (read by `lab`; nobody opens it) |
| Who runs commands | you; Claude may too (key login) | you; Claude only inside `drtc_major`, through your own SSH connection |
| Used for | training, serving for robot sessions, bench, offline | training, serving, bench, offline |

`lab/hosts.yaml` holds the same facts for the tools (`ssh`, `container`, `repo`, `secrets`, `cpu_budget`, `gpus`).

## Rules (both servers)

1. **Work only in the repo folder** (`~/drtc_major`, `/workspace/drtc_major`). Never write, change or delete
   anything outside it.
2. **Never delete without asking**, not even old caches. On the H200 nothing is ever deleted outside `drtc_major`.
3. **Never change the server**: no driver or CUDA upgrades, no system packages, no changes to other users' setups.
   The code adapts to the server, not the other way round.
4. **Other people's processes** are never touched. Stop only your own (`tmux`, `lab stop`).
5. **Look first**: `nvidia-smi` (who uses the GPU, how much memory is free) and `df -h ~` (disk) before a long job.
6. **Long jobs in tmux** so they survive a logout: `tmux new -s <name>`, Ctrl-b d to leave it running,
   `tmux attach -t <name>` to look, `tmux ls` to list.

## A100

**Update the code:**

```bash
ssh b230968ee@192.168.5.13
cd ~/drtc_major && . ./.server.env && git pull
lab/bin/a100-setup                      # only after a change of uv.lock (it says when nothing changed)
```

**Never a plain `uv sync` there.** The lock pins CUDA 13 builds (for the H200); on the A100's driver every GPU
process then fails with "The NVIDIA driver on your system is too old" (it happened on 2026-11-20 and stopped a
training). `lab/bin/a100-setup` installs the same packages with the +cu126 torch builds. Start every job with
`. ./.server.env` sourced in the same shell (it also sets the FFmpeg library path).

Good to know: the CPUs (24) are shared, so trainings use `cpu_budget` from hosts.yaml; Hugging Face downloads are
fast, GitHub clones slow. Our long jobs run in the tmux session `drtc` (one window per job).

## H200

**Get in and update the code** (git runs **inside** the container: the tracked files are owned by root, so a
`git pull` on the host stops halfway with "Permission denied"):

```bash
ssh jayaraj@192.168.41.118                       # or the Jupyter terminal
docker exec -it drtc_lab bash                    # the container must be running (below)
cd /workspace/drtc_major && . ./.server.env && git pull
uv sync --frozen --extra gpu --extra sim --python /usr/bin/python3    # only after a change of uv.lock
```

**The container.** The server guide ("Tensorflow & PyTorch in H200 GPU Server") prescribes the image
`nvcr.io/nvidia/pytorch:26.02-py3`, a free GPU device from `nvidia-smi -L`, `/home/<user>` mounted as `/workspace` and
Jupyter. `lab/bin/h200-start` starts it that way (plus `--shm-size` for data loaders and `--name drtc_lab`); run it in
a tmux session on the host so the container outlives your login:

```bash
tmux new -s drtc_lab
~/drtc_major/lab/bin/h200-start       # says so if drtc_lab is already running
```

Files the container makes belong to root; `lab/bin/h200-own` (inside the container) gives this checkout back to the
account.

**Reaching it from the laptop.** The container is only reachable through the host. `lab eval servers <eval> --server
h200` and `lab eval run <eval> --server h200` open one SSH connection (you type the password once), find the
container's address and forward every model's port.

**Good to know:** `/home` is nearly full and shared, so the new setup shares the old setup's Hugging Face model cache
(it only adds files) instead of keeping a second copy. The server's clock is not set to real time: don't compare
times of H200 runs (wandb, logs) with laptop commit times. Training jobs can also be queued from the laptop
(`lab submit <model> --host h200`, a `lab worker` in a tmux window in the container starts them when the GPU has
room; `lab jobs`, `lab stop <model>`).

## Which server for what

| Job | Where | Notes |
|---|---|---|
| Training | either; π0.5 (~77 GB) only on the H200 | every run logs to wandb (`drtc-lab`) and Trackio; `lab train` refuses without wandb ([TRAIN.md](TRAIN.md)) |
| Serving for robot sessions | either (or the laptop GPU for ACT, Diffusion, SmolVLA) | the whole eval on one server where possible; timing metrics are split per server |
| Bench (`bench_v2_*`) | either, `--server local` | 5 models need ~15 GB; one bench eval per server at a time |
| Offline M9/M10 | either | `lab eval offline --models ...` |

## Setting up a server from scratch

- **A100:** `git clone` the repo (branch `major`) into `~/drtc_major`, put `HF_TOKEN` and `WANDB_API_KEY` in
  `~/.drtc.env`, run `lab/bin/a100-setup`, then `lab check --logging`.
- **H200:** clone into `/home/jayaraj/drtc_major` from inside the container, `uv sync` as above, `lab check
  --logging`. The secrets file is the existing one.
- **A new server:** add it to `lab/hosts.yaml` (ssh, repo, secrets, gpus, cpu_budget), check its driver's CUDA
  version before installing (pick the torch build that it supports, as `a100-setup` does), and pass
  `lab check --logging` before its first training.
