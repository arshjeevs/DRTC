# Training

> The whole path from recording to training, in order: [WORKFLOW.md](../WORKFLOW.md).

Training runs on the GPU servers, the **H200** and the **A100**; you start and watch the runs (logins, updating the
code and the rules on both servers: [SERVERS.md](SERVERS.md)). Recipes are fixed (`lab/recipes/`, each policy's
original authors' recipe); a model is a recipe applied to a dataset.

## 1. Before training: dataset and model entries

1. The dataset is pushed and `lab/datasets/recorded.yaml` is committed ([DATA.md](DATA.md)).
2. The models are in `lab/models/<dataset>.yaml`, e.g. `lab/models/sort_one_v1.yaml`:

   ```yaml
   act_sort_one_v1_full: {recipe: act_original, dataset: so101_sort_one_v1, status: planned,
     notes: 'ACT on so101_sort_one_v1; original recipe'}
   ```

   Name: `<policy>_<task>_<version>_full` (`_lora` for MolmoAct 2). The seven recipes:

   | Policy | Recipe | Length | Time | Notes |
   |---|---|---|---|---|
   | ACT | `act_original` | 100k steps x 8 | ~3 h (H200 alone) | no language input; the edge fallback |
   | SmolVLA | `smolvla_authors` | 20k steps x 64 | ~1.1 h (H200 alone) | compiled; the VLM is frozen |
   | GR00T N1.5 | `groot_n15_nvidia` | 10k steps x 32 | ~1.4 h (H200 alone) | frame cache (prepared on the server) |
   | GR00T N1.7 | `groot_n17_nvidia` | 20k steps x 64 | several hours | relative arm actions (`lab prepare` makes their stats) |
   | π0.5 | `pi05_openpi` | 20k steps x 64 | ~27 h, ~77 GB | the H200 only; relative joints |
   | MolmoAct 2 | `molmoact2_lora` | 50k steps x 16 | ~5 days on the A100 | LoRA rank 64 on the VLM |
   | Diffusion | `diffusion_original` | 600 epochs x 64 | a few hours | ResNet-18 from scratch |
3. Commit and push; check on the laptop that the command builds (it downloads nothing):

   ```bash
   lab check
   lab train act_sort_one_v1_full --dry-run    # prints the dataset revision, the split, the full command
   ```

## 2. On the H200 (or the A100)

In the `drtc_lab` container ([SERVERS.md](SERVERS.md) has how to get in):

```bash
cd /workspace/drtc_major && . ./.server.env && git pull && uv sync --frozen --extra gpu --extra sim --python /usr/bin/python3
.venv/bin/lab check --logging            # once per setup: wandb and HF keys present
.venv/bin/lab prepare so101_sort_one_v1  # once per new dataset revision: download + prepare for the recipes
nvidia-smi                               # check that the GPU has room (other students share it)
```

Run `lab prepare` before starting several runs on the same dataset. Otherwise each run prepares the same copy at
the same time.

**One run:**

```bash
tmux new -s train '.venv/bin/lab train smolvla_sort_one_v1_full; bash'
```

**Several at once** (they share the GPU; start ACT first: the CPU limits from `lab/hosts.yaml` apply to runs
started while another one is already training, and ACT needs the least CPU):

```bash
tmux new -d -s sort -n act     '.venv/bin/lab train act_sort_one_v1_full; bash'
sleep 60
tmux new-window -t sort -n groot   '.venv/bin/lab train groot_sort_one_v1_full; bash'
tmux new-window -t sort -n smolvla '.venv/bin/lab train smolvla_sort_one_v1_full; bash'
tmux attach -t sort        # Ctrl-b n: next window; Ctrl-b d: detach (the runs go on)
```

Back later: `tmux attach -t sort`.

**On the A100** the same `lab` commands run on the host, no container. Update with `git pull` and, only after a
change of `uv.lock`, `lab/bin/a100-setup` (**never a plain `uv sync`** there), and source the environment before
every run:

```bash
ssh b230968ee@192.168.5.13
cd ~/drtc_major && . ./.server.env && git pull
.venv/bin/lab prepare so101_sort_one_v1
nvidia-smi && df -h ~                    # shared GPU; /home is nearly full
tmux new -d -s train -n act '.venv/bin/lab train act_sort_one_v1_full; bash'
```

## 3. Watching

- **wandb**, project `drtc-lab`: training loss, learning rate and the held-out loss, live from any machine.
- The tmux window: LeRobot's progress bar with the time left, plus a log line every few hundred steps.
- Held-out loss is measured about 20 times per run. Lower is better, but for flow models (SmolVLA, GR00T) it can
  rise while the model still improves. The offline action error and the robot trials decide.

## 4. Stopping, resuming

```bash
.venv/bin/lab stop act_sort_one_v1_full    # clean stop: checkpoint at the next step, push it, exit
.venv/bin/lab stop --all                   # every lab train on this machine
```

Ctrl-C in the window does the same. Continue later, here or on another server:

```bash
.venv/bin/lab train act_sort_one_v1_full --resume
```

`lab train` refuses to start when `outputs/train/<model>` already exists. Use `--resume`, or ask Claude before
deleting that folder.

## 5. The result


At the end each run pushes to `orangemoon599/<model>` on the Hub:

- the repo root holds the **best checkpoint by held-out loss**, the one `lab serve <model>` uses;
- `checkpoints/<step>/` holds the last checkpoint with its training state, so the run can be resumed.

On the server: `outputs/train/<model>/checkpoints/` keeps the latest two checkpoints and the best one (older ones
are deleted to save disk). `eval_losses.json` there lists the held-out loss per checkpoint.

To push a different checkpoint later:

```bash
.venv/bin/lab train act_sort_one_v1_full --push 60000
```

After a model is on the Hub: its offline evaluation, then it joins its task's baseline and bench evals (they grow:
[WORKFLOW, Growing an eval](../WORKFLOW.md#growing-an-eval-more-trials-more-models)).

## Other options

| Command | Use |
|---|---|
| `lab train <model> --steps 200 --no-push` | a short test run, kept local |
| `lab train <model> --no-wandb` | Trackio only (wandb is the standard; only on purpose) |
| `lab submit <model> --host h200` / `lab worker` / `lab jobs` | a job queue: the laptop queues (over SSH), the worker on the server starts jobs when the GPU has room (`--max-jobs 3` for three at once) |
| `lab list` | every model, recipe and dataset in the registry |

## When something goes wrong

| Message | What to do |
|---|---|
| `outputs/train/<model> exists; continue it with --resume` | add `--resume`, or ask Claude before deleting it |
| a stale local copy of the dataset ("another version under the same name") | `git pull` (the registry pins the newest revision); ask Claude before moving the folder |
| wandb missing / no key | `lab check --logging`; the keys are in `/workspace/drtc_lab/.secrets.env` (H200) or `~/.drtc.env` (A100) |
| A100: `The NVIDIA driver on your system is too old` | someone ran a plain `uv sync`: run `lab/bin/a100-setup`, then start again |
| CUDA out of memory | another job is using the GPU: `nvidia-smi`, start fewer runs at once |
| anything else | paste the last lines of the window to Claude |
