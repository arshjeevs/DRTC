# Lab

Our research layer on top of DRTC/LeRobot. Everything we own lives here; upstream code stays untouched, so `git pull upstream drtc` stays conflict-free.

```
lab/
├── rig.yaml                 hardware: hosts, ports, calibration IDs, cameras (only place they live)
├── models.yaml              model registry: policy type, checkpoint, fps, task, RTC on/off
├── experiments/
│   ├── _defaults.yaml       shared DRTC / RTC / filter settings
│   └── <group>/<name>.yaml  one file = one experimental condition
├── bin/                     entry points
└── tools/                   Python behind the entry points
results/                     (gitignored)
├── index.csv                one row per trial: experiment, success label, git commit, note
└── <group>/<name>/<model>/<run_id>/   metrics CSV/JSON + resolved config.yaml + run.json
```

Config precedence: `_defaults.yaml` < `rig.yaml` < `models.yaml[model]` < experiment file.

On Windows, see [WINDOWS.md](WINDOWS.md): same tools, run as `python -m lab.tools.<cmd>`, with `lab/rig.windows.yaml` selected by `LAB_RIG`.

## Commands

| Command | Where | Purpose |
|---|---|---|
| `lab/bin/server` | laptop | Policy server on the GPU. Loads whichever model each run asks for |
| `lab/bin/session [record]` | Pi / laptop | tmux: `phonecam` window + `work` window running preflight. Re-run to re-attach |
| `lab/bin/run <group/name> [--model M] [--trials N] [--note TEXT] [--no-label]` | Pi | Run a condition. Prompts for scene reset before and success after each trial |
| `lab/bin/train <model> [--dry-run] [--no-push] [--resume] [--push [ckpt]] [--<lerobot flag>=value ...]` | laptop | Train a `models.yaml` entry that has a `train:` block and push it to its Hub id. See [Training](#training) |
| `lab/bin/list` | any | Registered models and experiments |
| `lab/bin/record <user>/<dataset> --task "..." [options] [--<lerobot flag>=value ...]` | laptop | Record demos with the leader arm. Rerun viewer opens when a display exists. See [Recording flags](#recording-flags) |
| `lab/bin/calibrate follower\|leader` | where the arms are | Calibrate an arm with `rig.yaml`'s port and id. Backs up the old file. See [Recalibrating](#recalibrating) |
| `lab/bin/preflight [record]` | Pi / laptop | `run` (default): follower, cameras, server, clock. `record`: both arms, cameras |
| `lab/bin/phonecam` | Pi / laptop | DroidCam → `/dev/video10` bridge (started by `session`) |
| `lab/bin/pi-setup` | Pi, once | Packages, `/dev/video10` at boot, clock synced to laptop, venv |
| `lab/bin/laptop-setup` | laptop, once | Time server for the Pi, firewall, phone-camera packages, `/dev/video10` at boot, calibration links |

## Session

1. Laptop plugged in, hotspot on. Robot powered. Phone in the Pi's USB with DroidCam open.
2. Laptop: `git pull && lab/bin/server`
3. Pi: `git pull && lab/bin/session`. All preflight lines should say `ok`.
4. Pi, in the `work` window: `lab/bin/run smoke/jack_baseline --trials 3`

tmux keys: `Ctrl+b n`/`p` switches windows, `Ctrl+b d` detaches.

## Recording session (laptop)

Record on the laptop: you get the Rerun viewer, and the dataset lands where training runs.

1. Plug both arms, the Brio and the phone (DroidCam open) into the laptop. Robot powered.
2. `lab/bin/session record`. All preflight lines should say `ok`.
3. In the `work` window:
   ```bash
   lab/bin/record <hf_user>/<dataset> --task "<exact sentence>" --episodes 2 --no-push   # dry run
   rm -rf ~/.cache/huggingface/lerobot/<hf_user>/<dataset>                              # discard it
   lab/bin/record <hf_user>/<dataset> --task "<exact sentence>" --episodes 50
   ```
   If interrupted, continue with the same command plus `--resume`.

Controls, typed **in the terminal** running `record`, not in the Rerun window (click back on the terminal after Rerun opens):

| Key | Action |
|---|---|
| `→` or `n` | Episode done: save it and start the reset early. During the reset: start the next episode |
| `←` or `r` | Discard this episode and record it again |
| `Esc` or `q` | Stop: save the finished episodes and exit |

These replace LeRobot's own arrow-key listener, which misses keys on Wayland and over SSH. They also work over SSH on the Pi.

`rig.yaml` uses `by-id` device paths, so the same file works on both machines. Only the USB cables move.

### Recording flags

`lab/bin/record` builds a `lerobot-record` command from `rig.yaml`. Shortcuts: `--episodes` (50), `--episode-time` (30 s), `--reset-time` (10 s), `--resume`, `--no-push`, `--display/--no-display`, `--dry-run` (print the command, don't run).

Any other `lerobot-record` flag is passed through and **overrides** the generated value for the same key (`.venv/bin/lerobot-record --help` lists them all):
```bash
lab/bin/record arshjeevs/cube_v1 --task "..." --dry-run \
    --dataset.episode_time_s=45 \
    --robot.max_relative_target=10 \
    --dataset.private=true \
    --dataset.tags='[so101,cube]'
```
Always `--dry-run` first after changing flags. Overriding `--robot.cameras` replaces both cameras, so keep the names `camera1` and `camera2`.

## Training

A `models.yaml` entry with a `train:` block covers the whole lifecycle: `lab/bin/train` trains it and pushes the result to `pretrained_name_or_path`, and the policy server later loads it from that same field.

```bash
.venv/bin/hf auth login                                     # once, with a WRITE token
lab/bin/train smolvla_test_run_1 --dry-run                  # check the command
lab/bin/train smolvla_test_run_1 --no-push --steps=50 --output_dir=outputs/train/smoke   # ~5 min: GPU memory and speed
rm -rf outputs/train/smoke
lab/bin/train smolvla_test_run_1                            # full run, pushes to the Hub at the end
```
- Checkpoints go to `outputs/train/<model>/checkpoints/` every `save_freq` steps, and logs to `logs/train_<model>_*.log`.
- Interrupted? Run `lab/bin/train <model> --resume`.
- Finished, and want more steps? `--resume` only runs up to the run's saved target, so raise it: `lab/bin/train <model> --resume --steps=30000`. The learning-rate schedule is rebuilt for the new total, so the LR rises again partway through the cosine curve.
- Every run logs to wandb (project `drtc-smolvla`; checkpoints are not uploaded to wandb). Log in once with `.venv/bin/wandb login`, or pass `--wandb.enable=false`.
- Stopped early on purpose (Ctrl+C) and want to use what you have? `lab/bin/train <model> --push` uploads the latest checkpoint (or `--push 005000` for a specific one) to the model's Hub id. Training more later with `--resume` and pushing again replaces it.
- Out of GPU memory? Add `--batch_size=4`.
- Any `lerobot-train` flag passes through and overrides the generated one.
- Flags a model always needs go in its entry: `train.flags: {policy.dtype: bfloat16}`. Command-line flags still override them.

### pi0 and pi05: separate venv (automatic)

pi0 and pi05 need a patched `transformers` build that conflicts with the one SmolVLA and X-VLA use. They run from a second venv, `.venv-pi`, and the tools choose it for you from the model's `policy_type`:

```bash
lab/bin/train pi05_train_new_1_lora      # uses .venv-pi; creates it on first use (several minutes)
lab/bin/server pi05_train_new_1_lora     # serve pi0 / pi05 models (any pi0 / pi05 entry name works)
lab/bin/server                           # serve smolvla / act models from the main .venv
```
- `lab/bin/venv pi` creates or updates `.venv-pi` by hand. The CUDA build of PyTorch is detected from the GPU driver; override with `TORCH_BACKEND=cu128`.
- One server process serves one venv's policies. If a run asks a main-venv server for a pi0/pi05 model, the server error says to restart it with the model's name.
- `LAB_VENV=<path>` still forces a specific venv for any command.
- Which policies need which venv: `POLICY_VENVS` in `lab/tools/config.py`.

### Full fine-tuning or LoRA

The entry's `finetune:` picks the method. The client sends the same field to the policy server, which loads the model accordingly, so nothing else changes between the two.

| `finetune:` | What trains | What is saved / pushed | How the server loads it |
|---|---|---|---|
| `full` (default) | everything SmolVLA normally trains: the action expert and projections (~100M params; the VLM stays frozen) | the whole model, ~900 MB | `from_pretrained(repo)` |
| `lora` | low-rank adapters on the action expert, plus the small projections in full (8.5M params at r=32) | `adapter_config.json` + `adapter_model.safetensors`, ~21 MB at r=32 | base model named in the adapter (`lerobot/smolvla_base`) + the adapter, kept unmerged (the base weights are bf16; merging would round the update away) |

```bash
lab/bin/train smolvla_test_run_1_lora                          # entry has finetune: lora and train.lora settings
lab/bin/train smolvla_test_run_1_lora --lora.r=16 --lora.targets=vlm+expert   # override LoRA settings
lab/bin/train smolvla_test_run_1 --finetune lora --no-push --output_dir=outputs/train/try_lora  # one-off, local only
```
- LoRA settings (`train.lora:` or `--lora.<key>=`): `r` (32), `alpha` (32), `dropout` (0.05), `targets`: `expert` (default) or `vlm+expert` (adapts the VLM's language layers too; more GPU memory, since backprop then runs through the VLM).
- LoRA target presets exist for `smolvla`, `pi0`, `pi05` and `xvla` (`src/lerobot/policies/lora.py`). `expert` adapts the action expert / policy transformer; `vlm+expert` also adapts the language backbone, never the vision tower. Other policies need `--lora.targets=<module-name regex>`.
- Give each method its own entry and Hub id (as `smolvla_test_run_1` / `smolvla_test_run_1_lora`), so you can run the same experiment on both. `--finetune` that disagrees with the entry is refused unless `--no-push`: the server would load the pushed model the wrong way.
- `--resume` and `--push` detect a LoRA checkpoint by its `adapter_config.json`. Pushing one method removes the other's weights from the Hub repo.
- Testing a local LoRA checkpoint: set `pretrained_name_or_path: outputs/train/<run>/checkpoints/last/pretrained_model` with `finetune: lora`.

Then run it (laptop: `lab/bin/server`; Pi: `git pull`, then):
```bash
lab/bin/run replication/baseline --model smolvla_test_run_1 --trials 3
```

## Adding a model

Add an entry to `lab/models.yaml` (templates are in the file). `pretrained_name_or_path` is resolved **on the laptop**: a Hugging Face repo id, or a local checkpoint path such as `outputs/train/<run>/checkpoints/last/pretrained_model`, so you can test checkpoints without uploading them. `fps` is the fps of the training dataset. Set `rtc_enabled: true` only for flow-matching policies (smolvla, pi0, pi05); `run` refuses the invalid combination.

The same condition across models:
```bash
lab/bin/run replication/drop_obs --model smolvla_cube_v1 --trials 20
lab/bin/run replication/drop_obs --model act_cube_v1     --trials 20
```
`--model` replaces the experiment's model, including any model fields the experiment pinned.

## Adding an experiment

Create `lab/experiments/<group>/<name>.yaml` with only what differs from the defaults:

```yaml
description: Observation drops
model: smolvla_cube_v1
drop_obs:
  - {start_s: 3.0, duration_s: 2.0}
```

Fault keys: `drop_obs`, `drop_action`, `dup_obs`, `dup_action`, `reorder_obs`, `reorder_action`, `disconnect` (lists of `{start_s, duration_s}`), and `spikes` (list of `{start_s, delay_ms}`). For the remaining parameters, see the `ExperimentConfig` fields in `examples/experiments/run_drtc_experiment.py`.

Conventions:
- **Groups** are research questions, e.g. `smoke/`, `replication/`, `fallback/`.
- **Never edit a config after reporting results from it.** Copy it to a new name instead. Every run stores the resolved config and git commit, but the file name is what `index.csv` groups by.
- Commit configs before running, so `run.json` shows `git_dirty: false`.

## One-time setup

**Laptop**
```bash
uv venv --python 3.12 && uv pip install -e ".[smolvla,async,feetech]"
lab/bin/laptop-setup
```
Copy the calibration files to the Pi. The laptop's newer LeRobot uses different folder names:
```bash
C=.cache/huggingface/lerobot/calibration
ssh rasp@raspberrypi.local "mkdir -p ~/$C/robots/so101_follower ~/$C/teleoperators/so101_leader"
scp ~/$C/robots/so_follower/follower.json rasp@raspberrypi.local:~/$C/robots/so101_follower/
scp ~/$C/teleoperators/so_leader/leader.json rasp@raspberrypi.local:~/$C/teleoperators/so101_leader/
```

**Pi**: join the laptop hotspot, then:
```bash
git clone https://github.com/arshjeevs/DRTC.git ~/drtc && cd ~/drtc && lab/bin/pi-setup
```

**Phone**: install the classic *DroidCam - Webcam for PC* app (not DroidCam OBS). Turn on USB debugging, choose *Always allow* for the Pi, and set *Stay awake*, auto-rotate off, landscape.

**Device paths** (once, on either machine): `rig.yaml` uses `/dev/*/by-id/` names, which carry the USB vendor and serial number, so they are the same on the Pi and the laptop and survive re-plugging. Globs resolve to their single match: camera1 is `/dev/v4l/by-id/usb-*Brio*-video-index0`. For the arms, plug in only the follower and run `ls /dev/serial/by-id/`, then plug in the leader and run it again. Put the two paths in `robot.port` and `leader.port`. Never use `ttyACMn`: plug order changes it, and on the laptop the phone itself takes `ttyACM0`.

**Physical**: mount both cameras rigidly and tape-mark the camera, robot, cube and target positions. Keep lighting fixed. Never swap camera1 and camera2.

## Recalibrating

On the machine the arms are plugged into (`git pull` first, so the ports match):
```bash
lab/bin/calibrate follower
lab/bin/calibrate leader
```
Follow the prompts: put the arm in the middle of its range, press Enter, move every joint through its full range, then press Enter.

Both machines must use the **same** calibration files. Actions are stored in calibrated units, so a dataset recorded with one calibration and replayed with another moves the arm differently. After recalibrating on the Pi, copy the files to the laptop:
```bash
C=.cache/huggingface/lerobot/calibration
rm -f ~/$C/robots/so101_follower/follower.json ~/$C/teleoperators/so101_leader/leader.json  # old files or links
scp rasp@raspberrypi.local:~/$C/robots/so101_follower/follower.json ~/$C/robots/so101_follower/
scp rasp@raspberrypi.local:~/$C/teleoperators/so101_leader/leader.json ~/$C/teleoperators/so101_leader/
```
Recalibrating invalidates datasets recorded before it. Re-record them, or keep the old files (`*.bak`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `adb` shows `unauthorized` | Unlock the phone and tap *Allow*. If no prompt: `adb kill-server` |
| phonecam keeps retrying | DroidCam must be open, with no other viewer (browser, laptop client) connected |
| `/dev/video10` missing | `sudo modprobe v4l2loopback`. If that errors after a kernel update: `sudo apt install --reinstall v4l2loopback-dkms` and reboot |
| `failed to set fourcc ... /dev/video10` | Harmless; the virtual camera is YUYV |
| Arm unresponsive / port error | `ls -l /dev/serial/by-id/` and put the stable `by-id` paths in `rig.yaml` (ttyACM0/1 swap on re-plug) |
| `write failed: [Errno 19] No such device` | The arm's USB dropped mid-connect. `dmesg \| tail -30`: `over-current` or `disconnect` means power. Use the official 27 W (5 V/5 A) Pi supply, or a powered USB hub; don't charge the phone from the Pi |
| `FeetechMotorsBus motor check failed` with `ttyACM0` | That port is another device (on the laptop, the phone). Use the `by-id` paths in `rig.yaml` |
| record: dataset `already exists` | A failed run leaves the folder behind. Delete `~/.cache/huggingface/lerobot/<hf_user>/<dataset>`, or add `--resume` if it holds real episodes |
| `Failed to find Rerun Viewer` | Run through `lab/bin/record` (puts the venv on `PATH`) |
| Asked to recalibrate | Wrong `follower.json`; stop and check it |
| No motion for 20–45 s | Model loading on the laptop; normal |
| `obs_one_way_latency_ms` in the hours | Pi clock unsynced (chrony). RTT unaffected |
| `StreamActionsDense ... CANCELLED` at end | Normal shutdown |

## Reference numbers (2026-10-05, Jack's SmolVLA, RTC on, 8 flow steps, 60 Hz)

Control loop 16.7 ms · RTT ~270 ms · server inference ~220 ms (~130 ms with RTC off) · image upload ~25 ms · 0 stalls after startup.
