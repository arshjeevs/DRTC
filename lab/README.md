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

## Commands

| Command | Where | Purpose |
|---|---|---|
| `lab/bin/server` | laptop | Policy server on the GPU. Loads whichever model each run asks for |
| `lab/bin/session [record]` | Pi / laptop | tmux: `phonecam` window + `work` window running preflight. Re-run to re-attach |
| `lab/bin/run <group/name> [--model M] [--trials N] [--note TEXT] [--no-label]` | Pi | Run a condition. Prompts for scene reset before and success after each trial |
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
