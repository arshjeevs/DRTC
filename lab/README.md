# Lab

Our research layer on top of DRTC/LeRobot. Everything we own lives here; upstream code stays untouched, so `git pull upstream drtc` stays conflict-free.

```
lab/
├── rig.yaml                 hardware: hosts, ports, calibration IDs, cameras (only place they live)
├── experiments/
│   ├── _defaults.yaml       shared DRTC / RTC / filter settings
│   └── <group>/<name>.yaml  one file = one experimental condition
├── bin/                     entry points
└── tools/                   Python behind the entry points
results/                     (gitignored)
├── index.csv                one row per trial: experiment, success label, git commit, note
└── <group>/<name>/<run_id>/ metrics CSV/JSON + resolved config.yaml + run.json
```

Config precedence: `_defaults.yaml` < `rig.yaml` < experiment file.

## Commands

| Command | Where | Purpose |
|---|---|---|
| `lab/bin/server [--fps N]` | laptop | Policy server on the GPU. `--fps` must match the experiment (default 30) |
| `lab/bin/session` | Pi | tmux: `phonecam` window + `work` window running preflight. Re-run to re-attach |
| `lab/bin/run <group/name> [--trials N] [--note TEXT] [--no-label]` | Pi | Run a condition. Prompts for scene reset before and success after each trial |
| `lab/bin/record <user>/<dataset> --task "..." [--episodes 50] [--resume]` | Pi | Record demos with the leader arm, hardware taken from `rig.yaml` |
| `lab/bin/preflight` | Pi | Check ports, calibration, cameras, server, clock |
| `lab/bin/phonecam` | Pi | DroidCam → `/dev/video10` bridge (started by `session`) |
| `lab/bin/pi-setup` | Pi, once | Packages, `/dev/video10` at boot, clock sync, venv |

## Session

1. Laptop plugged in, hotspot on. Robot powered. Phone in the Pi's USB with DroidCam open.
2. Laptop: `git pull && lab/bin/server --fps 60`. Use 60 for `smoke/jack_baseline`; otherwise use the experiment's `fps`.
3. Pi: `git pull && lab/bin/session`. All preflight lines should say `ok`.
4. Pi, in the `work` window: `lab/bin/run smoke/jack_baseline --trials 3`

tmux keys: `Ctrl+b n`/`p` switches windows, `Ctrl+b d` detaches.

## Adding an experiment

Create `lab/experiments/<group>/<name>.yaml` with only what differs from `_defaults.yaml`:

```yaml
description: Observation drops, own SmolVLA
policy_type: smolvla
pretrained_name_or_path: <hf_user>/<model>
task: <the exact sentence used when recording>
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
sudo apt install -y chrony && printf "allow 10.42.0.0/24\nlocal stratum 10\n" | sudo tee -a /etc/chrony/chrony.conf && sudo systemctl restart chrony
# If ufw is active: sudo ufw allow 8080/tcp && sudo ufw allow 123/udp
```
Copy the calibration files to the Pi. The laptop's newer LeRobot uses different folder names:
```bash
C=.cache/huggingface/lerobot/calibration
ssh rasp@raspberrypi "mkdir -p ~/$C/robots/so101_follower ~/$C/teleoperators/so101_leader"
scp ~/$C/robots/so_follower/follower.json rasp@raspberrypi:~/$C/robots/so101_follower/
scp ~/$C/robots/so_follower/leader.json   rasp@raspberrypi:~/$C/teleoperators/so101_leader/
```

**Pi**: join the laptop hotspot, then:
```bash
git clone https://github.com/arshjeevs/DRTC.git ~/drtc && cd ~/drtc && lab/bin/pi-setup
```

**Phone**: install the classic *DroidCam - Webcam for PC* app (not DroidCam OBS). Turn on USB debugging, choose *Always allow* for the Pi, and set *Stay awake*, auto-rotate off, landscape.

**Physical**: mount both cameras rigidly and tape-mark the camera, robot, cube and target positions. Keep lighting fixed. Never swap camera1 and camera2.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `adb` shows `unauthorized` | Unlock the phone and tap *Allow*. If no prompt: `adb kill-server` |
| phonecam keeps retrying | DroidCam must be open, with no other viewer (browser, laptop client) connected |
| `/dev/video10` missing | `lab/bin/pi-setup` (makes it load at boot) |
| `failed to set fourcc ... /dev/video10` | Harmless; the virtual camera is YUYV |
| Arm unresponsive / port error | `lerobot-find-port`, then update `rig.yaml` (ttyACM0/1 swap on re-plug) |
| Asked to recalibrate | Wrong `follower.json`; stop and check it |
| No motion for 20–45 s | Model loading on the laptop; normal |
| `obs_one_way_latency_ms` in the hours | Pi clock unsynced (chrony). RTT unaffected |
| `StreamActionsDense ... CANCELLED` at end | Normal shutdown |

## Reference numbers (2026-10-05, Jack's SmolVLA, RTC on, 8 flow steps, 60 Hz)

Control loop 16.7 ms · RTT ~270 ms · server inference ~220 ms (~130 ms with RTC off) · image upload ~25 ms · 0 stalls after startup.
