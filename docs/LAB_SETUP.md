# Lab Setup Guide (SO101 + Raspberry Pi + Laptop GPU)

How to bring up the DRTC setup: the SO101 arm on a Raspberry Pi 5, two cameras (Logitech Brio 100 plus an Android phone running DroidCam), and the policy running on the laptop GPU.

- **First time on a machine:** do [One-time setup](#one-time-setup).
- **Every lab session:** do [Every session](#every-session). It takes about 5 minutes.

```
 SO101 follower ──USB── Raspberry Pi 5 ─────── Wi-Fi (laptop hotspot) ─────── Laptop RTX 4060
 SO101 leader ───USB──┤  DRTC client, 60 Hz loop        10.42.0.x ↔ 10.42.0.1     DRTC policy server
 Brio 100 ───────USB──┤  camera1                                                  SmolVLA on CUDA
 Phone (DroidCam) USB─┘  camera2 → ffmpeg → /dev/video10
```

## Reference values

| Thing | Value |
|---|---|
| Laptop hotspot IP (policy server) | `10.42.0.1`, port `8080` |
| Laptop repo | `~/Projects/drtc` (has the `smolvla,async,feetech` extras) |
| Pi repo | `~/drtc` (has the `async,feetech` extras) |
| Follower port / ID | `/dev/ttyACM0` / `follower` |
| Leader port / ID | `/dev/ttyACM1` / `leader` |
| camera1 (Brio 100) | `/dev/v4l/by-path/platform-xhci-hcd.1-usb-0:1:1.0-video-index0` |
| camera2 (phone) | `/dev/video10` (virtual camera fed by DroidCam) |
| Camera format | 800×600 @ 30 fps |
| DroidCam stream | `http://127.0.0.1:4747/video`, forwarded over USB with `adb` |
| Experiment config | `examples/experiments/configs/laptop_baseline.yaml` |

**Never swap camera1 and camera2.** Datasets and trained models depend on the order.

---

## One-time setup

### Laptop

```bash
cd ~/Projects/drtc
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[smolvla,async,feetech]"
```

**Hotspot.** The Pi connects to the laptop's Wi-Fi hotspot, which gives the laptop IP `10.42.0.1`. To make it start automatically:
```bash
nmcli connection show                      # find the hotspot connection name
nmcli connection modify <HOTSPOT_NAME> connection.autoconnect yes
```

**Time server for the Pi.** The Pi has no clock battery, so it syncs its clock from the laptop:
```bash
sudo apt install -y chrony
printf "allow 10.42.0.0/24\nlocal stratum 10\n" | sudo tee -a /etc/chrony/chrony.conf
sudo systemctl restart chrony
```

**Firewall** (only if `sudo ufw status` says it's active):
```bash
sudo ufw allow 8080/tcp     # policy server
sudo ufw allow 123/udp      # clock sync
```

**Copy the calibration files to the Pi.** These were made with a newer LeRobot version, which uses different folder names:
```bash
ssh rasp@raspberrypi "mkdir -p ~/.cache/huggingface/lerobot/calibration/robots/so101_follower ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader"
scp ~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json \
    rasp@raspberrypi:~/.cache/huggingface/lerobot/calibration/robots/so101_follower/follower.json
scp ~/.cache/huggingface/lerobot/calibration/robots/so_follower/leader.json \
    rasp@raspberrypi:~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/leader.json
```

### Raspberry Pi

Join the laptop hotspot once; the Pi remembers it after that:
```bash
sudo nmcli dev wifi connect "<HOTSPOT_SSID>" password "<PASSWORD>"
```

Clone the repo and run the one-time setup script. It installs the packages, makes `/dev/video10` appear at every boot, sets up clock sync, and installs Python:
```bash
git clone https://github.com/arshjeevs/DRTC.git ~/drtc
cd ~/drtc
./scripts/lab/pi_setup_once.sh
```

To check the clock sync (once the laptop runs chrony):
```bash
chronyc tracking      # "Reference ID" should show 10.42.0.1; "System time" should be well under 1 ms
```

### Phone (Android)

1. Install **DroidCam - Webcam for PC**, the classic app, **not** "DroidCam OBS".
2. Enable USB debugging: Settings → About phone → tap *Build number* 7 times → Developer options → turn on *USB debugging*.
3. Plug the phone into the Pi. When the phone asks, tap **Always allow from this computer**.
4. In the phone settings: disable auto-rotate, keep the screen on (Developer options → *Stay awake* while charging), and use landscape orientation.

### Physical setup (decide once, then don't change it)

- Mount the **Brio** and the **phone** rigidly. Tape-mark their positions on the table or mounts. If a camera moves after recording, the trained model sees a view it never trained on.
- Tape-mark the robot base, the cube's start zone and the target.
- Use the same lighting every session. Avoid windows; daylight changes through the day.
- Optionally lock the Brio's auto exposure and white balance, so the image doesn't drift:
  ```bash
  v4l2-ctl -d /dev/video0 -l                       # list controls (names vary by firmware)
  v4l2-ctl -d /dev/video0 -c auto_exposure=1 -c white_balance_automatic=0
  ```

---

## Every session

### 1. Power and cables
- Laptop **plugged in**: on battery the GPU slows down and all latency numbers change.
- Laptop hotspot on.
- Robot power on. Robot, leader, Brio and phone all plugged into the Pi.
- Phone: open **DroidCam** and leave it on the main screen.

### 2. Laptop: start the policy server
```bash
cd ~/Projects/drtc && git pull
./scripts/lab/start_laptop_server.sh             # use FPS=30 ./scripts/lab/... for 30 fps models
```
Leave it running. It prints the laptop IPs and warns you if the laptop is on battery.

### 3. Pi: start the session
```bash
ssh rasp@raspberrypi
cd ~/drtc && git pull
./scripts/lab/pi_session.sh
```
This opens a tmux session with two windows:
- **`phonecam`** runs the phone camera bridge. It waits for the phone, forwards the port and reconnects automatically if the stream drops.
- **`work`** activates the Python environment and runs the **preflight checks** automatically.

Every preflight line should say `[OK]`. Each `[FAIL]` line prints a hint telling you what to fix.

tmux keys: `Ctrl+b` then `n` / `p` switches windows, and `Ctrl+b` then `d` detaches. If SSH drops, reconnect and run `./scripts/lab/pi_session.sh` again; it re-attaches to the running session.

### 4. Run
In the `work` window:
```bash
./scripts/lab/run_experiment.sh                                # laptop_baseline config
./scripts/lab/run_experiment.sh path/to/other_config.yaml      # any other config
```
Results are written to `results/experiments/<run_name>/`. Plot them with:
```bash
python examples/experiments/plot_results.py --input results/experiments/<run_name>
```

### 5. End of session
- `Ctrl+C` the server on the laptop.
- On the Pi: `tmux kill-session -t robot`.
- Copy results to the laptop: `scp -r rasp@raspberrypi:~/drtc/results ~/Projects/drtc/`

---

## Recording demonstrations (leader arm)

Run on the Pi with the `phonecam` bridge running. Do a 2-episode test before recording the full set.
```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower \
  --robot.cameras="{camera1: {type: opencv, index_or_path: /dev/v4l/by-path/platform-xhci-hcd.1-usb-0:1:1.0-video-index0, width: 800, height: 600, fps: 30, fourcc: MJPG}, camera2: {type: opencv, index_or_path: /dev/video10, width: 800, height: 600, fps: 30}}" \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader \
  --dataset.repo_id=<hf_user>/<dataset_name> \
  --dataset.single_task="<one fixed task sentence>" \
  --dataset.num_episodes=50 --dataset.episode_time_s=30 --dataset.reset_time_s=10 \
  --display_data=false
```
- Log in to Hugging Face first, once: `huggingface-cli login`.
- Record at **30 fps**, because the cameras run at 30 fps. When running a model trained on this data, set `fps: 30` in its config and start the server with `FPS=30`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `adb devices` shows `unauthorized` | Phone hasn't trusted the Pi | Unlock the phone and tap **Allow**. If no prompt appears: `adb kill-server && adb devices` |
| ffmpeg: `Error opening input: End of file` | Port forwarded, but DroidCam not serving | Open DroidCam on its main screen. The bridge script retries on its own |
| ffmpeg: `Connection refused` | USB port forward dropped (phone re-plugged or locked) | The bridge script redoes the forward automatically. Manually: `adb forward tcp:4747 tcp:4747` |
| `/mjpegfeed` returns 404 | Newer DroidCam versions use `/video` | Scripts already use `/video` |
| Stream drops whenever you open it in a browser | DroidCam serves **one viewer at a time** | Don't open the stream elsewhere while the bridge runs |
| Phone camera has HTTP 404 on both URLs | "DroidCam OBS" app installed | Install the classic **DroidCam - Webcam for PC** |
| `/dev/video10` missing | v4l2loopback not loaded | `sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="PhoneCam" exclusive_caps=1`. Rerun `pi_setup_once.sh` so it loads at boot |
| `failed to set fourcc=MJPG ... /dev/video10` warning | The virtual camera is YUYV | Harmless |
| Arm doesn't respond, or port errors | `ttyACM0`/`ttyACM1` swapped after re-plugging | `lerobot-find-port`, then update `robot_port` (or use the path from `ls /dev/serial/by-id/`) |
| LeRobot asks to recalibrate | Calibration file doesn't match this arm | Stop and check the right `follower.json` was copied. Don't press Enter blindly |
| Client hangs at "Starting client..." | Server unreachable | Check the server is running, the Pi is on the hotspot (`ip a` shows `10.42.0.x`), and the firewall |
| No arm motion for ~20–45 s after starting | Laptop is loading the model | Normal on every run |
| `obs_one_way_latency_ms` is huge (hours) | Pi clock not synced | Set up chrony (one-time setup). Round-trip latency is unaffected |
| `StreamActionsDense ... CANCELLED` at the end of a run | Normal shutdown | Harmless |
| `Error finding RealSense cameras: name 'rs' is not defined` | RealSense library not installed | Harmless |
| `lerobot-find-cameras` lists `/dev/video19–35` | Pi 5 internal image-processing devices | Ignore them. Only `/dev/video0` (Brio) and `/dev/video10` (phone) matter |

## Baseline numbers (for sanity checks)

Measured on 2026-10-05 with Jack Vial's SmolVLA, RTC on, 8 flow-matching steps:

| Metric | Expected |
|---|---|
| Pi control loop (`loop_dt_ms`) | 16.7 ms (60 Hz) |
| Round trip (`total_latency_rtt_ms`) | ~270 ms |
| Server inference (`infer_total_ms`) | ~220 ms (~130 ms with RTC off) |
| Image upload (`obs_send_ms`) | ~25 ms |
| Stalls after startup | 0 |

If your numbers are far off, check whether the laptop is on battery, the Wi-Fi signal, and whether another program is using the GPU.
