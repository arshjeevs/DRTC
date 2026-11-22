# Windows setup and commands

How to run this project natively on a Windows 10/11 laptop with an NVIDIA GPU: record demos, train, and run DRTC with the policy server and the robot client on the same laptop. No WSL needed.

> Written and tested on Linux. The Windows-specific code paths (COM ports, camera indexes, console key controls, `LAB_RIG`) are covered by tests on Linux, but this guide has not yet been run end-to-end on a Windows machine. If a step fails, the [Troubleshooting](#troubleshooting) table lists the likely causes.

What is different from Linux:

| | Linux / Pi | Windows |
|---|---|---|
| Entry points | `lab/bin/<cmd>` (bash) | `python -m lab.tools.<cmd>` (the bash scripts are thin wrappers around these) |
| Rig file | `lab/rig.yaml` | `lab/rig.windows.yaml`, selected with `LAB_RIG` |
| Serial ports | `/dev/serial/by-id/...` | `COM5`, `COM6`, ... |
| Cameras | `/dev/v4l/by-id/...`, phone through v4l2loopback | OpenCV indexes `0`, `1`, ...; phone through the DroidCam Windows client |
| Server address | `10.42.0.1` (Pi client) | `127.0.0.1` (client on the same laptop) |
| Clock sync | chrony | not needed when client and server are on one machine |

## 1. Install the tools (once)

1. **NVIDIA driver**: install the latest Game Ready or Studio driver. Check it in PowerShell: `nvidia-smi`.
2. **Git for Windows**: https://git-scm.com/download/win (the defaults are fine).
3. **uv** (Python and package manager), in PowerShell:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   Close and reopen PowerShell afterwards so `uv` is on PATH.
4. **Developer Mode**: Settings > System > For developers > Developer Mode **On**. LeRobot marks the latest checkpoint with a symlink (`checkpoints/last`), and Windows only lets normal users create symlinks in Developer Mode. Without it, training crashes at the first checkpoint.
5. **Allow activation scripts** (once per user):
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```
6. **USB serial driver**: the SO101 boards use a CH340 chip (USB id `1a86`). Windows 10/11 normally installs the driver itself. If the arms do not show up as COM ports in Device Manager, install the CH341SER driver from the chip maker (WCH).

## 2. Get the code and the Python environment

```powershell
cd $HOME\Projects
git clone https://github.com/arshjeevs/DRTC.git drtc      # or git@github.com:arshjeevs/DRTC.git with an SSH key
cd drtc
git checkout drtc

uv venv --python 3.12
.venv\Scripts\activate
```

Install PyTorch **with CUDA first**. The default PyPI build for Windows is CPU-only, and installing the project first would pull that one in.

```powershell
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install -e ".[smolvla,async,feetech]"
```

Check that the GPU is visible:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

It must print `True NVIDIA GeForce RTX ...`. If it prints `False`, see [Troubleshooting](#troubleshooting).

Every new PowerShell window: `cd $HOME\Projects\drtc` and `.venv\Scripts\activate`. All commands below run from the repo root with the venv active.

## 3. Accounts (once)

```powershell
hf auth login       # Hugging Face token with WRITE access (datasets and models are pushed)
wandb login         # training logs go to wandb project drtc
```

## 4. Point the tools at the Windows rig

`lab/rig.yaml` stays the Linux/Pi rig. On Windows, set `LAB_RIG` to `lab\rig.windows.yaml`. Use an absolute path, so it works from any folder:

```powershell
setx LAB_RIG "$PWD\lab\rig.windows.yaml"     # permanent; open a NEW PowerShell window afterwards
$env:LAB_RIG = "$PWD\lab\rig.windows.yaml"   # or: only for the current window
```

Check that it is picked up:

```powershell
python -m lab.tools.config get server.host     # should print 127.0.0.1
```

## 5. Find the arms and cameras, and fill in `lab\rig.windows.yaml`

**Serial ports.** Plug in both arms (powered), then:

```powershell
lerobot-find-port
```

It asks you to unplug one arm and tells you which COM port disappeared. Do it for the follower, then for the leader, and put the results in `robot.port` and `leader.port`.

**Cameras.**
- Zebronics (wrist camera): plug it in.
- Phone (scene camera): install the **DroidCam** app on the phone and the **DroidCam Windows client** on the laptop, connect over USB or Wi-Fi, and start the video. The client exposes the phone as a webcam ("DroidCam Source").
- Close any other program that uses a camera (Camera app, Teams, Zoom, browser tabs); Windows lets only one program open a camera at a time.

Then:

```powershell
lerobot-find-cameras opencv
```

It lists the camera indexes and saves one test frame from each into `outputs\captured_images`. Open the images to see which index is the wrist camera and which is the phone, and set `camera1.path` (Zebronics, wrist) and `camera2.path` (phone, scene) in `lab\rig.windows.yaml`.

Keep `camera1` = Zebronics (wrist) and `camera2` = phone (scene), the same as on Linux, so datasets and models from both machines match.

COM numbers and camera indexes can change when you plug into a different USB socket. Use the same sockets every time, or re-run the two `find` commands.

## 6. Calibration

The calibration files must be identical on every machine that records or runs the robot, because actions are stored in calibrated units. Either copy them from the Linux laptop, or calibrate here and copy them to the other machines.

Files on Windows:

```
%USERPROFILE%\.cache\huggingface\lerobot\calibration\robots\so101_follower\follower.json
%USERPROFILE%\.cache\huggingface\lerobot\calibration\teleoperators\so101_leader\leader.json
```

**Copy from the Linux laptop** (OpenSSH is built into Windows 10/11):

```powershell
$C = "$HOME\.cache\huggingface\lerobot\calibration"
mkdir -Force "$C\robots\so101_follower", "$C\teleoperators\so101_leader"
scp <linux-user>@<linux-laptop-ip>:~/.cache/huggingface/lerobot/calibration/robots/so101_follower/follower.json "$C\robots\so101_follower\"
scp <linux-user>@<linux-laptop-ip>:~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/leader.json "$C\teleoperators\so101_leader\"
```

**Or calibrate here** (use your COM ports):

```powershell
lerobot-calibrate --robot.type=so101_follower --robot.port=COM5 --robot.id=follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=COM6 --teleop.id=leader
```

## 7. Check everything

```powershell
python -m lab.tools.preflight record     # arms, calibration files, both cameras
```

All checks should be `ok`. Camera indexes cannot be checked without opening the camera, so a wrong index shows up as a camera error here, not as a missing device.

## 8. Commands: Linux to Windows

| Linux / Pi | Windows (PowerShell, repo root, venv active) |
|---|---|
| `lab/bin/preflight record` | `python -m lab.tools.preflight record` |
| `lab/bin/preflight` | `python -m lab.tools.preflight` |
| `lab/bin/record <user>/<dataset> --task "..."` | `python -m lab.tools.record <user>/<dataset> --task "..."` |
| `lab/bin/train <model>` | `python -m lab.tools.train <model>` |
| `lab/bin/server` | `python examples/tutorial/async-inf/policy_server_drtc.py --host 0.0.0.0 --port 8080 --verbose-diagnostics` |
| `lab/bin/run <group/name> --model <model>` | `python -m lab.tools.run <group/name> --model <model>` |
| `lab/bin/list` | `python -m lab.tools.config list` |
| `lab/bin/calibrate follower` | `lerobot-calibrate --robot.type=so101_follower --robot.port=COM5 --robot.id=follower` (see step 6) |
| `lab/bin/session`, `laptop-setup`, `pi-setup` | not needed on Windows (phone camera through the DroidCam client, no chrony) |

All options are the same as on Linux (see `lab/README.md`), because the bash scripts only call these Python modules.

## 9. Record a dataset

```powershell
python -m lab.tools.record orangemoon599/train_new_2 --task "Pick the green cube and place it in the box" --episodes 50 --dry-run
python -m lab.tools.record orangemoon599/train_new_2 --task "Pick the green cube and place it in the box" --episodes 50
```

- Use the **exact same task sentence** for every dataset of a task; the model is conditioned on it.
- Keys work in the **PowerShell window** that runs the command (keep it focused, not the Rerun viewer): **Right arrow** or `n` = next, **Left arrow** or `r` = re-record, **Esc** or `q` = stop and save.
- The Rerun viewer opens automatically. Turn it off with `--no-display`.
- Add more episodes to a dataset later with `--resume` (`--episodes` then counts the **additional** episodes).
- The dataset is pushed to the Hub at the end (`--no-push` keeps it local).

## 10. Train

Add an entry to `lab/models.yaml` (copy an existing one, for example `smolvla_train_new_1_lora`), then:

```powershell
python -m lab.tools.train smolvla_train_new_1_lora --dry-run    # check the command and the push target
$env:PYTHONUNBUFFERED = "1"
python -m lab.tools.train smolvla_train_new_1_lora 2>&1 | Tee-Object -FilePath logs\train_smolvla_train_new_1_lora.log
```

(`mkdir -Force logs` first if the folder does not exist.) `finetune: full` or `lora`, `--resume`, `--push`, `--steps` and the LoRA flags work exactly as described in the Training section of `lab/README.md`. Keep the laptop plugged in and set Windows to never sleep while training.

## 11. Run DRTC on the laptop (server and robot client on one machine)

Two PowerShell windows, both in the repo root with the venv active.

**Window 1: policy server**

```powershell
python examples/tutorial/async-inf/policy_server_drtc.py --host 0.0.0.0 --port 8080 --verbose-diagnostics
```

Leave it running. It loads whichever model each run asks for.

**Window 2: preflight, then the run**

```powershell
python -m lab.tools.preflight                     # follower, cameras, server reachable at 127.0.0.1:8080
python -m lab.tools.run replication/baseline --model smolvla_train_new_1_lora --trials 3
```

Each trial asks you to reset the scene and press Enter, and asks for success afterwards. Results go to `results\` exactly as on the Pi. Fault experiments work the same way, e.g. `python -m lab.tools.run replication/drop_obs --model <model>`.

## 12. Later: Pi as the client, Windows laptop as the server

Only needed when you move the robot back to the Pi.

1. Start the server on Windows (step 11, window 1).
2. Allow the port through the Windows firewall (PowerShell **as administrator**, once):
   ```powershell
   New-NetFirewallRule -DisplayName "DRTC policy server" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow
   ```
3. Find the laptop's IP on the network the Pi uses: `ipconfig`.
4. On the Pi, set `server.host` in `lab/rig.yaml` to that IP, then use the normal Pi commands (`lab/bin/session`, `lab/bin/run ...`).
5. The Pi syncs its clock with chrony to the Linux laptop. Windows does not act as a time server by default, so the preflight clock check fails with a Windows server. Round-trip latency (what DRTC uses) is still valid; only one-way timings in the logs need synced clocks.

## Troubleshooting

| Problem | Fix |
|---|---|
| `torch.cuda.is_available()` is `False` | The CPU build of PyTorch got installed. `uv pip install --reinstall torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126`. Also check `nvidia-smi` works. |
| `.venv\Scripts\activate` "cannot be loaded because running scripts is disabled" | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (step 1.5) |
| Training fails at the first checkpoint with `WinError 1314` / "A required privilege is not held" | Developer Mode is off (step 1.4), so the `checkpoints\last` symlink cannot be created |
| `missing devices: COM5` | Wrong COM number, arm unpowered, or driver missing. `lerobot-find-port`, or look under "Ports (COM & LPT)" in Device Manager |
| `Access is denied` on a COM port | Another program has the port open (another record/run, an Arduino IDE, a serial monitor). Close it |
| A camera fails to open or gives the wrong image | Wrong index (re-run `lerobot-find-cameras opencv`), or another program is using it. Close the Camera app, Teams, Zoom and browser tabs |
| Camera opens but fails at 800x600 / 30 fps | Remove `fourcc: MJPG` from that camera in `lab\rig.windows.yaml`, or lower `width`/`height`. The DroidCam client may also be set to a different resolution |
| The phone camera works in preflight but fails in `run` | The run applies `camera1`'s `fourcc` to both cameras. If DroidCam rejects MJPG, remove `fourcc` from `camera1` and check that the Zebronics still delivers 30 fps |
| Keys do nothing while recording | Click the PowerShell window that runs the record command; the keys are read from that window |
| `python -m lab.tools.run` says the policy server is not reachable | Start the server first (step 11). `server.host` must be `127.0.0.1` on Windows; `0.0.0.0` does not work as a destination there |
| Text in `.yaml` files looks garbled, or a `UnicodeDecodeError` | Save the files as UTF-8 (VS Code does by default) |
| `lab.tools` not found | Run the commands from the repo root (`cd $HOME\Projects\drtc`) |
