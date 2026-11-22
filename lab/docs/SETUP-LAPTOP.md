# Setting up a second robot laptop

*For a project teammate's laptop (Ubuntu, NVIDIA GPU) that works alongside the first one: the **same arms, cameras
and phone**, the same Hugging Face account, wandb, A100 and H200. Written 2026-11-22. What each stage does and why:
[HOW-IT-WORKS.md](HOW-IT-WORKS.md); the commands for daily work: [../WORKFLOW.md](../WORKFLOW.md).*

Every step ends with a **check**: what you should see. Don't go on until it matches; if it doesn't, the
[troubleshooting table](#troubleshooting) at the end has the usual causes, or paste the output to Claude.

In this guide **"your laptop"** is the new one and **"the first laptop"** is the one the project has used so far
(`laptop` in `lab/hosts.yaml`). The new one gets the name `laptop2`.

What is shared and what is not:

| Shared, never changed | Only on the new laptop |
|---|---|
| The arms (follower, leader), the wrist camera, **the first laptop owner's phone** (scene camera) | the repo checkout `~/Projects/drtc-major` |
| The rig `so101_v2` (`lab/rigs/so101_v2.yaml`) and the **calibration files** (copied, never redone) | its entry `laptop2` in `lab/hosts.yaml` |
| Hugging Face account `orangemoon599` (datasets, models, results store), wandb, the A100 and H200 logins | your own GitHub account (a collaborator on the repo) and git name |

## 1. Check the laptop

```bash
lsb_release -ds      # e.g. Ubuntu 24.04.x LTS or newer
nvidia-smi           # the GPU and "Driver Version"
hostname             # the name lab will know this laptop by (write it down: step 6)
df -h ~              # free space
```

**Check:**
- `nvidia-smi` shows the GPU and **Driver Version 580 or newer**. The project's PyTorch is built for CUDA 13.0, which
  needs driver 580+. If `nvidia-smi` is missing or the driver is older: `sudo ubuntu-drivers install`, reboot, run
  `nvidia-smi` again.
- At least **40 GB free** in your home folder (the Python environment ~10 GB, the model downloads for serving several
  GB each).

## 2. System tools

```bash
sudo apt update
sudo apt install -y git curl tmux ffmpeg v4l-utils adb android-sdk-platform-tools-common \
                    v4l2loopback-dkms v4l2loopback-utils chrony openssh-client
sudo usermod -aG dialout,plugdev "$USER"     # access to the arms' serial ports and the phone over USB
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Log out and back in** (or reboot) so the new groups and `uv` on the PATH take effect.

**Check:** `id -nG` lists `dialout` and `plugdev`; `uv --version` prints a version.

## 3. GitHub and the code

The first laptop's owner adds your GitHub account as a **collaborator** on `arshjeevs/DRTC` (GitHub: the repo →
Settings → Collaborators). Accept the invitation (e-mail, or github.com/notifications).

Then an SSH key for GitHub, on your laptop:

```bash
ssh-keygen -t ed25519 -C "$(hostname)"       # Enter for the default file; a passphrase is optional
cat ~/.ssh/id_ed25519.pub                    # add this line on github.com -> Settings -> SSH and GPG keys -> New SSH key
```

The campus network blocks SSH to GitHub on port 22; GitHub also serves SSH on port 443. Add this to `~/.ssh/config`
(create the file if it does not exist):

```
Host github.com
    HostName ssh.github.com
    Port 443
    User git
```

```bash
ssh -T git@github.com                        # answer "yes" the first time
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
mkdir -p ~/Projects && cd ~/Projects
git clone -b major git@github.com:arshjeevs/DRTC.git drtc-major
```

**Check:** `ssh -T` says `Hi <your GitHub name>! You've successfully authenticated`; `git -C ~/Projects/drtc-major
status` says `On branch major`, nothing to commit.

## 4. The Python environment

```bash
cd ~/Projects/drtc-major
uv sync --frozen --extra gpu --extra sim     # Python 3.12, LeRobot 0.7, PyTorch for CUDA 13; ~10 min the first time
source .venv/bin/activate                    # once per terminal; then `lab ...` works
lab check
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Check:** `lab check` prints `registry: ... ok`; the last line prints `True` and your GPU's name. `False` means the
driver is missing or too old (step 1).

`--frozen` installs exactly the versions in `uv.lock`, the same as on every other machine. Run the same `uv sync`
line again whenever a `git pull` changes `uv.lock`.

## 5. Accounts: Hugging Face and wandb

Both use the shared accounts. Ask the first laptop's owner for:
- a Hugging Face **write** token of `orangemoon599` (huggingface.co → Settings → Access Tokens),
- the wandb API key (wandb.ai/authorize, logged in to the project's account).

```bash
cd ~/Projects/drtc-major && source .venv/bin/activate
read -rs HF_T && hf auth login --token "$HF_T"; unset HF_T   # paste the token (not shown), Enter
wandb login              # paste the wandb key
hf auth whoami           # -> user=orangemoon599
lab check --logging      # wandb logged in
```

**Check:** `hf auth whoami` prints `user=orangemoon599`; `lab check --logging` prints `wandb logged in as ...; runs go
to project drtc-lab`. Write
access matters: recordings, and the results of every official session, are pushed with this token.

## 6. Register the laptop in `lab/hosts.yaml`

`lab` finds out which machine it runs on by the hostname (`hostname`, step 1). Add this block under `hosts:` in
`lab/hosts.yaml`, right after the `laptop:` entry, with **your hostname** and GPU:

```yaml
  laptop2:
    label: laptop2
    hostname: <your hostname>
    roles: [record, client, serve]
    repo: ~/Projects/drtc-major
    gpus:
    - {name: RTX 4060 Laptop, memory_gb: 8}   # your GPU's name and memory, from nvidia-smi
```

```bash
lab check && lab list | grep laptop2   # laptop2 is listed under hosts
lab check --rig | grep '^rig'          # -> "rig so101_v2 on host laptop2"
git add lab/hosts.yaml
git commit -m "hosts.yaml: laptop2 (second robot laptop)"
git push
```

**Check:** `lab check --rig` names `host laptop2` (the arm lines can still say MISSING: step 8). The push works,
which proves your GitHub write access. Every session you run is now stored as `<date>-laptop2`.

## 7. Calibration: copy it, never redo it

The calibration files say where each motor's zero and limits are. Every dataset and every trial so far used the first
laptop's files; with other files the same joint angles would mean a different arm pose, and the models would no longer
match. So **copy the two files from the first laptop**, and never run `lab calibrate` on the new one.

On the **first laptop** (it writes `calibration-so101_v2.tgz` to the home folder):

```bash
tar czf ~/calibration-so101_v2.tgz -C ~/.cache/huggingface/lerobot/calibration \
    robots/so101_follower/follower.json teleoperators/so101_leader/leader.json
```

Move that file to your laptop (USB stick, or `scp` when both are on the same network), then on **your laptop**:

```bash
mkdir -p ~/.cache/huggingface/lerobot/calibration
tar xzf ~/calibration-so101_v2.tgz -C ~/.cache/huggingface/lerobot/calibration
lab check --rig | grep calibration
```

**Check:** the two calibration lines show the same fingerprints as on the first laptop. As of 2026-11-22 these are:

| File | Fingerprint (`sha`) |
|---|---|
| `robots/so101_follower/follower.json` | `197edb851610` |
| `teleoperators/so101_leader/leader.json` | `9e947397bbf4` |

If the arms are ever recalibrated on the first laptop, copy the files again (the same two commands) and compare the
fingerprints on both laptops.

## 8. The laptop's robot setup: phone camera and checks

```bash
cd ~/Projects/drtc-major && lab/bin/laptop-setup
```

It installs the phone camera bridge's device (`/dev/video10` at every boot), the time server and firewall rules used
when a Pi is on the laptop's hotspot (harmless otherwise), and checks the calibration files from step 7. It asks for
the sudo password. Ends with `Laptop setup complete.`

Then plug in, as for every session:
1. the follower arm (and its power supply) and the leader arm, over USB;
2. the wrist camera (Zebronics) over USB;
3. **the first laptop owner's phone** over USB, with DroidCam open. The first time, the phone asks to allow USB
   debugging for this computer: tap Allow (and "Always allow").

```bash
lab/bin/session record       # tmux "robot": the phone camera bridge in one window, the rig check in the other
```

**Check:** every line of the rig check says `ok`: the robot and leader ports, both calibrations, `camera1 wrist`, and
`camera2 scene /dev/video10`. Leave tmux with Ctrl-b d (the bridge keeps running); `lab/bin/session record` again
re-attaches.

Nothing in `lab/rigs/so101_v2.yaml` changes for this laptop: the arm ports are named by the arms' own USB chips
(`/dev/serial/by-id/...`) and the wrist camera by its model, so they are the same on every Linux laptop.

## 9. The model servers: A100, H200, or this laptop's GPU

**A100** (`b230968ee@192.168.5.13`): `lab` reaches it over SSH with a key, never a password. Two ways, pick one:
- **Add your key to the same account** (nothing private leaves the first laptop). Send your `~/.ssh/id_ed25519.pub`
  (step 3) to the first laptop's owner, who runs on the first laptop:

  ```bash
  ssh b230968ee@192.168.5.13 'cat >> ~/.ssh/authorized_keys' < id_ed25519.pub
  ```
- **Use the first laptop's key:** copy its key file to your laptop as `~/.ssh/id_a100` (`chmod 600`), and add to your
  `~/.ssh/config`:

  ```
  Host 192.168.5.13
      IdentityFile ~/.ssh/id_a100
  ```

**H200** (`jayaraj@192.168.41.118`, the professor's account): the same login as on the first laptop; `lab` asks for
its password once per connection.

**This laptop's GPU** (8 GB): ACT, Diffusion and SmolVLA only. Serve them here with `lab eval serve <eval> --models
...` and run with **`--server local`**. Not `--server laptop`: that always means the first laptop.

**Check:**

```bash
ssh -o BatchMode=yes b230968ee@192.168.5.13 hostname     # prints the A100's hostname, no password prompt
ssh jayaraj@192.168.41.118 hostname                       # asks the password, prints the H200's hostname
```

## 10. A first run that doesn't count

This runs the whole path once: a model served on this laptop's GPU, first on the replay robot (no motion), then a
test trial on the arm. Nothing is pushed or counted (`--replay` and `--test` keep it in `results/scratch/`).

Terminal 1, the model server (SmolVLA of `cubebox_v0_baseline`, ~1 GB on the GPU):

```bash
cd ~/Projects/drtc-major && source .venv/bin/activate
tmux new -s serve 'lab eval serve cubebox_v0_baseline --models smolvla_cubebox_v0_full; bash'
# wait for "ready" (the first time it downloads the model), then Ctrl-b d
```

Terminal 2, the robot side (arms, wrist camera and phone plugged in, step 8):

```bash
cd ~/Projects/drtc-major && source .venv/bin/activate
lab/bin/session record                                                                   # every line ok, Ctrl-b d
lab eval servers cubebox_v0_baseline --server local --models smolvla_cubebox_v0_full     # "ready"
lab eval run cubebox_v0_baseline --server local --models smolvla_cubebox_v0_full --replay
lab eval run cubebox_v0_baseline --server local --models smolvla_cubebox_v0_full --test  # the arm moves: 1 trial, then q
```

Afterwards: `tmux attach -t serve`, Ctrl-C, `exit`.

**Check:** the replay round ends without an error, and in the test trial the arm moves towards the cube. The keys
during a trial (`s`, `f`, `v`, `q`) are in [WORKFLOW.md Part 2, step 4](../WORKFLOW.md#4-run-the-session).

The laptop is ready. Official sessions use the commands of [WORKFLOW.md, "The plan, step by
step"](../WORKFLOW.md#the-plan-step-by-step), exactly as on the first laptop, with `--server h200`, `--server
a100`, or `--server local` for this laptop's own GPU.

## 11. Working together

- **The phone:** the first laptop owner's phone is the scene camera for every official trial and every recording. A
  different phone gives the models a different picture (field of view, colours) than the one they were trained on;
  any other phone is for practice and `--test` sessions only. Same for the arms and the wrist camera.
- **Official robot sessions run from one laptop** (the user's decision, 2026-11-22); this laptop is for recording,
  practice and `--test` sessions, or takes over that role.
- **One session at a time per eval.** Before each session `lab eval run` pulls every trial on the results store, from
  both laptops, and continues with the next round. Run sessions of one eval one after the other, never at the same
  time, and don't use `--offline` for official sessions (an unpushed session would not be seen by the other laptop).
- **`git pull` before each session.** Official sessions refuse a git tree with uncommitted changes; commit and push
  any change (e.g. an eval file) before running.
- **Who ran what:** sessions are stored as `<date>-laptop` or `<date>-laptop2`. Success counts (M1) are pooled; the
  report shows the timing metrics (M2-M6) separately per robot computer and server (METRICS.md rule 6). Keep each
  model on the same server for all its trials.
- **Calibration:** only on the first laptop, and only if an arm really needs it; then copy the files again (step 7).

## 12. Keeping it current

```bash
cd ~/Projects/drtc-major && git pull
git diff --stat HEAD@{1} -- uv.lock        # if uv.lock changed:
uv sync --frozen --extra gpu --extra sim
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `torch.cuda.is_available()` is `False`, or `NVIDIA driver too old` | the driver is older than 580: `sudo ubuntu-drivers install`, reboot (step 1) |
| `ssh -T git@github.com` hangs | port 22 blocked: the `~/.ssh/config` block of step 3 is missing |
| `git push`: permission denied | the collaborator invitation is not accepted yet, or the SSH key is not on your GitHub account |
| `rig so101_v2 on host (unknown: set LAB_HOST)` | the hostname in `lab/hosts.yaml` (step 6) doesn't match `hostname`; fix it, or `export LAB_HOST=laptop2` |
| robot or leader port `MISSING` | arm unplugged or unpowered; `ls /dev/serial/by-id/`; `Permission denied` on the port: not in `dialout` yet (log out and in, step 2) |
| `camera2 scene /dev/video10 MISSING` | `lab/bin/laptop-setup` not run, or no reboot since: `sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=PhoneCam exclusive_caps=1` |
| `waiting for phone (USB + 'Allow' prompt)` | unlock the phone and tap Allow; `adb devices` must show it as `device`, not `unauthorized` |
| `stream ended -- is DroidCam open...` | DroidCam is closed or another app shows its picture; open DroidCam, close other viewers |
| calibration fingerprints differ from the first laptop's | the wrong files were copied: repeat step 7 |
| `cannot connect to a100` | not on the campus network, or the SSH key is not set up (step 9) |
| `401` / `403` from the Hub | the token is missing or read-only: `hf auth login` with a write token (step 5) |
| `not served on local` | the server in terminal 1 has stopped or is still loading: `tmux attach -t serve` |
