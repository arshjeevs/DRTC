#!/bin/bash
# =============================================================================
# One-time Raspberry Pi setup for the lab (SO101 + Brio + DroidCam phone camera)
# =============================================================================
#
# Safe to re-run. Does:
#   1. apt packages: adb, ffmpeg, v4l2loopback, v4l-utils, tmux, chrony
#   2. v4l2loopback loads at every boot as /dev/video10 ("PhoneCam")
#   3. chrony syncs the Pi clock to the laptop (hotspot 10.42.0.1)
#   4. Python venv + project install (client extras only)
#
# Usage (on the Pi, from the repo root):
#   ./scripts/lab/pi_setup_once.sh
#
# Environment variables:
#   LAPTOP_IP   - Laptop address used as time server (default: 10.42.0.1)
# =============================================================================

set -e

LAPTOP_IP="${LAPTOP_IP:-10.42.0.1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "[1/4] Installing system packages..."
sudo apt update
sudo apt install -y adb ffmpeg v4l-utils tmux chrony v4l2loopback-dkms v4l2loopback-utils \
    "linux-headers-$(uname -r)"

echo "[2/4] Making /dev/video10 (PhoneCam) load at boot..."
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf >/dev/null
echo 'options v4l2loopback devices=1 video_nr=10 card_label="PhoneCam" exclusive_caps=1' \
    | sudo tee /etc/modprobe.d/v4l2loopback.conf >/dev/null
if [ ! -e /dev/video10 ]; then
    sudo modprobe v4l2loopback
fi
ls -l /dev/video10

echo "[3/4] Syncing clock to the laptop ($LAPTOP_IP) with chrony..."
CHRONY_CONF=/etc/chrony/chrony.conf
if ! grep -q "server $LAPTOP_IP" "$CHRONY_CONF"; then
    echo "server $LAPTOP_IP iburst prefer" | sudo tee -a "$CHRONY_CONF" >/dev/null
    # Allow a large initial correction (the Pi has no RTC battery)
    echo "makestep 1 -1" | sudo tee -a "$CHRONY_CONF" >/dev/null
fi
sudo systemctl restart chrony
echo "      (clock sync only works once the laptop runs chrony as a server -- see docs/LAB_SETUP.md)"

echo "[4/4] Python environment..."
cd "$PROJECT_ROOT"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
if [ ! -d .venv ]; then
    uv venv --python 3.12
fi
uv pip install -e ".[async,feetech]"

echo ""
echo "Done. Next: follow 'Every session' in docs/LAB_SETUP.md (start with ./scripts/lab/pi_session.sh)."
