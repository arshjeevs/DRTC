#!/bin/bash
# =============================================================================
# Start the DRTC policy server on the laptop GPU
# =============================================================================
#
# Usage:
#   ./scripts/lab/start_laptop_server.sh            # fps 60 (matches laptop_baseline.yaml)
#   FPS=30 ./scripts/lab/start_laptop_server.sh     # for models trained on 30 fps data
#
# Logs: logs/policy_server_<timestamp>.log (also printed to the terminal)
# =============================================================================

set -e

FPS="${FPS:-60}"
PORT="${PORT:-8080}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -x .venv/bin/python ]; then
    echo "ERROR: .venv missing. Run: uv venv --python 3.12 && uv pip install -e \".[smolvla,async,feetech]\""
    exit 1
fi

# GPU check
if ! .venv/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA not available to PyTorch (check nvidia-smi)."
    exit 1
fi

# Battery check: GPU throttles on battery and changes every latency number
for ac in /sys/class/power_supply/AC*/online /sys/class/power_supply/ADP*/online; do
    if [ -f "$ac" ] && [ "$(cat "$ac")" = "0" ]; then
        echo "WARNING: laptop is on battery -- plug in before running experiments."
    fi
done

# Port already in use?
if ss -tln | grep -q ":${PORT} "; then
    echo "ERROR: port ${PORT} already in use (another server running?). Stop it first: pkill -f policy_server_drtc"
    exit 1
fi

echo "=============================================="
echo "  DRTC policy server (laptop GPU)"
echo "=============================================="
echo "Addresses for the Pi (--remote-server-host):"
ip -4 -br addr | awk '$1 != "lo" {print "  " $1 ": " $3}'
echo "  (hotspot is usually 10.42.0.1)"
echo "Port: ${PORT}   FPS: ${FPS}"
echo "Note: after the Pi connects, model loading takes ~20-45 s before the arm moves."
echo ""

mkdir -p logs
LOG_FILE="logs/policy_server_$(date +%Y%m%d_%H%M%S).log"
PYTHONUNBUFFERED=1 .venv/bin/python examples/tutorial/async-inf/policy_server_drtc.py \
    --host 0.0.0.0 --port "$PORT" --fps "$FPS" --verbose-diagnostics 2>&1 | tee "$LOG_FILE"
