#!/bin/bash
# =============================================================================
# Run a DRTC experiment from the Pi against the laptop policy server
# =============================================================================
#
# Usage:
#   ./scripts/lab/run_experiment.sh                       # laptop_baseline
#   ./scripts/lab/run_experiment.sh drop_obs_multiple     # any config name or .yaml path
#   ./scripts/lab/run_experiment.sh my.yaml --experiment_name trial_03
#
# Environment variables:
#   LAPTOP_IP - Policy server host (default: 10.42.0.1, the laptop hotspot)
# =============================================================================

set -e

LAPTOP_IP="${LAPTOP_IP:-10.42.0.1}"
CONFIG="${1:-examples/experiments/configs/laptop_baseline.yaml}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

if ! timeout 2 bash -c "</dev/tcp/${LAPTOP_IP}/8080" 2>/dev/null; then
    echo "ERROR: policy server not reachable at ${LAPTOP_IP}:8080."
    echo "       Start ./scripts/lab/start_laptop_server.sh on the laptop and check the Pi is on its hotspot."
    exit 1
fi

exec ./scripts/run_drtc_experiment_with_remote_server.sh \
    --remote-server-host "$LAPTOP_IP" --config "$CONFIG" "$@"
