#!/bin/bash
# =============================================================================
# Start a lab session on the Raspberry Pi (run this after SSH-ing in)
# =============================================================================
#
# Opens (or re-attaches to) a tmux session "robot" with:
#   window 0 "phonecam": scripts/lab/start_phone_camera.sh (auto-reconnecting)
#   window 1 "work":     venv activated, preflight checks run automatically
#
# tmux survives SSH disconnects: reconnect and run this script again.
# Keys: Ctrl+b then n/p = next/prev window, Ctrl+b then d = detach.
# =============================================================================

SESSION=robot
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    exec tmux attach -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" -n phonecam -c "$PROJECT_ROOT"
tmux send-keys -t "$SESSION:phonecam" "./scripts/lab/start_phone_camera.sh" C-m

tmux new-window -t "$SESSION" -n work -c "$PROJECT_ROOT"
# Give the phone bridge a few seconds before checking cameras
tmux send-keys -t "$SESSION:work" \
    "source .venv/bin/activate && sleep 8 && python scripts/lab/preflight.py" C-m

exec tmux attach -t "$SESSION:work"
