#!/bin/bash
# Short test trainings of the planned recipes on LeRobot 0.7: 30 steps + one held-out evaluation each, no
# checkpoints, no Hub push, no logging. Confirms each recipe trains in the new setup and shows its GPU memory.
#
#   H200, in the drtc_lab container:  cd /workspace/drtc_major && . ./.server.env && bash tests/smoke/train.sh [models]
#
# Results: logs/smoke/<model>.log and logs/smoke/summary.txt. Deletes nothing but each test run's own output folder
# (outputs/train/<model>), and only inside this repo.
set -u
here="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$here"
[ -f pyproject.toml ] && grep -q '^name = "drtc-lab"' pyproject.toml || { echo "not in the drtc-lab repo: $here"; exit 1; }
MODELS=${*:-act_cubebox_v1_full diffusion_cubebox_v1_full smolvla_cubebox_v1_full xvla_cubebox_v1_full \
        pi0fast_cubebox_v1_full pi0_cubebox_v1_full pi05_cubebox_v1_full groot_cubebox_v1_full}
mkdir -p logs/smoke
summary=logs/smoke/summary.txt
echo "smoke test $(date -Is) commit $(git rev-parse --short HEAD)" > "$summary"
for m in $MODELS; do
    out="outputs/train/$m"
    if [ -e "$out" ]; then echo "$m: $out exists, skipping" | tee -a "$summary"; continue; fi
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    echo "== $m  start $(date +%T)  GPU free ${free} MiB" | tee -a "$summary"
    start=$(date +%s)
    .venv/bin/lab train "$m" --steps 30 --no-push --save_checkpoint=false --log_freq=10 --eval_steps=30 \
        --trackio.enable=false --num_workers=8 > "logs/smoke/$m.log" 2>&1
    code=$?
    {
        echo "   exit $code after $(( $(date +%s) - start )) s"
        grep -E "eval_loss=" "logs/smoke/$m.log" | tail -1 | sed 's/^/   /'
        grep -oE "mem_gb:[0-9.]+" "logs/smoke/$m.log" | tail -1 | sed 's/^/   peak /'
        [ $code -ne 0 ] && grep -E "Error|error:" "logs/smoke/$m.log" | tail -3 | sed 's/^/   /'
    } | tee -a "$summary"
    # The test run's own output folder, inside this repo only.
    case "$(realpath -m "$out")" in "$here"/outputs/train/*) rm -rf -- "$out" ;; esac
done
echo "done $(date -Is)" | tee -a "$summary"
