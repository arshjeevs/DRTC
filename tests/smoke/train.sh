#!/bin/bash
# Short test runs of the shortlisted models on LeRobot 0.7 (Phase 2), one model after another:
#   1. train 30 steps + one held-out evaluation (no Hub push, no logging), saving the step-30 checkpoint;
#   2. serve check on that checkpoint (tests/smoke/serve_check.py): LeRobot 0.7's policy server admits it (chunk,
#      plus rtc_guided for RTC policies) and the time per chunk is compared with DRTC's margin H - s_min.
# Confirms each recipe trains and serves in the new setup, and shows GPU memory and inference time.
#
#   H200, in the drtc_lab container:  cd /workspace/drtc_major && . ./.server.env && bash tests/smoke/train.sh [models]
#   Only the serve check, on checkpoints kept with KEEP=1:  SERVE_ONLY=1 bash tests/smoke/train.sh [models]
#
# Results: logs/smoke/<model>.log, logs/smoke/serve_<model>.json and logs/smoke/summary.txt. Deletes nothing but
# each test run's own output folder (outputs/train/<model>), only inside this repo, and only after the serve check
# (KEEP=1 keeps it). A model whose output folder already exists is skipped (it is not this script's).
set -u
here="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$here"
[ -f pyproject.toml ] && grep -q '^name = "drtc-lab"' pyproject.toml || { echo "not in the drtc-lab repo: $here"; exit 1; }
MODELS=${*:-act_cubebox_v1_full diffusion_cubebox_v1_full smolvla_cubebox_v1_full pi05_cubebox_v1_full \
        groot_cubebox_v1_full grootn17_cubebox_v1_full molmoact2_cubebox_v1_lora}
KEEP=${KEEP:-0}
SERVE_ONLY=${SERVE_ONLY:-0}
mkdir -p logs/smoke
summary=logs/smoke/summary.txt
echo "smoke test $(date -Is) commit $(git rev-parse --short HEAD)" >> "$summary"
for m in $MODELS; do
    out="outputs/train/$m"
    ckpt="$out/checkpoints/000030/pretrained_model"
    if [ "$SERVE_ONLY" != 1 ]; then
        if [ -e "$out" ]; then echo "$m: $out exists, skipping" | tee -a "$summary"; continue; fi
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        echo "== $m  start $(date +%T)  GPU free ${free} MiB" | tee -a "$summary"
        start=$(date +%s)
        .venv/bin/lab train "$m" --steps 30 --no-push --save_freq=30 --log_freq=10 --eval_steps=30 \
            --trackio.enable=false --num_workers=8 > "logs/smoke/$m.log" 2>&1
        code=$?
        {
            echo "   train exit $code after $(( $(date +%s) - start )) s"
            grep -E "eval_loss=" "logs/smoke/$m.log" | tail -1 | sed 's/^/   /'
            grep -oE "mem_gb:[0-9.]+" "logs/smoke/$m.log" | tail -1 | sed 's/^/   peak /'
            [ $code -ne 0 ] && grep -E "Error|error:" "logs/smoke/$m.log" | tail -3 | sed 's/^/   /'
        } | tee -a "$summary"
    fi
    if [ -d "$ckpt" ]; then
        .venv/bin/python tests/smoke/serve_check.py "$m" "$ckpt" > "logs/smoke/serve_$m.log" 2>&1
        echo "   serve exit $?: $(tail -1 "logs/smoke/serve_$m.log")" | tee -a "$summary"
    else
        echo "   serve: no checkpoint at $ckpt" | tee -a "$summary"
    fi
    # The test run's own output folder, inside this repo only.
    if [ "$KEEP" != 1 ] && [ "$SERVE_ONLY" != 1 ]; then
        case "$(realpath -m "$out")" in "$here"/outputs/train/*) rm -rf -- "$out" ;; esac
    fi
done
echo "done $(date -Is)" | tee -a "$summary"
