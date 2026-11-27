#!/bin/bash
# Phase 3 baseline: training speed of the shortlisted models on LeRobot 0.7 as-is (their recipes, one GPU).
# Per model: `lab train` for $STEPS steps (default 200) with no evaluation, no checkpoint, no logging service, then
# tests/bench/speed_report.py averages the logged step times after $WARMUP steps (default 50) and projects the
# recipe's full run. nvidia-smi samples GPU utilization once a second during the run.
#
#   H200, in the drtc_lab container:  cd /workspace/drtc_major && . ./.server.env && bash tests/bench/train_speed.sh [models]
#   Options (environment): STEPS=200 WARMUP=50 WORKERS=8 LABEL=baseline, EXTRA="--flag=value ..." (passed to
#   lab train, e.g. to compare a setting; use a different LABEL for it).
#
# Results: logs/bench/<label>/<model>.log, speed_<model>.json, and logs/bench/<label>/summary.txt. Deletes nothing
# but each run's own output folder (outputs/train/<model>), only inside this repo. A model whose output folder
# already exists is skipped (it is not this script's).
set -u
here="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$here"
[ -f pyproject.toml ] && grep -q '^name = "drtc-lab"' pyproject.toml || { echo "not in the drtc-lab repo: $here"; exit 1; }
MODELS=${*:-act_cubebox_v1_full diffusion_cubebox_v1_full smolvla_cubebox_v1_full pi05_cubebox_v1_full \
        groot_cubebox_v1_full grootn17_cubebox_v1_full molmoact2_cubebox_v1_lora}
STEPS=${STEPS:-200}
WARMUP=${WARMUP:-50}
WORKERS=${WORKERS:-8}
LABEL=${LABEL:-baseline}
EXTRA=${EXTRA:-}
dir="logs/bench/$LABEL"
mkdir -p "$dir"
summary="$dir/summary.txt"
{
    echo "train speed $(date -Is) commit $(git rev-parse --short HEAD) label $LABEL steps $STEPS warmup $WARMUP workers $WORKERS"
    echo "   $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1); CPUs $(nproc)"
    [ -n "$EXTRA" ] && echo "   extra: $EXTRA"
} | tee -a "$summary"
for m in $MODELS; do
    out="outputs/train/$m"
    if [ -e "$out" ]; then echo "$m: $out exists, skipping" | tee -a "$summary"; continue; fi
    echo "== $m  start $(date +%T)" | tee -a "$summary"
    nvidia-smi --query-gpu=timestamp,utilization.gpu --format=csv,noheader,nounits -i 0 -l 1 > "$dir/gpu_$m.csv" &
    smi=$!
    start=$(date +%s)
    # shellcheck disable=SC2086
    .venv/bin/lab train "$m" --steps "$STEPS" --no-push --save_checkpoint=false --eval_steps=0 --log_freq=10 \
        --trackio.enable=false --num_workers="$WORKERS" $EXTRA > "$dir/$m.log" 2>&1
    code=$?
    kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
    echo "   train exit $code after $(( $(date +%s) - start )) s" | tee -a "$summary"
    if [ $code -eq 0 ]; then
        .venv/bin/python tests/bench/speed_report.py "$m" "$dir/$m.log" --gpu-csv "$dir/gpu_$m.csv" \
            --warmup "$WARMUP" --out "$dir" 2>&1 | tail -1 | sed 's/^/   /' | tee -a "$summary"
    else
        grep -E "Error|error:" "$dir/$m.log" | tail -3 | sed 's/^/   /' | tee -a "$summary"
    fi
    case "$(realpath -m "$out")" in "$here"/outputs/train/*) rm -rf -- "$out" ;; esac
done
echo "done $(date -Is)" | tee -a "$summary"
