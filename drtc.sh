#!/bin/bash
# Usage: ./drtc.sh build | up | in [name] | ls | kill <name> | down | status
IMAGE=drtc-train:$USER
NAME=drtc_$USER
REPO=$(cd "$(dirname "$0")" && pwd)
ENVFILE=$HOME/.drtc.env

case "$1" in
  build)
    docker build -f "$REPO/Dockerfile.train" -t "$IMAGE" "$REPO" ;;
  up)
    [ -e "$REPO/.venv" ] || [ -L "$REPO/.venv" ] || ln -s /opt/venv "$REPO/.venv"
    mkdir -p "$HOME/.cache/huggingface"; touch "$ENVFILE"
    if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
      docker start "$NAME" >/dev/null
    else
      docker run -d --name "$NAME" --init \
        --gpus all --shm-size=32g \
        --user "$(id -u):$(id -g)" -e HOME=/tmp \
        --env-file "$ENVFILE" \
        -v "$REPO":/opt/drtc \
        -v "$HOME/.cache/huggingface":/cache/huggingface \
        "$IMAGE" >/dev/null
    fi
    echo "Container $NAME is running. Enter with: ./drtc.sh in [name]" ;;
  in)
    docker exec -it "$NAME" tmux new -A -s "${2:-main}" ;;
  ls)
    docker exec "$NAME" tmux ls 2>/dev/null || echo "No sessions running" ;;
  kill)
    [ -z "$2" ] && { echo "Usage: $0 kill <name>"; exit 1; }
    docker exec "$NAME" tmux kill-session -t "$2" ;;
  down)
    docker rm -f "$NAME" ;;
  status)
    docker ps -a --filter name="$NAME"
    echo; docker exec "$NAME" tmux ls 2>/dev/null
    echo; nvidia-smi ;;
  *)
    echo "Usage: $0 {build|up|in [name]|ls|kill <name>|down|status}" ;;
esac
