#!/usr/bin/env bash
# Run FROM the PC (Git Bash): push edge code + test assets to the Pi over
# Tailscale SSH, then optionally start the streamer.
#
#   bash scripts/deploy_pi.sh            # sync only
#   bash scripts/deploy_pi.sh --run ...  # sync + run edge_main.py (args after --run)
set -euo pipefail

PI_USER="${PI_USER:-saltz}"
PI_HOST="${PI_HOST:-100.93.84.79}"
PI_KEY="${PI_KEY:-/c/Users/ZHENG QIAORI/.ssh/id_rsa}"
PI_DIR="${PI_DIR:-edgefit}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -i "$PI_KEY")

echo ">> syncing code to $PI_USER@$PI_HOST:~/$PI_DIR"
ssh "${SSH_OPTS[@]}" "$PI_USER@$PI_HOST" "mkdir -p ~/$PI_DIR/scripts ~/$PI_DIR/models"
scp "${SSH_OPTS[@]}" -r common edge requirements-edge.txt "$PI_USER@$PI_HOST:~/$PI_DIR/"
scp "${SSH_OPTS[@]}" scripts/setup_pi.sh "$PI_USER@$PI_HOST:~/$PI_DIR/scripts/"
[[ -f models/detect.tflite ]] && scp "${SSH_OPTS[@]}" models/detect.tflite \
    "$PI_USER@$PI_HOST:~/$PI_DIR/models/" || echo "!! models/detect.tflite missing locally"

if [[ ! -f videos/curl_a.mp4 ]]; then
    echo "!! local videos/curl_a.mp4 missing; skip asset sync (fine if already on Pi)"
else
    echo ">> syncing test video (9.5MB)"
    ssh "${SSH_OPTS[@]}" "$PI_USER@$PI_HOST" "mkdir -p ~/$PI_DIR/videos"
    scp "${SSH_OPTS[@]}" videos/curl_a.mp4 "$PI_USER@$PI_HOST:~/$PI_DIR/videos/"
fi

if [[ "${1:-}" == "--run" ]]; then
    shift
    echo ">> starting edge_main.py with args: $*"
    ssh "${SSH_OPTS[@]}" "$PI_USER@$PI_HOST" "cd ~/$PI_DIR && .venv/bin/python -u -m edge.edge_main $*"
else
    echo "deploy_pi: synced. First time on this Pi? run setup first:"
    echo "  ssh $PI_USER@$PI_HOST 'cd ~/$PI_DIR && bash scripts/setup_pi.sh'"
    echo "then start manually:"
    echo "  ssh $PI_USER@$PI_HOST 'cd ~/$PI_DIR && .venv/bin/python -u -m edge.edge_main --help'"
fi
