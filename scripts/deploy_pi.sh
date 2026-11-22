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
ssh "${SSH_OPTS[@]}" "$PI_USER@$PI_HOST" "mkdir -p ~/$PI_DIR"
scp "${SSH_OPTS[@]}" -r common edge requirements-edge.txt "$PI_USER@$PI_HOST:~/$PI_DIR/"

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
    ssh "${SSH_OPTS[@]}" "$PI_USER@$PI_HOST" "cd ~/$PI_DIR && .venv/bin/python -m edge.edge_main $*"
else
    echo "deploy_pi: synced. Start manually:"
    echo "  ssh $PI_USER@$PI_HOST 'cd ~/$PI_DIR && .venv/bin/python -m edge.edge_main --help'"
fi
