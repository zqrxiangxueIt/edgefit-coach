#!/usr/bin/env bash
# Run ON the Raspberry Pi (M1): create venv and install edge dependencies.
#   bash setup_pi.sh [--with-litert]   (--with-litert adds ai-edge-litert for M4 cascade)
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root on the Pi

python3 -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -q opencv-python-headless pyzmq numpy

if [[ "${1:-}" == "--with-litert" ]]; then
    .venv/bin/pip install -q ai-edge-litert
fi

.venv/bin/python - <<'EOF'
import cv2, zmq, numpy
print("OK  cv2", cv2.__version__, "| pyzmq", zmq.__version__, "| numpy", numpy.__version__)
try:
    import ai_edge_litert  # noqa: F401
    print("OK  ai-edge-litert")
except ImportError:
    print("--  ai-edge-litert not installed (fine until M4)")
EOF
echo "setup_pi: done"
