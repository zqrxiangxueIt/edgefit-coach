#!/usr/bin/env bash
# Estimate A-clock minus B-clock offset (seconds) via SSH round-trip midpoint.
# Uses ControlMaster so the measured RTT is the muxed-connection round trip
# (~tens of ms), not a fresh connection+auth (~1s) which would poison the
# midpoint.  Accuracy ~ ±RTT/2 - good enough for the e2e latency display.
set -euo pipefail
PI_USER="${PI_USER:-saltz}"
PI_HOST="${PI_HOST:-100.93.84.79}"
PI_KEY="${PI_KEY:-/c/Users/ZHENG QIAORI/.ssh/id_rsa}"
CM="/tmp/edgefit-ssh-%r@%h:%p"

SSHCMD=(ssh -o BatchMode=yes -o ConnectTimeout=8 -i "$PI_KEY"
        -o ControlMaster=auto -o ControlPath="$CM" -o ControlPersist=10)

"${SSHCMD[@]}" "$PI_USER@$PI_HOST" true   # warm up the multiplexed connection

t_before=$(python -c "import time; print(time.time())")
t_pi=$("${SSHCMD[@]}" "$PI_USER@$PI_HOST" 'python3 -c "import time; print(time.time())"')
t_after=$(python -c "import time; print(time.time())")

python - "$t_before" "$t_pi" "$t_after" <<'EOF'
import sys
b, p, a = map(float, sys.argv[1:4])
offset = p - (a + b) / 2          # A-clock minus B-clock
print(f"RTT {(a-b)*1000:.1f} ms | A-B offset = {offset:+.3f} s")
print(f"pass to receiver:  --clock-offset {offset:.3f}")
EOF
