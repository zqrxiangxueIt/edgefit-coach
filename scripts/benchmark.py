"""Four-mode x rate-limit quantitative benchmark (M4-3).

For each streaming mode and each simulated link capacity, the receiver
applies a token-bucket throttle on incoming frame bytes: frames that do not
fit the bucket are counted as *undelivered* -- they model the frames a real
constrained uplink would never deliver.  A-side behaviour is unmodified
(its PUB socket streams regardless; there is no backpressure channel --
documented as future work).

Method notes (kept honest):
  * runs against the REAL cross-machine link (Pi -> PC over Tailscale);
  * the video source is a looped file, so scenarios are reproducible;
  * e2e uses the NTP-style offset estimated once over the control channel;
  * A-side compute (a_proc_ms, det_ms) comes straight from frame headers,
    so the numbers are the Pi's real measurements.

    .venv/Scripts/python scripts/benchmark.py --connect tcp://100.93.84.79:5555
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import zmq

from common.protocol import decode_frame
from server.pipeline import estimate_offset

MODES = ("raw", "event", "cascade", "privacy")


def control_endpoint(connect: str) -> str:
    host = connect.split("//")[1].rsplit(":", 1)[0]
    return f"tcp://{host}:{int(connect.rsplit(':', 1)[1]) + 1}"


def set_mode(ep: str, mode: str) -> dict:
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.RCVTIMEO, 3000)
    s.setsockopt(zmq.SNDTIMEO, 3000)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(ep)
    try:
        s.send_json({"cmd": "set_mode", "arg": {"mode": mode}})
        rep = s.recv_json()
    finally:
        s.close()
    if not str(rep.get("result", "")).startswith("ok"):
        # an error reply must fail the cell loudly: silently benchmarking the
        # OLD mode under the new label poisons the whole report
        raise RuntimeError(f"set_mode({mode}) rejected: {rep}")
    return rep


def run_cell(connect: str, mode: str, rate_kbps: float, duration: float,
             offset: float | None) -> dict:
    """Collect one (mode, rate) cell. rate 0 = unlimited."""
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"frame")
    sock.setsockopt(zmq.RCVHWM, 50)      # deeper queue: we throttle in software
    sock.connect(connect)
    time.sleep(0.3)
    set_mode(control_endpoint(connect), mode)
    time.sleep(1.5)                       # settle: mode switch + first frames
    # drain the queue: frames buffered during the settle sleep would poison
    # e2e (they are read ~1.5s after capture) and the delivered-fps median
    t_drain = time.time()
    while time.time() - t_drain < 0.5 and sock.poll(50):
        sock.recv_multipart()

    tokens = 0.0 if rate_kbps > 0 else float("inf")   # EMPTY bucket: no free
    last_refill = time.time()                          # burst, delivered<=rate
    t_start = last_refill
    t_end = t_start + duration

    n_sent = n_del = 0
    b_sent = b_del = 0
    e2e: deque = deque()
    a_proc: deque = deque()
    det_ms: deque = deque()

    while time.time() < t_end:
        if not sock.poll(500):
            continue
        parts = sock.recv_multipart()
        t_recv = time.time()
        header, payload = decode_frame(parts)
        size = len(payload)
        # refill bucket with elapsed bytes
        now = time.time()
        if rate_kbps > 0:
            tokens = min(rate_kbps * 1024.0, tokens + (now - last_refill) * rate_kbps * 1024.0)
        last_refill = now

        n_sent += 1
        b_sent += size
        a_proc.append(header.a_proc_ms)
        if header.extra.get("det_ms"):
            det_ms.append(header.extra["det_ms"])
        if offset is not None:
            e2e.append((t_recv - header.ts_capture + offset) * 1000)
        if size <= tokens:
            if rate_kbps > 0:
                tokens -= size
            n_del += 1
            b_del += size
    sock.close(0)

    wall = max(0.1, time.time() - t_start)   # true elapsed incl. poll gaps
    return {
        "mode": mode, "rate_kbps": rate_kbps,
        "sent_fps": round(n_sent / wall, 1),
        "sent_kbps": round(b_sent / wall / 1024, 1),
        "delivered_fps": round(n_del / wall, 1),
        "delivered_kbps": round(b_del / wall / 1024, 1),
        "drop_pct": round(100 * (1 - n_del / n_sent), 1) if n_sent else 0.0,
        "e2e_p50_ms": round(float(np.percentile(e2e, 50)), 1) if e2e else None,
        "a_proc_ms": round(float(np.mean(a_proc)), 1) if a_proc else None,
        "det_ms": round(float(np.mean(det_ms)), 1) if det_ms else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp://100.93.84.79:5555")
    ap.add_argument("--rates", default="0,2048,512,128",
                    help="simulated link capacities in KB/s, comma separated (0=unlimited)")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--scenario", default="curl_a.mp4 (person continuously present)")
    ap.add_argument("--out", default="docs/benchmark_data.json")
    args = ap.parse_args()

    rates = [float(r) for r in args.rates.split(",")]
    modes = args.modes.split(",")

    res = estimate_offset(control_endpoint(args.connect))
    offset = res[0] if res else None
    print(f"offset: {offset*1000:+.1f} ms" if offset is not None else "offset: FAILED (e2e n/a)")

    cells = []
    for mode in modes:
        for rate in rates:
            cell = run_cell(args.connect, mode, rate, args.duration, offset)
            cells.append(cell)
            print(f"{mode:8s} @ {rate if rate else '∞':>6} KB/s | "
                  f"sent {cell['sent_fps']:5.1f}fps {cell['sent_kbps']:7.1f}KB/s | "
                  f"delivered {cell['delivered_fps']:5.1f}fps {cell['delivered_kbps']:7.1f}KB/s "
                  f"drop {cell['drop_pct']:5.1f}% | e2e {cell['e2e_p50_ms']} ms | "
                  f"a_proc {cell['a_proc_ms']} ms"
                  + (f" det {cell['det_ms']} ms" if cell["det_ms"] else ""),
                  flush=True)

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"scenario": args.scenario, "cells": cells}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print(f"\ndata -> {args.out}")


if __name__ == "__main__":
    main()
