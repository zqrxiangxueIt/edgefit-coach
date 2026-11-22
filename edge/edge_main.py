"""Edge device A main: capture -> preprocess -> publish over ZMQ.

Runs on the Raspberry Pi (or loopback on the PC for testing):

    # loopback on PC (two shells)
    python -m edge.edge_main --bind tcp://127.0.0.1:5555 --video videos/curl_a.mp4
    python -m server.receiver --connect tcp://127.0.0.1:5555

    # on the Pi (deployed by scripts/deploy_pi.sh)
    .venv/bin/python -m edge.edge_main --bind tcp://0.0.0.0:5555 --video videos/curl_a.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import zmq

from common.protocol import FrameHeader, encode_frame
from edge.preprocess import RawPreprocessor
from edge.sources import CameraSource, VideoFileSource


def build_source(args):
    if args.source == "file":
        return VideoFileSource(args.video, realtime=True)
    return CameraSource(args.camera, args.width, args.height, args.fps)


def main() -> None:
    ap = argparse.ArgumentParser(description="edge device A streamer")
    ap.add_argument("--mode", default="raw", choices=["raw"])  # event/cascade/privacy land in M2/M4
    ap.add_argument("--source", default="file", choices=["file", "camera"])
    ap.add_argument("--video", default="videos/curl_a.mp4")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bind", default="tcp://0.0.0.0:5555")
    ap.add_argument("--sample-every", type=int, default=1)
    ap.add_argument("--max-width", type=int, default=960)
    ap.add_argument("--jpeg-quality", type=int, default=75)
    ap.add_argument("--loop", action="store_true", help="loop file source forever")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0=never)")
    args = ap.parse_args()

    src = build_source(args)
    pre = RawPreprocessor(args.sample_every, args.max_width, args.jpeg_quality)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 10)       # real-time: never queue stale frames
    sock.bind(args.bind)
    # control REP socket (one port up): answers NTP-style pings for clock
    # offset estimation; mode-switch commands ride the same socket from M3 on.
    ctrl = ctx.socket(zmq.REP)
    ctrl.bind(f"tcp://*:{int(args.bind.split(':')[-1]) + 1}")
    print(f"[edge] mode={args.mode} source={args.source} bind={args.bind} "
          f"({args.max_width}px, q{args.jpeg_quality}, sample_every={args.sample_every})")

    seq = 0
    t_start = time.time()
    t_stat, n_stat, b_stat = time.time(), 0, 0
    try:
        while True:
            if args.duration and time.time() - t_start > args.duration:
                break
            item = src.read()
            if item is None:
                if args.source == "file" and args.loop:
                    src.close()
                    src = build_source(args)
                    continue
                break
            frame, t_capture = item
            pf = pre.process(frame)
            if pf is None:
                continue
            seq += 1
            header = FrameHeader(seq=seq, ts_capture=t_capture, mode=args.mode,
                                 w=pf.w, h=pf.h, jpeg_q=pf.jpeg_q,
                                 motion=pf.motion, a_proc_ms=pf.a_proc_ms,
                                 extra=pf.extra)
            msg = encode_frame(header, pf.jpeg)
            sock.send_multipart(msg)
            if ctrl.poll(0):                    # serve clock pings / control
                req = ctrl.recv_json()
                ctrl.send_json({"t_reply": time.time(),
                                "mode": args.mode, "seq": seq, "echo": req})
            n_stat += 1
            b_stat += len(pf.jpeg)

            now = time.time()
            if now - t_stat >= 5.0:
                dt = now - t_stat
                print(f"[edge] sent {n_stat/dt:5.1f} fps  {b_stat/dt/1024:7.1f} KB/s  "
                      f"a_proc {pf.a_proc_ms:.1f} ms  seq={seq}")
                t_stat, n_stat, b_stat = now, 0, 0
    except KeyboardInterrupt:
        print("\n[edge] interrupted")
    finally:
        src.close()
        sock.close(0)
        ctrl.close(0)
        ctx.term()
        print(f"[edge] done, {seq} frames in {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
