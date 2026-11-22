"""Edge server B main: subscribe -> decode -> pose -> count -> report.

M1 focus: the cross-machine link.  Reports per-interval throughput,
bandwidth, B-side processing latency (recv -> analysed), A-side effective
FPS (from header seq/capture-timestamp intervals), and rep events.
The web dashboard replaces the console reporting in M3.

    python -m server.receiver --connect tcp://100.93.84.79:5555 --backend dml
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import zmq

from common.protocol import decode_frame
from server.analysis.curl_counter import CurlCounter
from server.analysis.filters import KeypointSmoother
from server.inference.pose_engine import PoseEngine


def estimate_offset(control_endpoint: str, n: int = 5) -> tuple[float, float] | None:
    """NTP-style clock offset (A-clock minus B-clock) over the ZMQ control
    channel.  Returns (offset_seconds, min_rtt_ms), best sample by RTT;
    None when the channel is dead (caller must NOT pretend offset is 0 --
    on the real pair A-B is ~-779ms, so a fake 0 poisons every e2e number).
    """
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.RCVTIMEO, 2000)
    s.setsockopt(zmq.SNDTIMEO, 2000)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(control_endpoint)
    best: tuple[float, float] | None = None
    try:
        for _ in range(n):
            t0 = time.time()
            try:
                s.send_json({"cmd": "ping", "t0": t0})
                rep = s.recv_json()
            except zmq.ZMQError:
                break
            t3 = time.time()
            if rep.get("echo", {}).get("cmd") == "ping":
                rtt, off = (t3 - t0) * 1000, rep["t_reply"] - (t0 + t3) / 2
                if best is None or rtt < best[1]:
                    best = (off, rtt)
            time.sleep(0.05)
    finally:
        s.close()
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="edge server B receiver")
    ap.add_argument("--connect", default="tcp://100.93.84.79:5555")
    ap.add_argument("--model", default="models/yolo11n-pose.onnx")
    ap.add_argument("--backend", default="dml", choices=["dml", "openvino", "cpu"])
    ap.add_argument("--clock-offset", type=float, default=0.0,
                    help="A-clock minus B-clock (s); 0 = auto-estimate via control channel")
    ap.add_argument("--record", default="", help="optional annotated mp4 output")
    ap.add_argument("--stats-every", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after N seconds (0=until Ctrl+C); Windows-friendly testing")
    args = ap.parse_args()

    offset: float | None = args.clock_offset
    if args.clock_offset == 0.0:
        control_ep = f"tcp://{args.connect.split('//')[1].rsplit(':', 1)[0]}:{int(args.connect.rsplit(':', 1)[1]) + 1}"
        res = estimate_offset(control_ep)
        if res is None:
            print("[server] WARN: clock-offset estimation failed; "
                  "e2e will be reported as n/a")
            offset = None
        else:
            offset, rtt = res
            print(f"[server] clock offset A-B = {offset*1000:+.1f} ms (min RTT {rtt:.1f} ms)")

    engine = PoseEngine(args.model, backend=args.backend)
    smoother = KeypointSmoother()
    counter = CurlCounter()
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"frame")
    sock.setsockopt(zmq.RCVHWM, 10)       # real-time: drop stale rather than lag
    sock.connect(args.connect)
    print(f"[server] connecting {args.connect} backend={args.backend} "
          f"model={Path(args.model).name}")

    writer = None
    # metric window sized to match the stats interval (~stats_every * fps,
    # with headroom) so percentiles don't mix pre/post mode-switch samples (N4)
    mlen = max(60, int(args.stats_every * 40))
    infer_ms: deque = deque(maxlen=mlen)
    e2e_ms: deque = deque(maxlen=mlen)     # capture->receive; None if no offset
    t_stat = time.time()
    n_stat = b_stat = 0
    first_seq = last_seq = None
    last_capture_t = None
    t_begin = time.time()
    capture_iv: deque = deque(maxlen=mlen)  # A-side inter-frame intervals

    try:
        while True:
            if args.duration and time.time() - t_begin > args.duration:
                print("\n[server] duration reached")
                break
            parts = sock.recv_multipart(flags=zmq.NOBLOCK) if sock.poll(1000) else None
            if parts is None:
                print("[server] waiting for frames ...")
                continue
            t_recv = time.time()
            header, payload = decode_frame(parts)
            frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            t0 = time.perf_counter()
            results = engine.detect(frame)
            infer_ms.append((time.perf_counter() - t0) * 1000)

            if results:
                kps = smoother.update(results[0].keypoints)
                for ev in counter.update(kps):
                    print(f"  [rep {ev.index}][{ev.side}] full={ev.full} "
                          f"min_angle={ev.min_angle:.0f}")
            else:
                smoother.update(np.zeros((17, 3), np.float32))
                counter.update(np.zeros((17, 3), np.float32))

            # ---- metrics bookkeeping ----
            n_stat += 1
            b_stat += len(payload)
            if first_seq is None:
                first_seq = header.seq
            last_seq = header.seq
            if last_capture_t is not None:
                capture_iv.append(header.ts_capture - last_capture_t)
            last_capture_t = header.ts_capture
            # e2e = (B_recv - A_capture) + (A-clock - B-clock): see estimate_offset
            if offset is not None:
                e2e_ms.append((t_recv - header.ts_capture + offset) * 1000)

            if args.record and results:
                if writer is None:
                    # estimate fps from observed A-side inter-frame intervals
                    # (N7: hardcoded 15 played back 25/30fps streams too fast)
                    est_fps = (1.0 / np.median(capture_iv)
                               if len(capture_iv) >= 5 else 25.0)
                    writer = cv2.VideoWriter(
                        args.record, cv2.VideoWriter_fourcc(*"mp4v"), est_fps,
                        (frame.shape[1], frame.shape[0]))
                    print(f"[server] recording at {est_fps:.1f} fps -> {args.record}")
                from scripts.offline_pipeline import draw
                writer.write(draw(frame, results[0], counter, kps))

            now = time.time()
            if now - t_stat >= args.stats_every:
                dt = now - t_stat
                got = (last_seq - first_seq + 1
                       if last_seq is not None and first_seq is not None else 0)
                loss = max(0, got - n_stat)       # frames A sent, B never saw (S4)
                loss_pct = f" ({100*loss/got:.0f}%)" if got else ""
                iv = np.mean(capture_iv) if capture_iv else float("nan")
                e2e_txt = (f"{np.percentile(e2e_ms, 50):5.0f} ms" if e2e_ms
                           else "  n/a")
                print(f"[server] rx {n_stat/dt:5.1f} fps  {b_stat/dt/1024:7.1f} KB/s  "
                      f"loss {loss}{loss_pct}  "
                      f"a_eff {1/iv if iv and iv > 0 else float('nan'):5.1f} fps "
                      f"(src gap {iv*1000 if iv==iv else 0:4.0f} ms)  "
                      f"infer p50 {np.percentile(infer_ms, 50):5.1f} / "
                      f"p95 {np.percentile(infer_ms, 95):5.1f} ms  "
                      f"e2e p50 {e2e_txt}  "
                      f"reps={counter.stats.reps} (suppressed {counter.stats.phantom_reps_suppressed})",
                      flush=True)
                t_stat, n_stat, b_stat = time.time(), 0, 0
                first_seq = None
    except KeyboardInterrupt:
        print("\n[server] interrupted")
    finally:
        if writer is not None:
            writer.release()
        sock.close(0)
        ctx.term()
        print(f"[server] total reps: {counter.stats.reps} "
              f"(full {counter.stats.full_reps} / half {counter.stats.half_reps})")


if __name__ == "__main__":
    main()
