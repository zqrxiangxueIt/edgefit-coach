"""Edge server B console frontend: run the shared StreamingPipeline and
print stats / rep events to the terminal (optionally record mp4).

The streaming loop itself lives in server/pipeline.py -- the web dashboard
(server/app.py) runs the exact same code.

    python -m server.receiver --connect tcp://100.93.84.79:5555 --backend dml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="edge server B receiver (console)")
    ap.add_argument("--connect", default="tcp://100.93.84.79:5555")
    ap.add_argument("--model", default="models/yolo11n-pose.onnx")
    ap.add_argument("--backend", default="dml", choices=["dml", "openvino", "cpu"])
    ap.add_argument("--clock-offset", type=float, default=None,
                    help="A-clock minus B-clock (s); omit = auto-estimate")
    ap.add_argument("--record", default="", help="optional annotated mp4 output")
    ap.add_argument("--stats-every", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after N seconds (0=until Ctrl+C)")
    args = ap.parse_args()

    from server.pipeline import StreamingPipeline

    pipe = StreamingPipeline(args.connect, args.model, args.backend,
                             clock_offset=args.clock_offset,
                             stats_every=min(args.stats_every, 2.0))
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        pipe.record_to(args.record)
    pipe.start()

    t0 = time.time()
    try:
        while pipe.is_alive():
            if args.duration and time.time() - t0 > args.duration:
                print("\n[server] duration reached")
                break
            time.sleep(args.stats_every)
            s = pipe.status()
            e2e = f"{s.e2e_p50_ms:5.0f} ms" if s.e2e_p50_ms else "  n/a"
            print(f"[server] rx {s.rx_fps:5.1f} fps  {s.kbps:7.1f} KB/s  "
                  f"loss {s.loss}  a_eff {s.a_eff_fps:5.1f} fps  "
                  f"infer p50 {s.infer_p50_ms:5.1f} / p95 {s.infer_p95_ms:5.1f} ms  "
                  f"e2e p50 {e2e}  reps={s.reps} (suppressed {s.suppressed}) "
                  f"mode={s.mode}", flush=True)
            if pipe.error:
                raise pipe.error
    except KeyboardInterrupt:
        print("\n[server] interrupted")
    finally:
        pipe.stop()
        pipe.join(timeout=5)
        s = pipe.status()
        print(f"[server] total reps: {s.reps} "
              f"(full {s.full_reps} / half {s.half_reps})")


if __name__ == "__main__":
    main()
