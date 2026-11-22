"""M0-3 smoke test: verify DirectML inference works on the AMD GPU.

Usage:
    .venv/Scripts/python scripts/smoke_dml.py [model.onnx]

Loads the pose ONNX model, creates a DirectML (or fallback CPU) session,
runs 50 warmup + 100 timed inferences on a random 640x640 input, and prints
the effective providers and per-frame latency.
"""

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL = Path(sys.argv[1] if len(sys.argv) > 1 else "models/yolo11n-pose.onnx")


def main() -> None:
    print(f"onnxruntime {ort.__version__}, available providers:")
    for p in ort.get_available_providers():
        print(f"  - {p}")

    providers = ["DmlExecutionProvider"] if "DmlExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    print(f"\nusing: {providers}")

    sess = ort.InferenceSession(str(MODEL), providers=providers)
    inp = sess.get_inputs()[0]
    name, shape = inp.name, [d if isinstance(d, int) else 1 for d in inp.shape]
    print(f"model input: {name} {shape}")

    x = np.random.rand(*shape).astype(np.float32)

    for _ in range(50):  # warmup: first DML calls include kernel compilation
        sess.run(None, {name: x})

    t0 = time.perf_counter()
    N = 100
    for _ in range(N):
        sess.run(None, {name: x})
    dt = (time.perf_counter() - t0) / N * 1000
    print(f"\ninference: {dt:.2f} ms/frame  (~{1000/dt:.1f} FPS ceiling)")


if __name__ == "__main__":
    main()
