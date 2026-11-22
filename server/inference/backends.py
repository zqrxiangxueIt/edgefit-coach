"""Unified inference session factory: DirectML / OpenVINO / CPU.

All backends speak plain ONNX Runtime, so switching is a one-word change and
the choice is benchmarkable (`scripts/benchmark.py`).

Strict by default: if the requested primary provider is unavailable we raise
instead of silently falling back to CPU -- a benchmark that thinks it is
testing "dml" but runs on CPU is worse than a crash (review S1).
"""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort

# Ordered preference per backend tag.
_PROVIDERS = {
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "openvino": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
}


def create_session(model_path: str | Path, backend: str = "dml",
                   strict: bool = True) -> ort.InferenceSession:
    if backend not in _PROVIDERS:
        raise ValueError(f"unknown backend {backend!r}, expected one of {sorted(_PROVIDERS)}")
    wanted_primary = _PROVIDERS[backend][0]
    available = set(ort.get_available_providers())
    if wanted_primary not in available:
        if strict:
            raise RuntimeError(
                f"backend {backend!r} requested {wanted_primary} but it is not "
                f"available on this machine (providers: {sorted(available)}). "
                f"Pass strict=False to fall back, or pick another backend.")
        print(f"[backends] WARNING: {wanted_primary} unavailable, falling back")
    providers = [p for p in _PROVIDERS[backend] if p in available] or ["CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    print(f"[backends] backend={backend} providers={sess.get_providers()}")
    return sess
