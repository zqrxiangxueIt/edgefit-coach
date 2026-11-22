"""Unified inference session factory: DirectML / OpenVINO / CPU.

All backends speak plain ONNX Runtime, so switching is a one-word change and
the choice is benchmarkable (`scripts/benchmark.py`).
"""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort

# Ordered preference per backend tag; ORT falls through if a provider is
# unavailable on the machine.
_PROVIDERS = {
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "openvino": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
}


def create_session(model_path: str | Path, backend: str = "dml") -> ort.InferenceSession:
    if backend not in _PROVIDERS:
        raise ValueError(f"unknown backend {backend!r}, expected one of {sorted(_PROVIDERS)}")
    available = set(ort.get_available_providers())
    providers = [p for p in _PROVIDERS[backend] if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    return sess
