"""Export pose models to ONNX, reproducibly.

MUST set YOLO_AUTOINSTALL=false BEFORE importing ultralytics: its runtime
check_requirements("onnxruntime") matches by dist name, does not accept
onnxruntime-directml, and would auto-install the CPU build -- clobbering
the DML package files and breaking every DML session in this venv
(review B, reproduced for real once).

Usage:
    .venv/Scripts/python scripts/export_models.py
"""

import os

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from pathlib import Path

from ultralytics import YOLO

MODELS = ("yolo11n-pose", "yolo26n-pose")


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "models"
    out_dir.mkdir(exist_ok=True)
    for name in MODELS:
        model = YOLO(f"{name}.pt")
        path = model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
        print(f"exported: {path}")


if __name__ == "__main__":
    main()
