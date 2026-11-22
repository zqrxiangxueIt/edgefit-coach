"""Backend factory strictness tests (review B / S1)."""

import pytest

from server.inference.backends import create_session

MODEL = "models/yolo11n-pose.onnx"


def test_missing_provider_raises_instead_of_silent_cpu():
    # OpenVINO EP is not installed in this venv -> strict must refuse loudly
    with pytest.raises(RuntimeError, match="OpenVINOExecutionProvider"):
        create_session(MODEL, "openvino")


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        create_session(MODEL, "tensorrt")


def test_dml_primary_provider_honored():
    sess = create_session(MODEL, "dml")
    assert sess.get_providers()[0] == "DmlExecutionProvider"
