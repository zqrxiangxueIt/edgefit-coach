"""Backend factory strictness tests (review B / S1).

Environment-coupled cases are skipped conditionally so the suite stays
portable (CI without DirectML, machines with OpenVINO, the Pi, ...).
"""

import onnxruntime as ort
import pytest

from server.inference.backends import create_session

MODEL = "models/yolo11n-pose.onnx"
HAS_DML = "DmlExecutionProvider" in ort.get_available_providers()
HAS_OV = "OpenVINOExecutionProvider" in ort.get_available_providers()


def test_missing_provider_raises_instead_of_silent_cpu():
    if HAS_OV:  # raise-case needs the EP to be genuinely absent
        pytest.skip("OpenVINO EP installed here; raise-path not exercisable")
    with pytest.raises(RuntimeError, match="OpenVINOExecutionProvider"):
        create_session(MODEL, "openvino")


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        create_session(MODEL, "tensorrt")


@pytest.mark.skipif(not HAS_DML, reason="DirectML not available on this machine")
def test_dml_primary_provider_honored():
    sess = create_session(MODEL, "dml")
    assert sess.get_providers()[0] == "DmlExecutionProvider"
