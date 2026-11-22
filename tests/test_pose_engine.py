"""PoseEngine output-layout guard tests (review R2 / N4 regression pin).

Uses a stub session so no model file or GPU is needed.
"""

import numpy as np
import pytest

from server.inference.pose_engine import PoseEngine, STRIDE


class _FakeInput:
    def __init__(self):
        self.name = "images"


class _FakeSession:
    def __init__(self, out_shape):
        self._out = np.random.rand(*out_shape).astype(np.float32)
        self._inp = _FakeInput()

    def get_inputs(self):
        return [self._inp]

    def run(self, _names, _feeds):
        return [self._out]


def _engine_with(out_shape) -> PoseEngine:
    eng = PoseEngine.__new__(PoseEngine)          # skip __init__ (no model load)
    eng.sess = _FakeSession(out_shape)
    eng.input_name = "images"
    eng.conf_thres = 0.35
    eng.iou_thres = 0.65
    return eng


def _frame():
    return np.zeros((480, 640, 3), np.uint8)


@pytest.mark.parametrize("bad", [(1, 57, 840), (1, 56, 840, 1), (56, 840)])
def test_bad_layout_raises(bad):
    with pytest.raises(ValueError, match="unexpected pose head layout"):
        _engine_with(bad).detect(_frame())


def test_good_layout_passes():
    res = _engine_with((1, STRIDE, 840)).detect(_frame())
    assert isinstance(res, list)
