import numpy as np
import pytest

from server.analysis.geometry import joint_angle, L_ELBOW, L_SHOULDER, L_WRIST


def make_kp(sh, el, wr):
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[L_SHOULDER] = [*sh, 1.0]
    kps[L_ELBOW] = [*el, 1.0]
    kps[L_WRIST] = [*wr, 1.0]
    return kps


def test_right_angle():
    # shoulder (0,0), elbow (1,0), wrist (1,1): 90 degrees at elbow
    ang, conf = joint_angle(make_kp((0, 0), (1, 0), (1, 1)), L_SHOULDER, L_ELBOW, L_WRIST)
    assert ang == pytest.approx(90.0, abs=0.5)
    assert conf == pytest.approx(1.0)


def test_straight_arm_180():
    ang, _ = joint_angle(make_kp((0, 0), (1, 0), (2, 0)), L_SHOULDER, L_ELBOW, L_WRIST)
    assert ang == pytest.approx(180.0, abs=0.5)


def test_degenerate_points():
    ang, conf = joint_angle(make_kp((1, 1), (1, 1), (2, 2)), L_SHOULDER, L_ELBOW, L_WRIST)
    assert ang == 180.0 and conf == 0.0


def test_confidence_drops_with_weak_keypoint():
    kps = make_kp((0, 0), (1, 0), (1, 1))
    kps[L_WRIST, 2] = 0.5
    _, conf = joint_angle(kps, L_SHOULDER, L_ELBOW, L_WRIST)
    assert conf == pytest.approx(0.5)
