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


def test_projected_angle_formula_roundtrip():
    """闭式解数值表：真40°在60°斜视下投影为72°；校正应还原回~40°。"""
    from server.analysis.geometry import correct_projected_angle
    import math
    phi = math.radians(53.13)           # cos(phi)=0.6, comfortably correctable
    true = math.radians(40)
    # forward: true -> proj  (tan(proj/2) = tan(true/2)/cos(phi))
    proj = 2 * math.degrees(math.atan(math.tan(true / 2) / math.cos(phi)))
    assert proj == pytest.approx(62.5, abs=1.0)
    # inverse: need sin(psi) = cos(phi) = 0.6 -> cos_psi = 0.8
    corrected = correct_projected_angle(proj, 0.8)
    assert corrected == pytest.approx(40.0, abs=1.5)


def test_correction_skips_frontal_and_extremes():
    from server.analysis.geometry import correct_projected_angle
    assert correct_projected_angle(90.0, 0.99) == 90.0   # near-frontal: skip
    assert correct_projected_angle(170.0, 0.3) == 170.0  # extreme: no-op
    assert correct_projected_angle(20.0, 0.3) == 20.0    # extreme: no-op


def test_correction_only_shrinks():
    from server.analysis.geometry import correct_projected_angle
    for proj in (50, 80, 110, 140):
        for cos_psi in (0.0, 0.3, 0.6):
            assert correct_projected_angle(proj, cos_psi) <= proj + 1e-9


def test_estimate_yaw_scale_validity():
    import numpy as np
    from server.analysis.geometry import estimate_yaw_scale
    kps = np.zeros((17, 3), np.float32)
    # sideways body: shoulders overlap (rho ~ rho_side) -> cos_psi ~ 0
    kps[5] = [100, 100, 0.9]; kps[6] = [110, 105, 0.9]
    kps[11] = [105, 300, 0.9]; kps[12] = [108, 302, 0.9]
    cos_psi, ok = estimate_yaw_scale(kps, rho_front=0.85, rho_side=0.30)
    assert ok and cos_psi < 0.2
    # frontal body: wide shoulders (rho ~ rho_front) -> cos_psi ~ 1
    kps[6] = [300, 100, 0.9]; kps[12] = [295, 302, 0.9]
    cos_psi, ok = estimate_yaw_scale(kps, rho_front=0.85, rho_side=0.30)
    assert ok and cos_psi > 0.8
    # low confidence -> invalid
    kps[5, 2] = 0.1
    assert estimate_yaw_scale(kps)[1] is False
