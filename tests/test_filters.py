"""One-Euro filter and KeypointSmoother behaviour tests."""

import numpy as np

from server.analysis.filters import KeypointSmoother, OneEuroFilter


def test_first_sample_passes_through():
    f = OneEuroFilter(freq=30.0)
    assert f(42.0) == 42.0


def test_jitter_is_damped():
    rng = np.random.default_rng(7)
    raw = 100.0 + rng.normal(0, 3.0, 200)      # static point + jitter
    f = OneEuroFilter(freq=30.0)               # calibrated defaults
    out = np.array([f(x) for x in raw])
    assert out[20:].std() < raw[20:].std() * 0.5


def test_fast_motion_is_followed():
    f = OneEuroFilter(freq=30.0)
    for _ in range(10):
        f(0.0)
    out = [f(30.0) for _ in range(5)]          # big step: must converge fast
    assert abs(out[-1] - 30.0) < 1.0


def test_reset_clears_velocity_state():
    f = OneEuroFilter(freq=30.0)
    for x in np.linspace(0, 50, 30):           # build up large dx
        f(x)
    assert f._dx_prev > 10.0                   # precondition: state exists
    f.reset()
    assert f(50.0) == 50.0                     # passthrough again
    assert f._dx_prev == 0.0                   # no stale velocity
    # next samples stay near the new position (no kick from old derivative)
    assert abs(f(50.5) - 50.5) < 0.5


def test_invalid_freq_rejected():
    import pytest
    with pytest.raises(ValueError):
        OneEuroFilter(freq=0)


def test_smoother_resets_lost_keypoints():
    sm = KeypointSmoother()
    kps = np.zeros((17, 3), np.float32)
    kps[9] = [10.0, 20.0, 0.9]
    out = sm.update(kps)
    assert out[9, 0] == 10.0
    kps[9] = [10.0, 20.0, 0.0]                 # lost
    sm.update(kps)
    assert sm._fx[9]._x_prev is None and sm._fx[9]._dx_prev == 0.0
    kps[9] = [200.0, 300.0, 0.9]               # re-acquired far away
    out = sm.update(kps)
    assert out[9, 0] == 200.0                  # instant, no smoothing lag
