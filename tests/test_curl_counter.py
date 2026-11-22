"""CurlCounter behaviour tests with synthetic keypoint sequences.

Keypoint layout: right arm only (auto side-pick locks to right after a few
frames).  Elbow at (200, 300); shoulder left of elbow; wrist placed so the
interior elbow angle equals `beta`:  wrist = elbow + (-cos(beta), sin(beta))
scaled by forearm length 100px.
"""

import math

import numpy as np

from server.analysis.curl_counter import CurlCounter, Phase
from server.analysis.geometry import (L_SHOULDER, L_ELBOW, L_WRIST, L_HIP)

ELBOW = np.array([200.0, 300.0])


def frame(beta: float, conf: float = 1.0):
    """One (17,3) keypoint frame whose right-elbow interior angle is beta."""
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[6] = [100.0, 300.0, conf]                     # R_SHOULDER
    kps[8] = [ELBOW[0], ELBOW[1], conf]               # R_ELBOW
    kps[10] = [ELBOW[0] - 100 * math.cos(math.radians(beta)),
               ELBOW[1] + 100 * math.sin(math.radians(beta)), conf]  # R_WRIST
    kps[12] = [150.0, 500.0, conf]                    # R_HIP (torso scale)
    return kps


def sweep(a: float, b: float, n: int):
    return [a + (b - a) * i / n for i in range(1, n + 1)]


def feed(counter, angles):
    return [e for ang in angles if (e := counter.update(frame(ang), t=ang)) is not None]


def test_five_full_reps():
    c = CurlCounter()
    events = feed(c, [170.0] * 8)                     # warmup, side locks
    assert c.phase is Phase.EXTENDED
    seq = []
    for _ in range(5):
        seq += sweep(170, 40, 5) + [40.0] * 5 + sweep(40, 170, 5) + [170.0] * 5
    events += feed(c, seq)
    assert len(events) == 5
    assert all(e.full for e in events)
    assert [e.index for e in events] == [1, 2, 3, 4, 5]
    assert c.stats.reps == 5 and c.stats.half_reps == 0


def test_half_rep_flagged_with_warning():
    c = CurlCounter()
    feed(c, [170.0] * 8)
    seq = (sweep(170, 95, 5) + [95.0] * 5 + sweep(95, 170, 5) + [170.0] * 5) * 2
    events = feed(c, seq)
    assert len(events) == 2
    assert all(not e.full for e in events)
    assert c.stats.half_reps == 2
    assert len(c.stats.warnings) == 2


def test_jitter_below_thresholds_no_counts():
    c = CurlCounter()
    feed(c, [170.0] * 8)
    # oscillate 60 <-> 90: enters CURLING but never returns to EXTENDED
    events = feed(c, [60.0 if i % 2 == 0 else 90.0 for i in range(40)])
    assert events == []
    assert c.stats.reps == 0
    assert c.phase is Phase.CURLING


def test_lost_person_resets_to_idle():
    c = CurlCounter()
    feed(c, [170.0] * 8)
    c.update(frame(40.0))                             # mid-rep
    c.update(frame(40.0, conf=0.0))                   # tracker lost the person
    assert c.phase is Phase.IDLE
    # person returns straight-armed: still no phantom rep
    events = feed(c, [170.0] * 6)
    assert events == [] and c.stats.reps == 0


def test_threshold_validation():
    import pytest
    with pytest.raises(ValueError):
        CurlCounter(up_angle=100, down_angle=110, full_rep_angle=105)
