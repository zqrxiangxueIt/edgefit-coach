"""CurlCounter behaviour tests with synthetic keypoint sequences.

Geometry: right arm — shoulder (100,300), elbow (200,300), wrist placed so
the interior elbow angle equals `beta`:  wrist = elbow + (-cos(beta), sin(beta))
* 100px.  Left arm is the mirror, offset 200px down.  Upper arm = 100px
(activity-gate threshold = 100px path with cap 40px/frame).
"""

import math

import numpy as np
import pytest

from server.analysis.curl_counter import CurlCounter, Phase


def frame(beta: float, beta_left: float | None = None, conf: float = 1.0):
    """One (17,3) keypoint frame; left arm included when beta_left given."""
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[6] = [100.0, 300.0, conf]                     # R_SHOULDER
    kps[8] = [200.0, 300.0, conf]                     # R_ELBOW
    kps[10] = [200.0 - 100 * math.cos(math.radians(beta)),
               300.0 + 100 * math.sin(math.radians(beta)), conf]      # R_WRIST
    kps[12] = [150.0, 300.0, conf]                    # R_HIP (torso scale)
    if beta_left is not None:
        # mirror: shoulder right of elbow so the interior angle is beta_left
        kps[5] = [300.0, 500.0, conf]                 # L_SHOULDER
        kps[7] = [200.0, 500.0, conf]                 # L_ELBOW
        kps[9] = [200.0 + 100 * math.cos(math.radians(beta_left)),
                  500.0 + 100 * math.sin(math.radians(beta_left)), conf]  # L_WRIST
        kps[11] = [350.0, 500.0, conf]                # L_HIP
    return kps


def sweep(a: float, b: float, n: int):
    return [a + (b - a) * i / n for i in range(1, n + 1)]


def feed(counter, frames):
    """Feed frames; returns all rep events. Timestamps are monotonic."""
    events = []
    for i, f in enumerate(frames):
        if isinstance(f, (int, float)):
            f = frame(f)
        events.extend(counter.update(f, t=i / 30.0))
    return events


def test_five_full_reps():
    c = CurlCounter()
    feed(c, [170.0] * 8)                     # warmup
    assert c.phase is Phase.EXTENDED
    seq = []
    for _ in range(5):
        seq += sweep(170, 40, 5) + [40.0] * 5 + sweep(40, 170, 5) + [170.0] * 5
    events = feed(c, seq)
    assert len(events) == 5
    assert all(e.full for e in events)
    assert [e.index for e in events] == [1, 2, 3, 4, 5]
    assert c.stats.reps == 5 and c.stats.half_reps == 0


def test_dual_arm_same_frame_both_counted():
    """杠铃弯举：双臂同帧完成 —— 两个事件都必须上报（B1 回归）。"""
    c = CurlCounter()
    frames = [(170.0, 170.0)] * 8
    for _ in range(3):
        for a, b in zip(sweep(170, 40, 5), sweep(170, 40, 5)):
            frames.append((a, b))
        frames += [(40.0, 40.0)] * 5
        for a, b in zip(sweep(40, 170, 5), sweep(40, 170, 5)):
            frames.append((a, b))
        frames += [(170.0, 170.0)] * 5
    events = feed(c, [frame(r, l) for r, l in frames])
    assert len(events) == 6
    assert sum(e.side == "left" for e in events) == 3
    assert sum(e.side == "right" for e in events) == 3
    assert c.stats.reps == 6
    # 同帧完成的两个 rep 时间戳一致
    ts = [e.t for e in events]
    assert ts[0] == ts[1]


def test_isolated_spikes_no_phantom():
    """B2 回归：CURLING 期插入孤立单帧伸展尖峰（非连续）不得凑成 rep。"""
    c = CurlCounter()
    feed(c, [170.0] * 8)
    seq = [40.0] * 5 + [170.0] + [40.0] * 5 + [170.0] + [40.0] * 5 + [170.0] + [40.0] * 5
    events = feed(c, seq)
    assert events == []
    assert c.stats.reps == 0
    assert c.stats.phantom_reps_suppressed == 0
    assert c.phase is Phase.CURLING


def test_half_rep_flagged_with_warning():
    c = CurlCounter()
    feed(c, [170.0] * 8)
    seq = (sweep(170, 95, 5) + [95.0] * 5 + sweep(95, 170, 5) + [170.0] * 5) * 2
    events = feed(c, seq)
    assert len(events) == 2
    assert all(not e.full for e in events)
    assert c.stats.half_reps == 2
    assert len(c.stats.warnings) == 2


def test_glitch_while_extended_does_not_pollute_min():
    """S1 回归：EXTENDED 期的单帧深屈毛刺不得把后续真实半程污染成全程。"""
    c = CurlCounter()
    feed(c, [170.0] * 8)
    seq = [60.0] + [170.0] * 4          # unconfirmed deep-flex glitch
    seq += sweep(170, 80, 5) + [80.0] * 5 + sweep(80, 170, 5) + [170.0] * 5
    events = feed(c, seq)
    assert len(events) == 1
    assert events[0].full is False
    assert events[0].min_angle == pytest.approx(80.0, abs=2.0)
    assert "幅度不足" in c.stats.warnings[0]


def test_fast_rep_without_hold_seeds_min_at_confirmation():
    """无停顿的快速动作：确认帧（最深点）必须播种 min，否则全程误判半程。"""
    c = CurlCounter()
    feed(c, [170.0] * 8)
    seq = []
    for _ in range(2):
        seq += sweep(170, 40, 4) + sweep(40, 170, 4) + [170.0] * 4
    events = feed(c, seq)
    assert len(events) == 2
    assert all(e.full for e in events)
    assert all(e.min_angle <= 60.0 for e in events)


def test_boundary_angle_exactly_full_rep():
    """min == full_rep_angle (75°) 按全程计（<= + 0.5° 容差）。"""
    c = CurlCounter()
    feed(c, [170.0] * 8)
    seq = sweep(170, 75, 5) + [75.0] * 5 + sweep(75, 170, 5) + [170.0] * 5
    events = feed(c, seq)
    assert len(events) == 1
    assert events[0].full is True
    assert events[0].min_angle == pytest.approx(75.0, abs=0.1)


def test_boundary_angle_just_past_half():
    """min = 76° 超出容差 -> 半程。"""
    c = CurlCounter()
    feed(c, [170.0] * 8)
    seq = sweep(170, 76, 5) + [76.0] * 5 + sweep(76, 170, 5) + [170.0] * 5
    events = feed(c, seq)
    assert len(events) == 1
    assert events[0].full is False


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
    c.update(frame(40.0), t=9.0)                     # mid-rep
    c.update(frame(40.0, conf=0.0), t=9.1)           # tracker lost the person
    assert c.phase is Phase.IDLE
    events = feed(c, [170.0] * 6)                    # person returns straight-armed
    assert events == [] and c.stats.reps == 0


def test_confidence_flicker_around_gate():
    """conf 在 min_conf 边界抖动：掉线帧触发 lost()，不得崩溃或误计。"""
    c = CurlCounter()
    feed(c, [170.0] * 8)
    seq = [frame(40.0 if i % 4 < 2 else 170.0, conf=0.34 if i % 3 == 0 else 0.9)
           for i in range(30)]
    events = feed(c, seq)
    assert c.stats.reps == len(events)


def test_hanging_arm_phantom_suppressed():
    """单臂场景：垂下的右臂手腕被误检"搬运"到屈臂姿势（遮挡伪影），
    手腕并无真实位移轨迹 -> 活动门控应吞掉这次幽灵计数。"""
    c = CurlCounter()
    seq = [170.0] * 8 + [40.0] * 12 + [170.0] * 8   # 静止垂臂 + 静止假屈臂 + 回垂
    events = feed(c, seq)
    assert events == []
    assert c.stats.reps == 0
    assert c.stats.phantom_reps_suppressed == 1


def test_real_single_arm_rep_passes_gate():
    """真实单臂弯举（手腕沿圆弧扫过）必须照常计数——门控不能误伤。"""
    c = CurlCounter()
    feed(c, [170.0] * 8)
    seq = []
    for _ in range(3):
        seq += sweep(170, 40, 6) + [40.0] * 4 + sweep(40, 170, 6) + [170.0] * 4
    events = feed(c, seq)
    assert len(events) == 3
    assert c.stats.phantom_reps_suppressed == 0
    assert all(e.full for e in events)


def test_torso_swing_warning():
    """躯干大幅横移（借力）应触发警告。"""
    c = CurlCounter()
    frames = [170.0] * 8
    # during the rep the shoulder slides 80px sideways (torso=50px -> span 1.6x)
    swing_seq = sweep(170, 40, 5) + [40.0] * 5 + sweep(40, 170, 5) + [170.0] * 5
    for i, beta in enumerate(swing_seq):
        kps = frame(beta)
        kps[6, 0] = 100.0 + (80.0 if 8 <= i < 8 + len(swing_seq) / 2 else 0.0)
        frames.append(kps)
    events = feed(c, frames)
    assert len(events) == 1
    assert any("躯干摆动" in w for w in c.stats.warnings)


def test_threshold_validation():
    with pytest.raises(ValueError):
        CurlCounter(up_angle=100, flex_angle=110, full_rep_angle=75)
    with pytest.raises(ValueError):
        CurlCounter(debounce=0)
    with pytest.raises(ValueError):
        CurlCounter(min_conf=1.5)


def _feed_timed(counter, seq):
    """Feed (angle, t_seconds) pairs; returns events."""
    events = []
    for beta, t in seq:
        events.extend(counter.update(frame(beta), t=t))
    return events


def test_tempo_measured_per_rep():
    """t_con/t_ecc 从相位锚点计算：伸展→最低点=向心，最低点→完成=离心。"""
    c = CurlCounter()
    warm = [(170.0, i / 10) for i in range(10)]          # extended 0..0.9s
    fast_lift = [(170 - 130 * (i + 1) / 6, 1.0 + i * 0.05) for i in range(6)]  # 1.0→1.25
    hold_bottom = [(40.0, 1.3 + i * 0.1) for i in range(3)]                    # 1.3..1.5
    slow_lower = [(40 + 130 * (i + 1) / 8, 1.6 + i * 0.4) for i in range(8)]   # 1.6→4.4
    done = [(170.0, 4.5 + i * 0.15) for i in range(4)]
    events = _feed_timed(c, warm + fast_lift + hold_bottom + slow_lower + done)
    assert len(events) == 1
    ev = events[0]
    # lift: last-extended ~0.9 -> min at ~1.3  => t_con ~0.4s
    assert 0.2 <= ev.t_con <= 0.6
    # lower: min ~1.3 -> completion ~4.5+ => t_ecc ~3s
    assert 2.5 <= ev.t_ecc <= 3.6
    assert not any("下放过快" in w for w in c.stats.warnings)


def test_tempo_fast_eccentric_warns():
    c = CurlCounter()
    seq = [(170.0, i / 10) for i in range(10)]
    seq += [(170 - 130 * (i + 1) / 6, 1.0 + i * 0.05) for i in range(6)]
    seq += [(40.0, 1.3), (40.0, 1.4)]                    # brief bottom
    seq += [(40 + 130 * (i + 1) / 6, 1.5 + i * 0.05) for i in range(6)]  # back up fast
    seq += [(170.0, 1.9 + i * 0.15) for i in range(4)]
    events = _feed_timed(c, seq)
    assert len(events) == 1
    assert events[0].t_ecc < 1.0
    assert any("下放过快" in w for w in c.stats.warnings)
