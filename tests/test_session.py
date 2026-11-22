"""TrainingSession state machine tests (M5: sets/reps/rest/countdown)."""

import pytest

from server.session import SessionConfig, SessionPhase, TrainingSession


class FakeRep:
    def __init__(self, index):
        self.index = index


def test_config_validation():
    with pytest.raises(ValueError):
        SessionConfig(sets=0).validate()
    with pytest.raises(ValueError):
        SessionConfig(reps_per_set=0).validate()
    with pytest.raises(ValueError):
        SessionConfig(rest_sec=1).validate()


def test_full_session_flow():
    s = TrainingSession()
    t0 = 1000.0
    s.start(total_reps=0, now=t0, sets=2, reps_per_set=2, rest_sec=30)
    assert s.snapshot(t0 + 0.5)["phase"] == "countdown"

    # countdown elapses -> set 1 begins
    s.update(0, [], now=t0 + 3.0)
    assert s.snapshot(t0 + 3.1)["phase"] == "set"
    assert s.snapshot(t0 + 3.1)["set_index"] == 1

    # two reps complete set 1
    s.update(1, [FakeRep(1)], now=t0 + 5.0)
    s.update(2, [FakeRep(2)], now=t0 + 7.0)
    snap = s.snapshot(t0 + 7.1)
    assert snap["phase"] == "rest" and snap["remaining_sec"] > 28
    assert snap["set_reps"] == [2]

    # rest elapses -> set 2
    s.update(2, [], now=t0 + 37.1)
    assert s.snapshot(t0 + 37.2)["phase"] == "set"
    assert s.snapshot(t0 + 37.2)["set_index"] == 2

    # final two reps -> done
    s.update(3, [FakeRep(3)], now=t0 + 40)
    s.update(4, [FakeRep(4)], now=t0 + 42)
    snap = s.snapshot(t0 + 42.1)
    assert snap["phase"] == "done" and snap["set_reps"] == [2, 2]


def test_reps_during_countdown_ignored():
    s = TrainingSession()
    s.start(total_reps=0, now=0.0, sets=1, reps_per_set=3)
    s.update(3, [FakeRep(i) for i in (1, 2, 3)], now=1.0)  # warm-up during countdown
    assert s.snapshot(1.1)["phase"] == "countdown"
    s.update(3, [], now=3.0)                              # countdown over
    snap = s.snapshot(3.1)
    # warm-up reps do NOT count: the set baseline is taken at set start
    assert snap["phase"] == "set" and snap["reps_in_set"] == 0
    for i, t in ((4, 4.0), (5, 5.0), (6, 6.0)):
        s.update(i, [FakeRep(i)], now=t)
    assert s.snapshot(6.1)["phase"] == "done"
    assert s.snapshot(6.1)["set_reps"] == [3]


def test_skip_rest():
    s = TrainingSession()
    s.start(total_reps=0, now=0.0, sets=2, reps_per_set=1, rest_sec=60)
    s.update(0, [], now=3.0)                 # countdown -> set 1
    s.update(1, [FakeRep(1)], now=4.0)       # set 1 done -> rest 60s
    assert s.snapshot(4.1)["phase"] == "rest"
    s.skip_rest(now=5.0)
    assert s.snapshot(5.1)["phase"] == "set"
    assert s.snapshot(5.1)["set_index"] == 2


def test_baseline_uses_total_at_start():
    """中途开训练：已有计数作为基线，不计入本组。"""
    s = TrainingSession()
    s.start(total_reps=7, now=0.0, sets=1, reps_per_set=2)
    s.update(7, [], now=3.0)
    assert s.snapshot(3.1)["reps_in_set"] == 0
    s.update(8, [FakeRep(8)], now=5.0)
    s.update(9, [FakeRep(9)], now=6.0)
    assert s.snapshot(6.1)["phase"] == "done"
    assert s.snapshot(6.1)["set_reps"] == [2]


def test_stop_returns_to_free_mode():
    s = TrainingSession()
    s.start(total_reps=0, now=0.0, sets=1, reps_per_set=5)
    s.stop()
    assert s.snapshot()["phase"] == "idle"
