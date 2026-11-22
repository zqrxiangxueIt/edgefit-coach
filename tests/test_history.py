"""HistoryStore persistence tests (tmp DB per test)."""

import tempfile
from pathlib import Path

import pytest

from server.history import HistoryStore


@pytest.fixture()
def store():
    # tempfile instead of pytest tmp_path: the default basetemp under
    # C:\temp is not writable on this machine
    with tempfile.TemporaryDirectory() as d:
        s = HistoryStore(str(Path(d) / "test.db"))
        yield s
        s.close()


def test_session_lifecycle(store):
    sid = store.begin(sets_planned=3, reps_per_set=2, rest_sec=60)
    assert store.current_id == sid
    store.feed([
        {"index": 1, "side": "left", "full": True, "min_angle": 40.0,
         "t": 5.2, "t_con": 1.1, "t_ecc": 2.2},
        {"index": 2, "side": "right", "full": False, "min_angle": 90.0,
         "t": 9.7, "t_con": 1.0, "t_ecc": 1.5},
    ])
    # dedup: re-feeding the same events must not duplicate
    store.feed([
        {"index": 2, "side": "right", "full": False, "min_angle": 90.0,
         "t": 9.7, "t_con": 1.0, "t_ecc": 1.5},
        {"index": 3, "side": "left", "full": True, "min_angle": 45.0,
         "t": 14.0, "t_con": 1.2, "t_ecc": 2.0},
    ])
    store.finish({"reps": 3, "full_reps": 2, "half_reps": 1, "suppressed": 0},
                 {"set_index": 2, "elapsed_sec": 42.0})
    assert store.current_id is None

    sessions = store.recent_sessions()
    assert len(sessions) == 1
    s = sessions[0]
    assert s["reps_total"] == 3 and s["full_reps"] == 2
    assert s["full_rate"] == pytest.approx(2 / 3)
    assert s["min_angle_med"] == pytest.approx(45.0)  # median of 40/90/45

    reps = store.recent_reps()
    assert len(reps) == 3  # dedup held


def test_last_session_filters_exercise(store):
    store.begin(exercise="curl")
    store.finish({"reps": 1}, {})
    store.begin(exercise="squat")
    store.finish({"reps": 1}, {})
    assert store.last_session("curl")["exercise"] == "curl"
    assert store.last_session("squat")["exercise"] == "squat"


def test_feed_without_open_session_is_noop(store):
    store.feed([{"index": 1, "side": "left", "full": True,
                 "min_angle": 40.0, "t": 1.0}])
    assert store.recent_reps() == []
