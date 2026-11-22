"""A-side preprocessor tests (M2 event mode)."""

import numpy as np

from edge.preprocess import EventPreprocessor, RawPreprocessor

# module-level rng: a per-call seed would make two "noisy" frames pixel-identical
# (diff == 0), silently testing nothing (N6)
_RNG = np.random.default_rng(3)


def gray_frame(w=640, h=480, box_x=None, noise=0.0):
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    if box_x is not None:
        img[200:280, box_x:box_x + 60] = 220
    if noise > 0:
        img = np.clip(img.astype(int) + _RNG.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def test_first_frame_always_sent():
    p = EventPreprocessor()
    assert p.process(gray_frame()) is not None


def test_static_scene_goes_silent():
    p = EventPreprocessor()
    p.process(gray_frame(box_x=100))          # reference frame
    assert p.process(gray_frame(box_x=100)) is None
    assert p.process(gray_frame(box_x=100)) is None


def test_motion_wakes_stream_with_higher_quality():
    p = EventPreprocessor(q_min=50, q_max=90, idle_thr=0.5)
    p.process(gray_frame(box_x=100))
    slow = p.process(gray_frame(box_x=110))   # small step
    fast = p.process(gray_frame(box_x=180))   # big step
    assert slow is not None and fast is not None
    assert fast.jpeg_q >= slow.jpeg_q
    assert 50 <= slow.jpeg_q <= 90
    assert fast.motion > slow.motion


def test_quality_bounds_respected():
    p = EventPreprocessor(q_min=55, q_max=85, motion_ref=5.0)
    p.process(gray_frame())
    # full-frame brightness change: diff mean ~180 >> motion_ref -> saturates
    bright = np.full((480, 640, 3), 220, dtype=np.uint8)
    huge = p.process(bright)
    assert huge.jpeg_q == 85                  # clamped at q_max
    assert huge.motion == 1.0


def test_keepalive_reinjects_static_frame():
    p = EventPreprocessor(keepalive_every=3, idle_thr=0.5)
    p.process(gray_frame(box_x=50))
    assert p.process(gray_frame(box_x=50)) is None   # i=2 idle
    ka = p.process(gray_frame(box_x=50))             # i=3 keepalive
    assert ka is not None and ka.motion == 0.0


def test_raw_preprocessor_passes_every_frame():
    p = RawPreprocessor(jpeg_q=70)
    outs = [p.process(gray_frame(box_x=i)) for i in range(5)]
    assert all(o is not None for o in outs)
    assert all(o.jpeg_q == 70 for o in outs)
    assert outs[0].motion == 0.0
