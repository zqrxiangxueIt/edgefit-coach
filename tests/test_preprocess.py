"""A-side preprocessor tests (M2 event mode, M4 cascade/privacy)."""

import numpy as np

from edge.preprocess import (CascadePreprocessor, EventPreprocessor,
                             PrivacyPreprocessor, RawPreprocessor)


class FakeDetector:
    """Deterministic stand-in for the LiteRT person detector."""

    def __init__(self, boxes_by_frame):
        self.boxes_by_frame = boxes_by_frame
        self.i = 0
        self.last_ms = 3.0

    def detect(self, frame):
        boxes = self.boxes_by_frame[min(self.i, len(self.boxes_by_frame) - 1)]
        self.i += 1
        return boxes

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


def test_cascade_sends_only_when_person_present():
    boxes = [[100, 100, 300, 400, 0.9]]        # person box (x1,y1,x2,y2,score)
    det = FakeDetector([boxes] * 5 + [[]] * 30 + [boxes] * 3)
    p = CascadePreprocessor(det, grace_frames=5)
    outs = [p.process(gray_frame()) for _ in range(38)]
    sent = [o for o in outs if o is not None]
    # 5 with person + 5 grace + first of re-acquisition (grace re-arms on hit)
    assert len(sent) == 5 + 5 + 3
    assert all("persons" in o.extra for o in sent)
    assert sent[0].extra["det_ms"] == 3.0


def test_cascade_silence_after_grace():
    det = FakeDetector([[]] * 40)
    p = CascadePreprocessor(det, grace_frames=3)
    outs = [p.process(gray_frame()) for _ in range(40)]
    # only the very first frame goes out (self._i == 1 special case)
    assert sum(o is not None for o in outs) == 1


def test_privacy_crops_person_roi():
    import cv2
    det = FakeDetector([[[50, 40, 200, 440, 0.9]]])   # one frame, one 150x400 box
    p = PrivacyPreprocessor(det)
    out = p.process(gray_frame())
    assert out is not None
    x, y, w, h = out.extra["roi"]
    assert x >= 0 and y >= 0 and 100 < w < 640       # padded, inside frame
    img = cv2.imdecode(np.frombuffer(out.jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert abs(img.shape[1] - w) <= 2                # crop width == ROI width
    assert abs(img.shape[0] - h) <= 2


def test_privacy_no_person_goes_silent_except_keepalive():
    det = FakeDetector([[]] * 30)
    p = PrivacyPreprocessor(det, keepalive_every=5)
    outs = [p.process(gray_frame()) for _ in range(20)]
    # frames 5, 10, 15, 20 are keepalives (tiny black marker frames)
    assert sum(o is not None for o in outs) == 4
    ka = outs[4]
    assert ka is not None and ka.extra["roi"] is None and ka.extra["persons"] == []


def test_privacy_roi_smoothed():
    det = FakeDetector([[[100, 100, 200, 300, 0.9]], [[300, 100, 400, 300, 0.9]]])
    p = PrivacyPreprocessor(det, smooth=0.5)
    p.process(gray_frame())
    out = p.process(gray_frame())
    x = out.extra["roi"][0]
    # EMA 0.5: between 100-ish and 300-ish, strictly inside (100, 300)
    assert 120 < x < 280


def test_privacy_survives_detector_blink():
    """检测器间歇漏检（人还在镜头前）时，ROI 内实时画面必须持续，不得黑屏。"""
    det = FakeDetector([[[50, 40, 200, 440, 0.9]]] * 5 + [[]] * 20)
    p = PrivacyPreprocessor(det, lost_reset=45)
    outs = [p.process(gray_frame()) for _ in range(25)]
    # all 25 frames must transmit: 5 with detection + 20 persisted live crops
    assert all(o is not None for o in outs)
    assert outs[4].extra["persons"]            # detected
    assert outs[10].extra["persons"] == []     # blink: no detection ...
    assert outs[10].extra["roi"] is not None   # ... but live ROI continues


def test_privacy_goes_dark_after_window():
    det = FakeDetector([[[50, 40, 200, 440, 0.9]]] * 5 + [[]] * 60)
    p = PrivacyPreprocessor(det, lost_reset=45, keepalive_every=10)
    outs = [p.process(gray_frame()) for _ in range(65)]
    # after the 45-frame window expires, only keepalives (black markers) remain
    tail = outs[55:]
    assert all(o is None or o.extra["roi"] is None for o in tail)
    assert any(o is not None and o.extra["roi"] is None for o in tail)  # keepalive fired
