"""One-Euro filter for keypoint smoothing.

The One-Euro filter (Casiez et al., CHI 2012) is the de-facto standard for
pointer/pose filtering: a low-pass filter whose cutoff rises with speed, so
slow jitter is damped hard while fast, deliberate motion stays responsive.

The derivative estimate uses the difference of *raw* input values, matching
the paper's reference implementation (OneEuroFilter.h), so heavy low-pass
damping on the position cannot suppress the speed estimate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class OneEuroFilter:
    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.5,
                 beta: float = 0.05, d_cutoff: float = 1.0) -> None:
        """Defaults calibrated for pixel-scale keypoints at ~30 FPS:
        beta=0.05 damps static jitter ~50% while a fast wrist snap
        converges within a few frames (see tests/test_filters.py)."""
        if freq <= 0:
            raise ValueError("freq must be > 0")
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._x_raw_prev: Optional[float] = None
        self._dx_prev = 0.0

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def reset(self) -> None:
        """Clear all state: next sample passes through unfiltered."""
        self._x_prev = None
        self._x_raw_prev = None
        self._dx_prev = 0.0

    def __call__(self, x: float) -> float:
        if self._x_prev is None:
            self._x_prev, self._x_raw_prev = x, x
            return x
        dx = (x - self._x_raw_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a_c = self._alpha(cutoff)
        x_hat = a_c * x + (1.0 - a_c) * self._x_prev
        self._x_prev, self._x_raw_prev, self._dx_prev = x_hat, x, dx_hat
        return x_hat


class KeypointSmoother:
    """Per-coordinate One-Euro smoothing over a (17, 3) keypoint array.

    Keypoints with conf == 0 fully reset their filter (both position and
    derivative state): re-acquisition is instantaneous and unbiased.
    """

    def __init__(self, n_kps: int = 17, **filter_kwargs) -> None:
        self._fx = [OneEuroFilter(**filter_kwargs) for _ in range(n_kps)]
        self._fy = [OneEuroFilter(**filter_kwargs) for _ in range(n_kps)]

    def update(self, kps: np.ndarray, dt: float | None = None) -> np.ndarray:
        """dt: real inter-frame interval when known (from A-side capture
        timestamps). The filter's 30fps assumption breaks under throttled /
        narrowband delivery where the effective rate is far lower."""
        if dt is not None and dt > 1e-3:
            freq = 1.0 / dt
            for f in self._fx + self._fy:
                f.freq = freq
        out = kps.copy()
        for i in range(kps.shape[0]):
            if kps[i, 2] <= 0.0:
                # Lost keypoint: reset so re-acquisition is instantaneous.
                self._fx[i].reset()
                self._fy[i].reset()
                continue
            out[i, 0] = self._fx[i](float(kps[i, 0]))
            out[i, 1] = self._fy[i](float(kps[i, 1]))
        return out
