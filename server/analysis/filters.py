"""One-Euro filter for keypoint smoothing.

The One-Euro filter (Casiez et al., CHI 2012) is the de-facto standard for
pointer/pose filtering: a low-pass filter whose cutoff rises with speed, so
slow jitter is damped hard while fast, deliberate motion stays responsive.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class OneEuroFilter:
    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.7,
                 beta: float = 0.3, d_cutoff: float = 1.0) -> None:
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._dx_prev = 0.0

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        if self._x_prev is None:
            self._x_prev = x
            return x
        dx = (x - self._x_prev) * self.freq
        dx_hat = self._alpha(self.d_cutoff) * dx + (1 - self._alpha(self.d_cutoff)) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        x_hat = self._alpha(cutoff) * x + (1 - self._alpha(cutoff)) * self._x_prev
        self._x_prev, self._dx_prev = x_hat, dx_hat
        return x_hat


class KeypointSmoother:
    """Per-coordinate One-Euro smoothing over a (17, 3) keypoint array.

    Keypoints with conf == 0 reset their filter: there is nothing to smooth.
    """

    def __init__(self, n_kps: int = 17, **filter_kwargs) -> None:
        self._fx = [OneEuroFilter(**filter_kwargs) for _ in range(n_kps)]
        self._fy = [OneEuroFilter(**filter_kwargs) for _ in range(n_kps)]

    def update(self, kps: np.ndarray) -> np.ndarray:
        out = kps.copy()
        for i in range(kps.shape[0]):
            if kps[i, 2] <= 0.0:
                # Lost keypoint: reset so re-acquisition is instantaneous.
                self._fx[i]._x_prev = None
                self._fy[i]._x_prev = None
                continue
            out[i, 0] = self._fx[i](float(kps[i, 0]))
            out[i, 1] = self._fy[i](float(kps[i, 1]))
        return out
