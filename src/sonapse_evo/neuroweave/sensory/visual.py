"""Local visual feature extraction.  A frame is reduced to a handful of scalars and discarded.

For an H x W x 3 frame (uint8 0..255 or float 0..1), after block-averaging to a small grid:

    luminance   = mean(0.299 R + 0.587 G + 0.114 B)
    red/green   = mean(R - G)                       in [-1, 1]
    blue/yellow = mean(B - (R + G) / 2)             in [-1, 1]
    edge density= fraction of grid cells whose Sobel gradient magnitude exceeds ``edge_threshold``
    motion      = clip(4 * mean |L_t - L_{t-1}|, 0, 1)
    spatial     = luminance centroid (x, y) and the four quadrant energies
    change      = fraction of cells whose luminance changed by more than ``change_threshold``
    regions     = count and area fraction of connected bright regions (cells above mean + std)
    flicker     = exponential average of |d mean luminance|

The only retained state is the previous down-sampled luminance grid (a few hundred floats) for motion; it is
cleared by ``reset()`` and is never part of any exported state.  No method returns pixels.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage


def _grid(frame: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("frame must have shape (H, W, 3)")
    img = frame[..., :3].astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    rows = np.linspace(0, img.shape[0], shape[0] + 1).astype(int)
    cols = np.linspace(0, img.shape[1], shape[1] + 1).astype(int)
    out = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            block = img[rows[i]:max(rows[i + 1], rows[i] + 1), cols[j]:max(cols[j + 1], cols[j] + 1)]
            out[i, j] = block.reshape(-1, 3).mean(axis=0)
    return np.clip(out, 0.0, 1.0)


def extract_visual_features(frame: np.ndarray, previous: Optional[np.ndarray] = None, grid: Tuple[int, int] = (24, 32),
                            edge_threshold: float = 0.15, change_threshold: float = 0.1) -> Tuple[Dict[str, float], np.ndarray]:
    """Stateless extraction.  Returns ``(features, luminance_grid)``; pass the grid back as ``previous``."""
    small = _grid(frame, grid)
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    grad = np.hypot(ndimage.sobel(lum, axis=0), ndimage.sobel(lum, axis=1))
    diff = np.zeros_like(lum) if previous is None or previous.shape != lum.shape else np.abs(lum - previous)
    total = float(lum.sum()) + 1e-9
    ys, xs = np.mgrid[0:lum.shape[0], 0:lum.shape[1]]
    h2, w2 = lum.shape[0] // 2, lum.shape[1] // 2
    quadrants = [float(lum[:h2, :w2].mean()), float(lum[:h2, w2:].mean()), float(lum[h2:, :w2].mean()), float(lum[h2:, w2:].mean())]
    bright = lum > (lum.mean() + lum.std() + 1e-6)
    _, count = ndimage.label(bright)
    features = {
        "visual_luminance": float(lum.mean()),
        "visual_red_green": float(np.clip((r - g).mean(), -1.0, 1.0)),
        "visual_blue_yellow": float(np.clip((b - 0.5 * (r + g)).mean(), -1.0, 1.0)),
        "visual_edge_density": float((grad > edge_threshold).mean()),
        "visual_motion": float(np.clip(4.0 * diff.mean(), 0.0, 1.0)),
        "visual_centroid_x": float((lum * xs).sum() / total / max(1, lum.shape[1] - 1)),
        "visual_centroid_y": float((lum * ys).sum() / total / max(1, lum.shape[0] - 1)),
        "visual_change": float((diff > change_threshold).mean()),
        "visual_object_count": float(min(1.0, count / 16.0)),
        "visual_object_area": float(bright.mean()),
        "visual_quadrants": quadrants,
    }
    return features, lum


class VisualFeatureExtractor:
    def __init__(self, grid: Tuple[int, int] = (24, 32)):
        self.grid = grid
        self._previous: Optional[np.ndarray] = None       # tiny luminance grid only; never exported
        self._last_mean: Optional[float] = None
        self.flicker = 0.0
        self.frames = 0

    def reset(self) -> None:
        self._previous, self._last_mean, self.flicker = None, None, 0.0

    def extract(self, frame: np.ndarray) -> Dict[str, float]:
        features, lum = extract_visual_features(frame, self._previous, self.grid)
        mean = features["visual_luminance"]
        if self._last_mean is not None:
            self.flicker = 0.9 * self.flicker + 0.1 * abs(mean - self._last_mean)
        features["visual_flicker"] = float(min(1.0, 10.0 * self.flicker))
        self._previous, self._last_mean = lum, mean
        self.frames += 1
        return features
