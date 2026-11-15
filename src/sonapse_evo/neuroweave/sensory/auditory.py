"""Local auditory feature extraction.  An audio buffer is reduced to a few scalars and discarded.

    amplitude   = RMS of the buffer
    loudness    = clip(4 * amplitude, 0, 1)
    band_k      = loudness * (power in band k / total power)   for low (< 300 Hz), mid, high (> 2 kHz)
    flux        = sum_k max(0, share_k - previous share_k) + |d amplitude|        (temporal change)
    onset       = 1 if flux > ``onset_threshold`` else 0
    periodicity = max normalised autocorrelation over lags >= ``min_lag``          (sound-pattern score)

Only the previous band shares and amplitude are retained; samples are never stored.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class AudioFeatureExtractor:
    def __init__(self, sample_rate: int = 16000, band_edges: Tuple[float, float] = (300.0, 2000.0),
                 onset_threshold: float = 0.2, min_lag: int = 16):
        self.sample_rate, self.band_edges = int(sample_rate), band_edges
        self.onset_threshold, self.min_lag = float(onset_threshold), int(min_lag)
        self.reset()

    def reset(self) -> None:
        self._shares = np.zeros(3)
        self._amplitude = 0.0
        self.buffers = 0

    def extract(self, samples: np.ndarray, sample_rate: int = None) -> Dict[str, float]:
        x = np.asarray(samples, dtype=np.float64).ravel()
        if x.size < 8 or not np.all(np.isfinite(x)):
            raise ValueError("audio buffer must contain at least 8 finite samples")
        rate = int(sample_rate or self.sample_rate)
        amplitude = float(np.sqrt(np.mean(x ** 2)))
        loudness = float(np.clip(4.0 * amplitude, 0.0, 1.0))
        power = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
        freqs = np.fft.rfftfreq(x.size, d=1.0 / rate)
        lo, hi = self.band_edges
        bands = np.array([power[(freqs > 20) & (freqs < lo)].sum(), power[(freqs >= lo) & (freqs < hi)].sum(), power[freqs >= hi].sum()])
        shares = bands / (bands.sum() + 1e-12)
        flux = float(np.clip(np.maximum(0.0, shares - self._shares).sum() * loudness + abs(amplitude - self._amplitude), 0.0, 1.0))
        centred = x - x.mean()
        energy = float(centred @ centred) + 1e-12
        max_lag = min(x.size // 2, 512)
        if max_lag > self.min_lag:
            auto = [float(centred[:-lag] @ centred[lag:]) / energy for lag in range(self.min_lag, max_lag)]
            periodicity = float(np.clip(max(auto), 0.0, 1.0))
        else:
            periodicity = 0.0
        self._shares, self._amplitude = shares, amplitude
        self.buffers += 1
        return {"audio_low": float(loudness * shares[0]), "audio_mid": float(loudness * shares[1]),
                "audio_high": float(loudness * shares[2]), "audio_amplitude": float(min(1.0, amplitude)),
                "audio_flux": flux, "audio_onset": 1.0 if flux > self.onset_threshold else 0.0, "audio_periodicity": periodicity}
