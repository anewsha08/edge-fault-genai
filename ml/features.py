"""Windowing and feature extraction.

Keep this simple on purpose: the same features must later be reimplemented in C on the
Cortex-M node (CMSIS-DSP FFT), and the C output must match this file within tolerance.

Feature vector (N_FEATURES = 5 + N_BANDS + 2 = 23 with defaults):
  0 rms            root mean square (after removing the mean)
  1 ptp            peak-to-peak
  2 crest          max(|x|) / rms
  3 kurtosis       fourth standardized moment (not excess); impulsive faults raise it
  4 skew           third standardized moment
  5..20 band_k     log10 energy in N_BANDS equal-width bands from 0 to fs/2
  21 peak_freq     frequency of the largest spectral bin, divided by fs/2
  22 centroid      spectral centroid, divided by fs/2
"""
import numpy as np

N_BANDS = 16
EPS = 1e-12


def window_signal(x, win=1024, hop=512):
    """Cut a 1-D signal into overlapping windows. Returns array (n_windows, win)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) < win:
        return np.empty((0, win))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def extract_features(w, n_bands=N_BANDS):
    """Features for one window w (1-D array). Frequencies are normalized, so fs is not needed."""
    w = np.asarray(w, dtype=np.float64)
    w = w - w.mean()
    rms = np.sqrt(np.mean(w**2))
    ptp = w.max() - w.min()
    crest = np.max(np.abs(w)) / (rms + EPS)
    s = w.std() + EPS
    kurt = np.mean((w / s) ** 4)
    skew = np.mean((w / s) ** 3)

    spec = np.abs(np.fft.rfft(w * np.hanning(len(w)))) ** 2
    spec = spec[1:]  # drop DC bin
    n = len(spec)
    edges = np.linspace(0, n, n_bands + 1).astype(int)
    bands = np.array([spec[edges[i]:edges[i + 1]].sum() for i in range(n_bands)])
    log_bands = np.log10(bands + EPS)

    freqs = (np.arange(n) + 1) / n  # normalized: 1.0 == fs/2
    peak_freq = freqs[np.argmax(spec)]
    centroid = (freqs * spec).sum() / (spec.sum() + EPS)

    return np.concatenate([[rms, ptp, crest, kurt, skew], log_bands, [peak_freq, centroid]])


def feature_names(n_bands=N_BANDS):
    return (["rms", "ptp", "crest", "kurtosis", "skew"]
            + [f"band_{i}" for i in range(n_bands)]
            + ["peak_freq", "centroid"])


def windows_to_features(windows):
    return np.array([extract_features(w) for w in windows])
