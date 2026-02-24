from __future__ import annotations
import numpy as np


def _slope(x: np.ndarray) -> np.ndarray:
    """
    Compute slope per feature using a simple least squares fit over time.
    x: (T, D) -> returns (D,)
    """
    T, D = x.shape
    t = np.arange(T, dtype=float)
    t = (t - t.mean()) / (t.std() + 1e-12)

    # slope = cov(t, y) / var(t)
    vt = np.sum(t * t) + 1e-12
    return (t[:, None] * x).sum(axis=0) / vt


def window_features(X_windows: np.ndarray) -> np.ndarray:
    """
    Convert windows to tabular feature vectors.
    X_windows: (N, T, D)
    Output: (N, F) where F = 5*D (mean, std, min, max, slope)
    """
    if X_windows.ndim != 3:
        raise ValueError("X_windows must be 3D: (n_windows, window_len, n_features)")

    N, T, D = X_windows.shape
    eps = 1e-12

    mean = X_windows.mean(axis=1)
    std = X_windows.std(axis=1) + eps
    mn = X_windows.min(axis=1)
    mx = X_windows.max(axis=1)

    slopes = np.zeros((N, D), dtype=float)
    for i in range(N):
        slopes[i] = _slope(X_windows[i])

    # Concatenate features
    return np.concatenate([mean, std, mn, mx, slopes], axis=1)


def feature_names(sensor_names: list[str]) -> list[str]:
    """
    Names for the tabular features in the same order as window_features().
    """
    out = []
    for prefix in ["mean", "std", "min", "max", "slope"]:
        out.extend([f"{prefix}__{c}" for c in sensor_names])
    return out
