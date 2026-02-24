from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class WindowSpec:
    window_sec: float = 30.0
    step_sec: float = 15.0


def estimate_fs_hz(time_index: pd.Index) -> float:
    """
    Estimate sampling frequency from Session Time index (seconds).
    """
    t = np.asarray(time_index, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("Cannot estimate sampling rate: time index has no valid increments.")
    median_dt = float(np.median(dt))
    return 1.0 / median_dt


def make_windows(df: pd.DataFrame, spec: WindowSpec, fs_hz: float | None = None):
    """
    Slice a time-indexed dataframe into overlapping windows.
    Returns:
      X_windows: np.ndarray shape (n_windows, window_len, n_features)
      window_ranges: list of (start_time, end_time)
      fs_hz: sampling rate used
    """
    if fs_hz is None:
        fs_hz = estimate_fs_hz(df.index)

    win_len = int(round(spec.window_sec * fs_hz))
    step_len = int(round(spec.step_sec * fs_hz))

    if win_len <= 1:
        raise ValueError("Window too small. Increase window_sec or check fs_hz.")
    if step_len <= 0:
        raise ValueError("Step too small. Increase step_sec or check fs_hz.")

    values = df.values.astype(float, copy=False)
    n, d = values.shape

    windows = []
    ranges = []

    for start in range(0, n - win_len + 1, step_len):
        end = start + win_len
        w = values[start:end, :]
        windows.append(w)
        ranges.append((float(df.index[start]), float(df.index[end - 1])))

    X = np.stack(windows, axis=0) if windows else np.empty((0, win_len, d), dtype=float)
    return X, ranges, fs_hz
