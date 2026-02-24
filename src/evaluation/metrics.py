from __future__ import annotations
import numpy as np
import pandas as pd


def label_windows_from_time_mask(
    window_ranges: list[tuple[float, float]],
    time_mask: pd.Series,
) -> np.ndarray:
    """
    Label each window as injected (1) if it overlaps injected timepoints.
    time_mask: boolean Series indexed by time (same index space as df)
    """
    t_index = time_mask.index.to_numpy(dtype=float)
    m = time_mask.to_numpy(dtype=bool)

    # quick lookup: times where injection happened
    injected_times = t_index[m]
    if injected_times.size == 0:
        return np.zeros(len(window_ranges), dtype=int)

    labels = np.zeros(len(window_ranges), dtype=int)
    for i, (a, b) in enumerate(window_ranges):
        # overlap check: any injected time in [a,b]
        # using searchsorted for speed
        left = np.searchsorted(injected_times, a, side="left")
        right = np.searchsorted(injected_times, b, side="right")
        labels[i] = 1 if right > left else 0
    return labels


def detection_at_k(scores: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """
    Among top-k highest anomaly scores, what fraction are injected windows?
    """
    if len(scores) == 0:
        return 0.0
    k = min(k, len(scores))
    idx = np.argsort(scores)[::-1][:k]
    return float(y_true[idx].mean()) if k > 0 else 0.0


def false_alarm_rate(scores: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """
    False alarm rate among top-k: fraction of top-k that are NOT injected.
    """
    return 1.0 - detection_at_k(scores, y_true, k)
