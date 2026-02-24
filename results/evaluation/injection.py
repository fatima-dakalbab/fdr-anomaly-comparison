from __future__ import annotations
import numpy as np
import pandas as pd


def _rng(seed: int | None):
    return np.random.default_rng(seed)


def choose_sensors(columns: list[str], k: int, seed: int | None = None) -> list[str]:
    r = _rng(seed)
    cols = list(columns)
    if k >= len(cols):
        return cols
    idx = r.choice(len(cols), size=k, replace=False)
    return [cols[i] for i in idx]


def inject_spike(
    df: pd.DataFrame,
    sensors: list[str],
    t_start: float,
    t_end: float,
    magnitude: float = 4.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Spike: add magnitude * std to selected sensors within [t_start, t_end].
    Returns (df_injected, mask_timepoints).
    """
    out = df.copy()
    mask = (out.index >= t_start) & (out.index <= t_end)
    std = out[sensors].std(axis=0).replace(0, 1e-6)
    out.loc[mask, sensors] = out.loc[mask, sensors] + magnitude * std.values
    return out, pd.Series(mask, index=out.index)


def inject_drift(
    df: pd.DataFrame,
    sensors: list[str],
    t_start: float,
    t_end: float,
    magnitude: float = 3.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Drift: add a ramp from 0 to magnitude*std over the interval.
    """
    out = df.copy()
    mask = (out.index >= t_start) & (out.index <= t_end)
    idx = np.where(mask.values)[0]
    if len(idx) == 0:
        return out, pd.Series(mask, index=out.index)

    ramp = np.linspace(0.0, 1.0, len(idx))
    std = out[sensors].std(axis=0).replace(0, 1e-6)

    add = (ramp[:, None] * (magnitude * std.values)[None, :])
    out.iloc[idx, out.columns.get_indexer(sensors)] = (
        out.iloc[idx, out.columns.get_indexer(sensors)].values + add
    )
    return out, pd.Series(mask, index=out.index)


def inject_stuck(
    df: pd.DataFrame,
    sensors: list[str],
    t_start: float,
    t_end: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Stuck sensor: flatline to the value at t_start.
    """
    out = df.copy()
    mask = (out.index >= t_start) & (out.index <= t_end)
    if mask.any():
        first_row = out.loc[mask].iloc[0][sensors]
        out.loc[mask, sensors] = first_row.values
    return out, pd.Series(mask, index=out.index)


def inject_dropout(
    df: pd.DataFrame,
    sensors: list[str],
    t_start: float,
    t_end: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Dropout: set NaNs in interval (then later preprocessing fill will affect it).
    Useful to see if pipeline/model is robust.
    """
    out = df.copy()
    mask = (out.index >= t_start) & (out.index <= t_end)
    out.loc[mask, sensors] = np.nan
    return out, pd.Series(mask, index=out.index)


def inject_noise_burst(
    df: pd.DataFrame,
    sensors: list[str],
    t_start: float,
    t_end: float,
    magnitude: float = 2.0,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Noise burst: add gaussian noise with std = magnitude*sensor_std.
    """
    out = df.copy()
    mask = (out.index >= t_start) & (out.index <= t_end)
    idx = np.where(mask.values)[0]
    if len(idx) == 0:
        return out, pd.Series(mask, index=out.index)

    r = _rng(seed)
    std = out[sensors].std(axis=0).replace(0, 1e-6).values
    noise = r.normal(0.0, magnitude * std, size=(len(idx), len(sensors)))

    out.iloc[idx, out.columns.get_indexer(sensors)] = (
        out.iloc[idx, out.columns.get_indexer(sensors)].values + noise
    )
    return out, pd.Series(mask, index=out.index)
