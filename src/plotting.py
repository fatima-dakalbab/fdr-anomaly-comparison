from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_score_timeline(df_scores: pd.DataFrame, score_col: str, outpath: str | Path):
    """
    Plot anomaly score over time (window midpoints).
    """
    mid = (df_scores["start_time"] + df_scores["end_time"]) / 2.0
    y = df_scores[score_col].astype(float)

    plt.figure()
    plt.plot(mid, y)
    plt.xlabel("Session Time (s)")
    plt.ylabel(f"Anomaly score: {score_col}")
    plt.title(f"Anomaly timeline ({score_col})")
    plt.tight_layout()

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def top_windows(df_scores: pd.DataFrame, score_col: str, k: int = 20) -> pd.DataFrame:
    """
    Return top-k anomalous windows for a given score.
    """
    return df_scores.sort_values(score_col, ascending=False).head(k)[
        ["start_time", "end_time", score_col]
    ].reset_index(drop=True)
