from pathlib import Path
import sys

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from pathlib import Path
import numpy as np
import pandas as pd

from src.preprocessing import preprocess_fdr
from src.windowing import WindowSpec, make_windows
from src.features import window_features

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from src.plotting import plot_score_timeline, top_windows


def zscore_baseline(X: np.ndarray) -> np.ndarray:
    """
    Simple baseline: z-score on tabular features -> per-window anomaly score
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-12
    z = np.abs((X - mu) / sd)
    return z.mean(axis=1)


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "raw" / "fdr_sample_flight.csv"
    results_dir = root / "results" / "baselines"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    data = preprocess_fdr(str(data_path))
    df = data["scaled"]  # use scaled for distance-based models

    # windows -> features
    spec = WindowSpec(window_sec=30.0, step_sec=15.0)
    Xw, ranges, fs = make_windows(df, spec)
    X = window_features(Xw)

    print("fs_hz:", fs)
    print("windows:", Xw.shape, "tabular:", X.shape)

    scores = {}

    # Baseline
    scores["zscore"] = zscore_baseline(X)

    # Isolation Forest (higher score = more anomalous)
    iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=0)
    iso.fit(X)
    scores["isoforest"] = -iso.score_samples(X)

    # One-Class SVM
    oc = OneClassSVM(kernel="rbf", nu=0.01, gamma="scale")
    oc.fit(X)
    scores["ocsvm"] = -oc.score_samples(X)

    # LOF (novelty=True so we can score on training data)
    lof = LocalOutlierFactor(n_neighbors=35, novelty=True)
    lof.fit(X)
    scores["lof"] = -lof.score_samples(X)

    # GMM negative log-likelihood
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0)
    gmm.fit(X)
    scores["gmm_nll"] = -gmm.score_samples(X)

    # DBSCAN: score = distance to nearest core (rough heuristic)
    # If DBSCAN labels a point as noise (-1), we score it high.
    db = DBSCAN(eps=2.5, min_samples=15).fit(X)
    scores["dbscan_noise"] = (db.labels_ == -1).astype(float)

    # Save results
    out = pd.DataFrame({
        "start_time": [r[0] for r in ranges],
        "end_time": [r[1] for r in ranges],
        **{k: v for k, v in scores.items()}
    })

    out_path = results_dir / "baseline_scores.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)


    # Reload scores from disk (or use `out` directly)
    df_scores = out

    # Make plots for key models
    plot_score_timeline(df_scores, "isoforest", results_dir / "timeline_isoforest.png")
    plot_score_timeline(df_scores, "lof", results_dir / "timeline_lof.png")
    plot_score_timeline(df_scores, "ocsvm", results_dir / "timeline_ocsvm.png")
    plot_score_timeline(df_scores, "gmm_nll", results_dir / "timeline_gmm.png")

    # Save top anomalous windows tables
    top_windows(df_scores, "isoforest", 20).to_csv(results_dir / "top20_isoforest.csv", index=False)
    top_windows(df_scores, "lof", 20).to_csv(results_dir / "top20_lof.csv", index=False)
    top_windows(df_scores, "ocsvm", 20).to_csv(results_dir / "top20_ocsvm.csv", index=False)
    top_windows(df_scores, "gmm_nll", 20).to_csv(results_dir / "top20_gmm.csv", index=False)

    print("Saved plots + top20 tables in results/")

if __name__ == "__main__":
    main()
