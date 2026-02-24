from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.preprocessing import preprocess_fdr
from src.windowing import WindowSpec, make_windows
from src.features import window_features
from src.feature_groups import assign_groups, summarize
from src.plotting import plot_score_timeline, top_windows

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN


def run_models(X: np.ndarray) -> dict[str, np.ndarray]:
    scores = {}

    iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=0)
    iso.fit(X)
    scores["isoforest"] = -iso.score_samples(X)

    oc = OneClassSVM(kernel="rbf", nu=0.01, gamma="scale")
    oc.fit(X)
    scores["ocsvm"] = -oc.score_samples(X)

    lof = LocalOutlierFactor(n_neighbors=35, novelty=True)
    lof.fit(X)
    scores["lof"] = -lof.score_samples(X)

    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0)
    gmm.fit(X)
    scores["gmm_nll"] = -gmm.score_samples(X)

    db = DBSCAN(eps=2.5, min_samples=15).fit(X)
    scores["dbscan_noise"] = (db.labels_ == -1).astype(float)

    return scores


def main():
    root = PROJECT_ROOT
    data_path = root / "data" / "raw" / "fdr_sample_flight.csv"

    data = preprocess_fdr(str(data_path))
    df = data["scaled"]  # scaled numeric

    groups = assign_groups(df.columns.tolist())
    summarize(groups)

    spec = WindowSpec(window_sec=30.0, step_sec=15.0)

    for group_name in ["flight_dynamics", "engine", "controls", "navigation"]:
        cols = groups.get(group_name, [])
        if len(cols) < 5:
            print(f"Skipping {group_name}: too few features ({len(cols)})")
            continue

        out_dir = root / "results" / "grouped" / group_name
        out_dir.mkdir(parents=True, exist_ok=True)

        df_g = df[cols]

        Xw, ranges, fs = make_windows(df_g, spec)
        X = window_features(Xw)

        print(f"\n[{group_name}] fs={fs:.2f} windows={Xw.shape} tabular={X.shape} features={len(cols)}")

        scores = run_models(X)

        out = pd.DataFrame({
            "start_time": [r[0] for r in ranges],
            "end_time": [r[1] for r in ranges],
            **scores
        })

        out.to_csv(out_dir / "scores.csv", index=False)

        # plots + top tables
        for model in ["isoforest", "lof", "ocsvm", "gmm_nll"]:
            plot_score_timeline(out, model, out_dir / f"timeline_{model}.png")
            top_windows(out, model, 20).to_csv(out_dir / f"top20_{model}.csv", index=False)

        # DBSCAN
        plot_score_timeline(out, "dbscan_noise", out_dir / "timeline_dbscan_noise.png")
        top_windows(out, "dbscan_noise", 20).to_csv(out_dir / "top20_dbscan_noise.csv", index=False)

        print(f"Saved grouped results to: {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
