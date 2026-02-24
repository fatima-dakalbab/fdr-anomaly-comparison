from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler

from src.preprocessing import load_raw_csv, drop_fully_missing_columns, convert_mixed_numeric, select_numeric, fill_missing, set_time_index
from src.windowing import WindowSpec, make_windows
from src.features import window_features
from src.evaluation.injection import (
    choose_sensors,
    inject_spike,
    inject_drift,
    inject_stuck,
    inject_dropout,
    inject_noise_burst,
)
from src.evaluation.metrics import label_windows_from_time_mask, detection_at_k, false_alarm_rate


def score_models(X: np.ndarray) -> dict[str, np.ndarray]:
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


def preprocess_for_eval(csv_path: str):
    """
    Preprocess but keep it transparent for injection:
      - load raw
      - drop fully missing
      - convert mixed numeric
      - numeric only
      - fill
      - set Session Time index
      - robust scale
    """
    df = load_raw_csv(csv_path)
    t = df["Session Time"]
    df = drop_fully_missing_columns(df)
    df = convert_mixed_numeric(df)
    df_num = select_numeric(df)
    df_num = fill_missing(df_num)
    df_num = set_time_index(df_num, t)

    # ✅ Force numeric float time index (critical)
    df_num.index = pd.to_numeric(df_num.index, errors="coerce")
    df_num = df_num[~df_num.index.isna()]
    df_num = df_num.sort_index()
    df_num.index = df_num.index.astype(float)

    scaler = RobustScaler()
    Xs = scaler.fit_transform(df_num.values)
    df_scaled = pd.DataFrame(Xs, index=df_num.index, columns=df_num.columns)

    # ensure sorted
    df_scaled = df_scaled.sort_index()
    return df_num, df_scaled


def main():
    root = PROJECT_ROOT
    data_path = root / "data" / "raw" / "fdr_sample_flight.csv"
    out_dir = root / "results" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_num, df_scaled = preprocess_for_eval(str(data_path))
    print("Index dtype:", df_scaled.index.dtype)
    print("Index range:", float(df_scaled.index.min()), "to", float(df_scaled.index.max()))

    # --- Choose injection interval by existing samples (guaranteed overlap) ---
    inj_len_sec = 60.0  # inject over 60 seconds
    times = df_scaled.index.to_numpy(dtype=float)

    # we already estimated fs from windowing, but for injection interval we can infer from index
    dt = np.diff(times)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    fs_est = 1.0 / np.median(dt)

    inj_points = int(round(inj_len_sec * fs_est))

    # choose center in the middle of available samples
    center = len(times) // 2
    start_i = max(0, center - inj_points // 2)
    end_i = min(len(times) - 1, start_i + inj_points)

    t_start = float(times[start_i])
    t_end = float(times[end_i])

    print(f"Interval selection by points: fs_est={fs_est:.2f}Hz inj_points={inj_points} start_i={start_i} end_i={end_i}")
    sanity_mask = (times >= t_start) & (times <= t_end)
    print("SANITY points in interval:", int(sanity_mask.sum()))


    # Window spec
    spec = WindowSpec(window_sec=30.0, step_sec=15.0)

    # Window original (not injected) just to know K
    Xw0, ranges0, fs = make_windows(df_scaled, spec)
    n_windows = len(ranges0)
    k = max(10, int(0.05 * n_windows))  # top 5% windows
    print(f"fs_hz={fs:.2f} windows={n_windows} top_k={k}")

    # Sensor subset for injection
    # Choose continuous sensors: highest std to ensure meaningful injection
    sensor_std = df_scaled.std().sort_values(ascending=False)
    sensors = sensor_std.head(5).index.tolist()
    print("Injecting into sensors:", sensors)
    print(f"Injection interval: [{t_start:.2f}, {t_end:.2f}] seconds")

    scenarios = [
        ("spike", lambda d: inject_spike(d, sensors, t_start, t_end, magnitude=4.0)),
        ("drift", lambda d: inject_drift(d, sensors, t_start, t_end, magnitude=3.0)),
        ("stuck", lambda d: inject_stuck(d, sensors, t_start, t_end)),
        ("dropout", lambda d: inject_dropout(d, sensors, t_start, t_end)),
        ("noise_burst", lambda d: inject_noise_burst(d, sensors, t_start, t_end, magnitude=2.0, seed=0)),
    ]

    rows = []

    for scenario_name, fn in scenarios:
        df_inj, mask_time = fn(df_scaled)

        # if dropout created NaNs, fill to keep model runnable
        df_inj = df_inj.ffill().bfill()

        Xw, ranges, _ = make_windows(df_inj, spec, fs_hz=fs)
        X = window_features(Xw)

        y = label_windows_from_time_mask(ranges, mask_time)

        scores = score_models(X)

        for model_name, s in scores.items():
            det = detection_at_k(s, y, k=k)
            far = false_alarm_rate(s, y, k=k)

            rows.append({
                "scenario": scenario_name,
                "model": model_name,
                "top_k": k,
                "detection_at_k": det,
                "false_alarm_rate_at_k": far,
                "n_windows": len(y),
                "n_injected_windows": int(y.sum()),
            })
        print(scenario_name, "mask_true_points=", int(mask_time.sum()))
        print(f"Done scenario: {scenario_name} (injected_windows={int(y.sum())})")

    summary = pd.DataFrame(rows).sort_values(["scenario", "model"]).reset_index(drop=True)
    out_path = out_dir / "injection_eval_summary.csv"
    summary.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
