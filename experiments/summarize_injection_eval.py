from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd


def main():
    root = PROJECT_ROOT
    path = root / "results" / "evaluation" / "injection_eval_summary.csv"
    out_dir = root / "results" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path)

    # Average across scenarios for each model
    model_avg = (
        df.groupby("model", as_index=False)[["detection_at_k", "false_alarm_rate_at_k"]]
        .mean()
    )

    # A simple combined score: high detection, low FAR
    # (You can report both metrics; this is just for ranking convenience.)
    model_avg["combined_score"] = model_avg["detection_at_k"] - model_avg["false_alarm_rate_at_k"]

    model_avg = model_avg.sort_values("combined_score", ascending=False).reset_index(drop=True)

    # Also scenario-wise best model
    scenario_best = (
        df.sort_values(["scenario", "detection_at_k", "false_alarm_rate_at_k"], ascending=[True, False, True])
        .groupby("scenario", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    model_avg.to_csv(out_dir / "model_ranking.csv", index=False)
    scenario_best.to_csv(out_dir / "best_model_per_scenario.csv", index=False)

    print("Saved:")
    print(" -", out_dir / "model_ranking.csv")
    print(" -", out_dir / "best_model_per_scenario.csv")

    print("\nTop models (by combined_score):")
    print(model_avg.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
