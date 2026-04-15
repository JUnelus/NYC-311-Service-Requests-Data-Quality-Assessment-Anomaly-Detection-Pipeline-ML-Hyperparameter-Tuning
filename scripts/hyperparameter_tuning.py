from __future__ import annotations

import itertools
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

try:
    from scripts.pipeline_utils import (
        DEFAULT_BEST_PARAMS_PATH,
        DEFAULT_RAW_DATA_PATH,
        DEFAULT_TUNING_LOG_PATH,
        DEFAULT_TUNING_PLOT_PATH,
        ensure_parent_dir,
        load_dataset,
        prepare_model_matrix,
        resolve_project_path,
    )
except ModuleNotFoundError:
    from pipeline_utils import (
        DEFAULT_BEST_PARAMS_PATH,
        DEFAULT_RAW_DATA_PATH,
        DEFAULT_TUNING_LOG_PATH,
        DEFAULT_TUNING_PLOT_PATH,
        ensure_parent_dir,
        load_dataset,
        prepare_model_matrix,
        resolve_project_path,
    )


def tune_hyperparams(
    input_csv=DEFAULT_RAW_DATA_PATH,
    log_csv=DEFAULT_TUNING_LOG_PATH,
    plot_file=DEFAULT_TUNING_PLOT_PATH,
    best_params_file=DEFAULT_BEST_PARAMS_PATH,
):
    df = load_dataset(input_csv)
    X, feature_info = prepare_model_matrix(df)

    log_path = resolve_project_path(log_csv, DEFAULT_TUNING_LOG_PATH)
    plot_path = resolve_project_path(plot_file, DEFAULT_TUNING_PLOT_PATH)
    best_params_path = resolve_project_path(best_params_file, DEFAULT_BEST_PARAMS_PATH)
    ensure_parent_dir(log_path)
    ensure_parent_dir(plot_path)
    ensure_parent_dir(best_params_path)

    param_grid = {
        "n_estimators": [50, 100],
        "max_samples": ['auto', 0.7, 0.9],
        "contamination": [0.01, 0.05]
    }

    keys, values = zip(*param_grid.items())
    best_score = -np.inf
    best_params = None
    best_summary = None
    best_model = None

    with open(log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'n_estimators',
            'max_samples',
            'contamination',
            'mean_anomaly_score',
            'score_std',
            'score_gap_p50_p01',
            'predicted_anomaly_rate',
        ])

        print("Tuning IsolationForest params (manual grid search):")
        for v in itertools.product(*values):
            params = dict(zip(keys, v))
            model = IsolationForest(**params, random_state=42)
            model.fit(X)
            scores = model.score_samples(X)
            mean_score = float(scores.mean())
            score_std = float(scores.std())
            score_gap = float(np.percentile(scores, 50) - np.percentile(scores, 1))
            anomaly_rate = float((model.predict(X) == -1).mean())

            print(
                f"Params: {params} | Mean anomaly score: {mean_score:.4f} | "
                f"Score gap (P50-P01): {score_gap:.4f} | Predicted anomaly rate: {anomaly_rate:.4%}"
            )
            writer.writerow([
                params['n_estimators'],
                params['max_samples'],
                params['contamination'],
                mean_score,
                score_std,
                score_gap,
                anomaly_rate,
            ])
            if score_gap > best_score:
                best_score = score_gap
                best_params = params
                best_model = model
                best_summary = {
                    "feature_info": feature_info,
                    "mean_anomaly_score": mean_score,
                    "score_std": score_std,
                    "score_gap_p50_p01": score_gap,
                    "predicted_anomaly_rate": anomaly_rate,
                }

    print(f"\nBest Params: {best_params} | Best score gap (P50-P01): {best_score:.4f}")

    with open(best_params_path, "w", encoding="utf-8") as best_params_handle:
        json.dump({"best_params": best_params, **best_summary}, best_params_handle, indent=2)

    scores = best_model.score_samples(X)
    plt.hist(scores, bins=50)
    plt.title("Distribution of Isolation Forest Anomaly Scores (Best Params)")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Histogram saved to: {plot_path}")
    print(f"Best parameter summary saved to: {best_params_path}")

    return best_params


if __name__ == "__main__":
    tune_hyperparams()
