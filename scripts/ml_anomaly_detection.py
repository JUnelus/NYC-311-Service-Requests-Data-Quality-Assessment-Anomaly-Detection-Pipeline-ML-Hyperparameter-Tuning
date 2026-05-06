from __future__ import annotations

from sklearn.ensemble import IsolationForest

try:
    from scripts.pipeline_utils import (
        DEFAULT_ANOMALIES_PATH,
        DEFAULT_RAW_DATA_PATH,
        ensure_parent_dir,
        load_dataset,
        prepare_model_matrix,
        resolve_project_path,
    )
except ModuleNotFoundError:
    from pipeline_utils import (
        DEFAULT_ANOMALIES_PATH,
        DEFAULT_RAW_DATA_PATH,
        ensure_parent_dir,
        load_dataset,
        prepare_model_matrix,
        resolve_project_path,
    )

def detect_anomalies(input_csv=DEFAULT_RAW_DATA_PATH, output_csv=DEFAULT_ANOMALIES_PATH, contamination=0.01):
    df = load_dataset(input_csv)
    X, feature_info = prepare_model_matrix(df)

    model = IsolationForest(n_estimators=100, max_samples=0.9, contamination=contamination, random_state=42)
    model.fit(X)

    df['anomaly_score'] = model.score_samples(X)
    df['anomaly'] = model.predict(X)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

    anomaly_path = resolve_project_path(output_csv, DEFAULT_ANOMALIES_PATH)
    ensure_parent_dir(anomaly_path)
    df[df['anomaly'] == 1].sort_values('anomaly_score').to_csv(
        anomaly_path,
        index=False,
        lineterminator="\n",
    )

    print("Model features:", feature_info)
    print(df['anomaly'].value_counts())
    print(f"Anomalies saved to {anomaly_path}")
    return df


if __name__ == "__main__":
    detect_anomalies()
