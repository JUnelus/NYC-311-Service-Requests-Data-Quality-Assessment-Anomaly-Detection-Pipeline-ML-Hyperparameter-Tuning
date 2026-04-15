from __future__ import annotations

from pathlib import Path

try:
    from scripts.data_quality_check import profile_data
    from scripts.fetch_nyc311 import fetch_nyc311
    from scripts.hyperparameter_tuning import tune_hyperparams
    from scripts.ml_anomaly_detection import detect_anomalies
    from scripts.pipeline_utils import (
        DEFAULT_ANOMALIES_PATH,
        DEFAULT_BEST_PARAMS_PATH,
        DEFAULT_QUALITY_REPORT_PATH,
        DEFAULT_RAW_DATA_PATH,
        DEFAULT_TUNING_LOG_PATH,
        DEFAULT_TUNING_PLOT_PATH,
    )
except ModuleNotFoundError:
    from data_quality_check import profile_data
    from fetch_nyc311 import fetch_nyc311
    from hyperparameter_tuning import tune_hyperparams
    from ml_anomaly_detection import detect_anomalies
    from pipeline_utils import (
        DEFAULT_ANOMALIES_PATH,
        DEFAULT_BEST_PARAMS_PATH,
        DEFAULT_QUALITY_REPORT_PATH,
        DEFAULT_RAW_DATA_PATH,
        DEFAULT_TUNING_LOG_PATH,
        DEFAULT_TUNING_PLOT_PATH,
    )


def run_pipeline(limit: int = 10000) -> dict[str, str]:
    dataset_path = Path(DEFAULT_RAW_DATA_PATH)

    print("[1/4] Fetching NYC 311 data...")
    fetch_nyc311(limit=limit, output_csv=dataset_path)

    print("[2/4] Profiling dataset...")
    profile_data(path=dataset_path, report_path=DEFAULT_QUALITY_REPORT_PATH)

    print("[3/4] Detecting anomalies...")
    detect_anomalies(input_csv=dataset_path, output_csv=DEFAULT_ANOMALIES_PATH)

    print("[4/4] Tuning hyperparameters...")
    tune_hyperparams(
        input_csv=dataset_path,
        log_csv=DEFAULT_TUNING_LOG_PATH,
        plot_file=DEFAULT_TUNING_PLOT_PATH,
        best_params_file=DEFAULT_BEST_PARAMS_PATH,
    )

    outputs = {
        "dataset": str(DEFAULT_RAW_DATA_PATH),
        "quality_report": str(DEFAULT_QUALITY_REPORT_PATH),
        "anomalies": str(DEFAULT_ANOMALIES_PATH),
        "tuning_log": str(DEFAULT_TUNING_LOG_PATH),
        "best_params": str(DEFAULT_BEST_PARAMS_PATH),
        "tuning_plot": str(DEFAULT_TUNING_PLOT_PATH),
    }
    print("\nPipeline completed. Generated artifacts:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    return outputs


if __name__ == "__main__":
    run_pipeline()
