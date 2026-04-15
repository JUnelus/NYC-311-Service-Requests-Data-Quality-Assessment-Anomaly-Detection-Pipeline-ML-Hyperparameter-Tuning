from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_RAW_DATA_PATH = DATA_DIR / "nyc311_sample.csv"
DEFAULT_ANOMALIES_PATH = DATA_DIR / "nyc311_anomalies.csv"
DEFAULT_TUNING_LOG_PATH = DATA_DIR / "hyperparam_search_results.csv"
DEFAULT_TUNING_PLOT_PATH = DATA_DIR / "anomaly_score_hist.png"
DEFAULT_BEST_PARAMS_PATH = DATA_DIR / "nyc311_best_hyperparams.json"
DEFAULT_QUALITY_REPORT_PATH = DATA_DIR / "nyc311_quality_report.json"

PathLike = Union[str, Path]


def resolve_project_path(path: Union[PathLike, None], fallback: Path) -> Path:
    if path is None:
        return fallback

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def ensure_parent_dir(path: PathLike) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def load_dataset(path: Union[PathLike, None] = None) -> pd.DataFrame:
    dataset_path = resolve_project_path(path, DEFAULT_RAW_DATA_PATH)
    return pd.read_csv(dataset_path)


def prepare_model_matrix(df: pd.DataFrame) -> Tuple[object, Dict[str, list]]:
    working = df.copy()

    created_ts = pd.Series(pd.NaT, index=working.index, dtype="datetime64[ns]")
    closed_ts = pd.Series(pd.NaT, index=working.index, dtype="datetime64[ns]")

    if "created_date" in working.columns:
        created_ts = pd.to_datetime(working["created_date"], errors="coerce")
        working["created_hour"] = created_ts.dt.hour
        working["created_dayofweek"] = created_ts.dt.dayofweek
        working["created_month"] = created_ts.dt.month

    if "closed_date" in working.columns:
        closed_ts = pd.to_datetime(working["closed_date"], errors="coerce")

    if "created_date" in working.columns and "closed_date" in working.columns:
        working["resolution_hours"] = (closed_ts - created_ts).dt.total_seconds() / 3600.0

    categorical_features = [
        column
        for column in [
            "agency",
            "complaint_type",
            "descriptor",
            "location_type",
            "borough",
            "city",
            "incident_zip",
        ]
        if column in working.columns
    ]
    numeric_features = [
        column
        for column in [
            "latitude",
            "longitude",
            "created_hour",
            "created_dayofweek",
            "created_month",
            "resolution_hours",
        ]
        if column in working.columns
    ]

    if not categorical_features and not numeric_features:
        raise ValueError("No supported feature columns were found for anomaly modeling.")

    if categorical_features:
        working[categorical_features] = working[categorical_features].fillna("missing").astype(str)

    transformers = []
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]),
                categorical_features,
            )
        )
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    X = preprocessor.fit_transform(working)
    feature_info = {
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
    }
    return X, feature_info

