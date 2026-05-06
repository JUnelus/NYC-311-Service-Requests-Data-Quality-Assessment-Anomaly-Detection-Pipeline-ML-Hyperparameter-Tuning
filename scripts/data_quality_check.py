from __future__ import annotations

import json

import pandas as pd

try:
    from scripts.pipeline_utils import (
        DEFAULT_QUALITY_REPORT_PATH,
        DEFAULT_RAW_DATA_PATH,
        ensure_parent_dir,
        resolve_project_path,
    )
except ModuleNotFoundError:
    from pipeline_utils import (
        DEFAULT_QUALITY_REPORT_PATH,
        DEFAULT_RAW_DATA_PATH,
        ensure_parent_dir,
        resolve_project_path,
    )


def profile_data(path=DEFAULT_RAW_DATA_PATH, report_path=DEFAULT_QUALITY_REPORT_PATH):
    input_path = resolve_project_path(path, DEFAULT_RAW_DATA_PATH)
    output_report_path = resolve_project_path(report_path, DEFAULT_QUALITY_REPORT_PATH)

    df = pd.read_csv(input_path)
    report = {
        "source_path": str(input_path),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": {column: int(value) for column, value in df.isnull().sum().items()},
        "missing_pct": {column: round(float(value), 4) for column, value in (df.isnull().mean() * 100).items()},
        "top_complaint_types": df["complaint_type"].value_counts().head(10).to_dict() if "complaint_type" in df.columns else {},
        "borough_distribution": df["borough"].value_counts().to_dict() if "borough" in df.columns else {},
    }

    ensure_parent_dir(output_report_path)
    with open(output_report_path, "w", encoding="utf-8", newline="\n") as report_file:
        json.dump(report, report_file, indent=2)

    print("Data shape:", df.shape)
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nSample stats:\n", df.describe(include='all'))
    print("\nSample rows:\n", df.head())
    print(f"\nQuality report saved to {output_report_path}")
    return df


if __name__ == "__main__":
    profile_data()
