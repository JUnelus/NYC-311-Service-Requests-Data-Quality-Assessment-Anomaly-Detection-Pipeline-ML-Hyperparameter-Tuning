from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PROJECT_ROOT / "README.md"
DATA_DIR = PROJECT_ROOT / "data"

QUALITY_REPORT_PATH = DATA_DIR / "nyc311_quality_report.json"
BEST_PARAMS_PATH = DATA_DIR / "nyc311_best_hyperparams.json"
HYPERPARAM_RESULTS_PATH = DATA_DIR / "hyperparam_search_results.csv"
HISTOGRAM_PATH = DATA_DIR / "anomaly_score_hist.png"

START_MARKER = "<!-- AUTO_REPORT_START -->"
END_MARKER = "<!-- AUTO_REPORT_END -->"


def _format_number(value: object) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    return data


def _read_top_hyperparams(path: Path, limit: int = 5) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return []

    def _score(row: dict[str, str]) -> float:
        try:
            return float(row.get("score_gap_p50_p01", "-inf"))
        except ValueError:
            return float("-inf")

    rows.sort(key=_score, reverse=True)
    return rows[:limit]


def _markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No hyperparameter search results available."

    header = "| Rank | n_estimators | max_samples | contamination | score_gap_p50_p01 | predicted_anomaly_rate |"
    divider = "|---:|---:|---:|---:|---:|---:|"
    lines = [header, divider]

    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    row.get("n_estimators", "-"),
                    row.get("max_samples", "-"),
                    row.get("contamination", "-"),
                    row.get("score_gap_p50_p01", "-"),
                    row.get("predicted_anomaly_rate", "-"),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def build_auto_report_section() -> str:
    quality = _load_json(QUALITY_REPORT_PATH)
    best = _load_json(BEST_PARAMS_PATH)
    top_rows = _read_top_hyperparams(HYPERPARAM_RESULTS_PATH)

    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    top_complaints = quality.get("top_complaint_types", {})
    top_complaint_lines = []
    for name, count in list(top_complaints.items())[:5]:
        top_complaint_lines.append(f"- {name}: {_format_number(count)}")

    if not top_complaint_lines:
        top_complaint_lines.append("- No complaint type distribution available.")

    best_params = best.get("best_params", {})

    return "\n".join(
        [
            START_MARKER,
            "### Weekly Automated Report Snapshot",
            f"_Last updated: {generated_utc}_",
            "",
            "#### Data quality report (`data/nyc311_quality_report.json`)",
            f"- Row count: {_format_number(quality.get('row_count', '-'))}",
            f"- Column count: {_format_number(quality.get('column_count', '-'))}",
            f"- Duplicate rows: {_format_number(quality.get('duplicate_rows', '-'))}",
            "- Top complaint types:",
            *top_complaint_lines,
            "",
            "#### Best hyperparameters (`data/nyc311_best_hyperparams.json`)",
            f"- n_estimators: {_format_number(best_params.get('n_estimators', '-'))}",
            f"- max_samples: {_format_number(best_params.get('max_samples', '-'))}",
            f"- contamination: {_format_number(best_params.get('contamination', '-'))}",
            f"- Predicted anomaly rate: {_format_number(best.get('predicted_anomaly_rate', '-'))}",
            f"- Score gap (P50-P01): {_format_number(best.get('score_gap_p50_p01', '-'))}",
            "",
            "#### Hyperparameter search top runs (`data/hyperparam_search_results.csv`)",
            _markdown_table(top_rows),
            "",
            "#### Anomaly score distribution (`data/anomaly_score_hist.png`)",
            "![Latest anomaly score histogram](data/anomaly_score_hist.png)",
            END_MARKER,
        ]
    )


def update_readme() -> None:
    if not README_PATH.exists():
        raise FileNotFoundError(f"README not found at {README_PATH}")

    missing = [
        str(path)
        for path in [QUALITY_REPORT_PATH, BEST_PARAMS_PATH, HYPERPARAM_RESULTS_PATH, HISTOGRAM_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Required report artifacts are missing:\n" + "\n".join(missing))

    readme_text = README_PATH.read_text(encoding="utf-8")
    generated_section = build_auto_report_section()

    if START_MARKER in readme_text and END_MARKER in readme_text:
        start = readme_text.index(START_MARKER)
        end = readme_text.index(END_MARKER) + len(END_MARKER)
        updated = readme_text[:start] + generated_section + readme_text[end:]
    else:
        updated = readme_text.rstrip() + "\n\n" + generated_section + "\n"

    README_PATH.write_text(updated, encoding="utf-8")
    print(f"README updated: {README_PATH}")


if __name__ == "__main__":
    update_readme()

