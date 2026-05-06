from __future__ import annotations

import pandas as pd
import requests

try:
    from scripts.pipeline_utils import DEFAULT_RAW_DATA_PATH, ensure_parent_dir, resolve_project_path
except ModuleNotFoundError:
    from pipeline_utils import DEFAULT_RAW_DATA_PATH, ensure_parent_dir, resolve_project_path

NYC311_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
DEFAULT_LIMIT = 10000

def fetch_nyc311(limit=DEFAULT_LIMIT, output_csv=None, timeout=30):
    params = {
        "$limit": limit,
        "$order": "created_date DESC",
    }
    resp = requests.get(NYC311_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data)
    cols = ['unique_key', 'created_date', 'closed_date', 'agency', 'complaint_type', 'descriptor', 'location_type',
            'incident_zip', 'city', 'borough', 'latitude', 'longitude']
    df = df[[col for col in cols if col in df.columns]]
    csv_path = resolve_project_path(output_csv, DEFAULT_RAW_DATA_PATH)
    ensure_parent_dir(csv_path)
    df.to_csv(csv_path, index=False, lineterminator="\n")
    print(f"Fetched and saved {len(df)} records to {csv_path}.")
    return df

if __name__ == "__main__":
    fetch_nyc311()