import requests
import pandas as pd
import os

NYC311_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
PARAMS = {
    "$limit": 10000,
    "$order": "created_date DESC"
}

def fetch_nyc311():
    resp = requests.get(NYC311_URL, params=PARAMS)
    data = resp.json()
    df = pd.DataFrame(data)
    cols = ['unique_key', 'created_date', 'closed_date', 'agency', 'complaint_type', 'descriptor', 'location_type',
            'incident_zip', 'city', 'borough', 'latitude', 'longitude']
    df = df[[col for col in cols if col in df.columns]]
    # Resolve data directory outside 'scripts'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "nyc311_sample.csv")
    df.to_csv(csv_path, index=False)
    print(f"Fetched and saved {len(df)} records to {csv_path}.")
    return df

if __name__ == "__main__":
    fetch_nyc311()