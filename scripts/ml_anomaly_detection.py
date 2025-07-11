import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(project_root, "data", "nyc311_sample.csv")

def detect_anomalies(input_csv):
    df = pd.read_csv(input_csv)
    # Pick a set of useful features for ML
    features = ['agency', 'complaint_type', 'location_type', 'borough', 'incident_zip']
    features = [f for f in features if f in df.columns]
    df_features = df[features].astype(str).fillna("missing")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X = encoder.fit_transform(df_features)
    model = IsolationForest(n_estimators=100, max_samples=0.9, contamination=0.01, random_state=42)
    df['anomaly'] = model.fit_predict(X)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})
    anomaly_path = os.path.join(project_root, "data", "nyc311_anomalies.csv")
    df[df['anomaly'] == 1].to_csv(anomaly_path, index=False)
    print(df['anomaly'].value_counts())
    print(f"Anomalies saved to {anomaly_path}")
    return df

if __name__ == "__main__":
    detect_anomalies(csv_path)