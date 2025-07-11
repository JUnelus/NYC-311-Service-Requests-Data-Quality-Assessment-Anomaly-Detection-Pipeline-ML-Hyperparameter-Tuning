import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
import itertools
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(project_root, "data", "nyc311_sample.csv")
log_path = os.path.join(project_root, "data", "hyperparam_search_results.csv")
plot_path = os.path.join(project_root, "data", "anomaly_score_hist.png")

def tune_hyperparams(input_csv):
    df = pd.read_csv(input_csv)
    features = ['agency', 'complaint_type', 'location_type', 'borough', 'incident_zip']
    features = [f for f in features if f in df.columns]
    df_features = df[features].astype(str).fillna("missing")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X = encoder.fit_transform(df_features)

    param_grid = {
        "n_estimators": [50, 100],
        "max_samples": ['auto', 0.7, 0.9],
        "contamination": [0.01, 0.05]
    }

    keys, values = zip(*param_grid.items())
    best_score = -np.inf
    best_params = None

    # Open the CSV file for logging
    with open(log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['n_estimators', 'max_samples', 'contamination', 'mean_anomaly_score'])

        print("Tuning IsolationForest params (manual grid search):")
        for v in itertools.product(*values):
            params = dict(zip(keys, v))
            model = IsolationForest(**params, random_state=42)
            model.fit(X)
            score = model.score_samples(X).mean()
            print(f"Params: {params} | Mean anomaly score: {score:.4f}")
            writer.writerow([params['n_estimators'], params['max_samples'], params['contamination'], score])
            if score > best_score:
                best_score = score
                best_params = params
                best_model = model  # Save the best model

    print(f"\nBest Params: {best_params} | Best mean anomaly score: {best_score:.4f}")

    # After grid search, plot the anomaly scores of the best model
    scores = best_model.score_samples(X)
    plt.hist(scores, bins=50)
    plt.title("Distribution of Isolation Forest Anomaly Scores (Best Params)")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.savefig(plot_path)
    print(f"Histogram saved to: {plot_path}")
    plt.show()

    return best_params

if __name__ == "__main__":
    tune_hyperparams(csv_path)