import pandas as pd
import os

# Get project root and data path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(project_root, "data", "nyc311_sample.csv")  # Or use your absolute file path

def profile_data(path):
    df = pd.read_csv(path)
    print("Data shape:", df.shape)
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nSample stats:\n", df.describe(include='all'))
    print("\nSample rows:\n", df.head())
    return df

if __name__ == "__main__":
    profile_data(csv_path)