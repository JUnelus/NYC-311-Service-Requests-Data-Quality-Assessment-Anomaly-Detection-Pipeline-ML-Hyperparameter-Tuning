from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
sys.path.append("/path/to/your/scripts")

from scripts.fetch_nyc311 import fetch_nyc311
from scripts.data_quality_check import profile_data
from scripts.ml_anomaly_detection import detect_anomalies
from scripts.hyperparameter_tuning import tune_hyperparams

default_args = {"owner": "airflow", "retries": 1}
with DAG("nyc311_data_quality_pipeline", start_date=datetime(2024, 7, 1), schedule_interval="@daily", catchup=False, default_args=default_args) as dag:

    t0 = PythonOperator(
        task_id="fetch_nyc311",
        python_callable=fetch_nyc311,
    )
    t1 = PythonOperator(
        task_id="profile_data",
        python_callable=profile_data,
        op_kwargs={"path": "data/nyc311_sample.csv"}
    )
    t2 = PythonOperator(
        task_id="detect_anomalies",
        python_callable=detect_anomalies,
        op_kwargs={"input_csv": "data/nyc311_sample.csv"}
    )
    t3 = PythonOperator(
        task_id="tune_hyperparams",
        python_callable=tune_hyperparams,
        op_kwargs={"input_csv": "data/nyc311_sample.csv"}
    )

    t0 >> t1 >> [t2, t3]
