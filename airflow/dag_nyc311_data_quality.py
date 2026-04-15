import os
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AIRFLOW_HOME = PROJECT_ROOT / ".airflow_home"
AIRFLOW_HOME.mkdir(parents=True, exist_ok=True)
airflow_db_path = (AIRFLOW_HOME / "airflow.db").resolve().as_posix()
airflow_sqlalchemy_conn = f"sqlite:///{airflow_db_path}"
os.environ.setdefault("AIRFLOW_HOME", str(AIRFLOW_HOME))
os.environ.setdefault("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN", airflow_sqlalchemy_conn)
os.environ.setdefault("AIRFLOW__CORE__SQL_ALCHEMY_CONN", airflow_sqlalchemy_conn)

removed_sys_paths = []
for entry in list(sys.path):
    resolved_entry = Path(entry or ".").resolve()
    if resolved_entry == PROJECT_ROOT:
        sys.path.remove(entry)
        removed_sys_paths.append(entry)

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

for entry in reversed(removed_sys_paths):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from scripts.fetch_nyc311 import fetch_nyc311
from scripts.data_quality_check import profile_data
from scripts.ml_anomaly_detection import detect_anomalies
from scripts.hyperparameter_tuning import tune_hyperparams
from scripts.pipeline_utils import DEFAULT_RAW_DATA_PATH

default_args = {"owner": "airflow", "retries": 1}
with DAG("nyc311_data_quality_pipeline", start_date=datetime(2024, 7, 1), schedule="@daily", catchup=False, default_args=default_args) as dag:

    dataset_path = str(DEFAULT_RAW_DATA_PATH)

    t0 = PythonOperator(
        task_id="fetch_nyc311",
        python_callable=fetch_nyc311,
        op_kwargs={"output_csv": dataset_path},
    )
    t1 = PythonOperator(
        task_id="profile_data",
        python_callable=profile_data,
        op_kwargs={"path": dataset_path}
    )
    t2 = PythonOperator(
        task_id="detect_anomalies",
        python_callable=detect_anomalies,
        op_kwargs={"input_csv": dataset_path}
    )
    t3 = PythonOperator(
        task_id="tune_hyperparams",
        python_callable=tune_hyperparams,
        op_kwargs={"input_csv": dataset_path}
    )

    t0 >> t1 >> [t2, t3]
