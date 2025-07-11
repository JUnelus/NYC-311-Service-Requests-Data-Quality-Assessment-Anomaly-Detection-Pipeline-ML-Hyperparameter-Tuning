# NYC-311-Service-Requests-Data-Quality-Assessment-Anomaly-Detection-Pipeline-ML-Hyperparameter-Tuning
This project pulls live 311 service request data via NYC Open Data API, performs automated data profiling and validation, detects anomalies using machine learning (Isolation Forest), and optimizes the ML model’s hyperparameters.

# NYC 311 Data Quality ML Pipeline

This project pulls live data from NYC 311 Service Requests, performs data profiling, detects outliers/anomalies using Isolation Forest, and tunes ML model hyperparameters.

**Pipeline:**  
- Fetches up to 10,000 recent NYC 311 requests via public API
- Data profiling: missing values, descriptive stats
- Anomaly detection: Isolation Forest with categorical encoding
- Automated hyperparameter tuning with GridSearchCV

**Tech:** Python, Pandas, Scikit-learn, Airflow

**Live Data API:** [NYC 311 Service Requests - API](https://data.cityofnewyork.us/resource/erm2-nwe9.json)
