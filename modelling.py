import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load & Preprocessing Sederhana (Harus sama dengan tahap eksperimen)
file_path = "default of credit card clients.xls"
df = pd.read_excel(file_path, header=1)
df.rename(columns={'default payment next month': 'Y'}, inplace=True)

if 'ID' in df.columns:
    df.drop('ID', axis=1, inplace=True)

# Sinkronisasi Kategori (seperti di eksperimen)
df['X3'] = df['EDUCATION'].replace([0, 5, 6], 4)
df['X4'] = df['MARRIAGE'].replace([0], 3)

X = df.drop(['Y', 'EDUCATION', 'MARRIAGE'], axis=1)
y = df['Y']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Manual Logging ke MLflow
# Pastikan mlflow server sudah jalan atau akan disimpan secara lokal di folder mlruns
with mlflow.start_run(run_name="CreditCard_RF_Model"):
    params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    # Log Parameters & Metrics
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1", f1_score(y_test, y_pred))
    
    # Log Model (Ini yang akan jadi artefak)
    mlflow.sklearn.log_model(rf, "model")
    
    print("Training selesai dan berhasil dicatat di MLflow!")