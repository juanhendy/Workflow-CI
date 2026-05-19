import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def run_retraining():
    # --- 1. PARSE ARGUMEN CLI ---
    parser = argparse.ArgumentParser(description="MLflow Project Retraining Script")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    # --- 2. AUTENTIKASI & INISIALISASI DAGSHUB ---
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if dagshub_token:
        dagshub.auth.add_app_token(dagshub_token)

    repo_owner = 'juan10082002'
    repo_name  = 'Membangun_model'
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    mlflow.sklearn.autolog(disable=True)

    # --- 3. MEMBACA DATASET ---
    data_path = "MLProject/namadataset_preprocessing/cleaned_data.csv"
    if not os.path.exists(data_path):
        print(f"Error: File dataset tidak ditemukan di {data_path}")
        return

    df = pd.read_csv(data_path)
    X  = df.drop('target', axis=1)
    y  = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- 4. TRAINING & MLFLOW LOGGING ---
    mlflow.set_experiment("CI_Retraining_Experiment")

    with mlflow.start_run(run_name="CI_Automated_Retrain"):
        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth",    args.max_depth)
        mlflow.log_metric("accuracy",    acc)

        # Log model ke DagsHub (remote)
        mlflow.sklearn.log_model(rf, "model")

        current_run_id = mlflow.active_run().info.run_id

        # --- 5. SIMPAN MODEL LOKAL (untuk GitHub Artifact upload) ---
        # Disimpan ke ./model_output/ agar bisa diupload oleh workflow
        # tanpa perlu download ulang dari DagsHub
        local_model_path = "./model_output"
        os.makedirs(local_model_path, exist_ok=True)
        mlflow.sklearn.save_model(rf, local_model_path)
        print(f"Model berhasil disimpan lokal di: {local_model_path}")

        print(f"TARGET_RUN_ID:{current_run_id}")
        print(f"Retraining Success! Model Accuracy: {acc:.4f}")

if __name__ == "__main__":
    run_retraining()
