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
    parser.add_argument("--max_depth",    type=int, default=10)
    args = parser.parse_args()

    # --- 2. AUTENTIKASI & INISIALISASI DAGSHUB ---
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if dagshub_token:
        dagshub.auth.add_app_token(dagshub_token)

    dagshub.init(repo_owner='juan10082002', repo_name='Membangun_model', mlflow=True)
    mlflow.sklearn.autolog(disable=True)

    # --- 3. MEMBACA DATASET ---
    # Cari dataset dari beberapa kemungkinan path
    possible_paths = [
        "MLProject/namadataset_preprocessing/cleaned_data.csv",
        "namadataset_preprocessing/cleaned_data.csv",
        "../namadataset_preprocessing/cleaned_data.csv",
    ]
    data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            break

    if data_path is None:
        raise FileNotFoundError("Dataset tidak ditemukan di path manapun!")

    print(f"Dataset ditemukan di: {data_path}")
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
            max_depth=args.max_depth if args.max_depth > 0 else None,
            random_state=42
        )
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth",    args.max_depth)
        mlflow.log_metric("accuracy",    acc)
        mlflow.sklearn.log_model(rf, "model")

        current_run_id = mlflow.active_run().info.run_id

        # --- 5. SIMPAN MODEL KE GITHUB_WORKSPACE/model_output ---
        # GITHUB_WORKSPACE = root folder runner (bukan subfolder MLProject)
        # Sehingga workflow bisa menemukan ./model_output di root
        workspace = os.getenv("GITHUB_WORKSPACE", os.getcwd())
        local_model_path = os.path.join(workspace, "model_output")
        os.makedirs(local_model_path, exist_ok=True)
        mlflow.sklearn.save_model(rf, local_model_path)

        print(f"Model disimpan lokal di: {local_model_path}")
        print(f"TARGET_RUN_ID:{current_run_id}")
        print(f"Retraining selesai! Accuracy: {acc:.4f}")

if __name__ == "__main__":
    run_retraining()
