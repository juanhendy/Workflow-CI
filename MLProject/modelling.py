import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

def run_retraining():
    # --- 1. PARSE ARGUMEN CLI ---
    parser = argparse.ArgumentParser(description="MLflow Project Retraining Script")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth",    type=int, default=10)
    args = parser.parse_args()

    # --- 2. AUTENTIKASI DAGSHUB (Dihapus karena sudah di-handle oleh GitHub Env) ---
    mlflow.sklearn.autolog(disable=True)

    # --- 3. MEMBACA DATASET ---
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
    # CATATAN: mlflow.set_experiment dihapus agar mengikuti Experiment ID dari perintah 'mlflow run'

    # nested=True tetap dipertahankan karena berjalan di dalam lingkungan 'mlflow run'
    with mlflow.start_run(run_name="CI_Automated_Retrain", nested=True):
        max_depth = args.max_depth if args.max_depth > 0 else None
        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth",    args.max_depth)
        mlflow.log_metric("accuracy",    acc)
        mlflow.log_metric("f1_score",    f1)
        mlflow.log_metric("precision",   prec)
        mlflow.log_metric("recall",      rec)
        mlflow.sklearn.log_model(rf, "model")

        # --- 5. SIMPAN MODEL LOKAL → model_output/ (untuk artifact upload CI) ---
        workspace    = os.getenv("GITHUB_WORKSPACE", os.getcwd())
        model_output = os.path.join(workspace, "model_output")
        
        # Bersihkan folder jika sudah ada isinya agar mlflow.sklearn.save_model tidak error
        if os.path.exists(model_output):
            import shutil
            shutil.rmtree(model_output)
            
        os.makedirs(model_output, exist_ok=True)
        mlflow.sklearn.save_model(rf, model_output)

        print(f"Model disimpan di : {model_output}")
        print(f"Accuracy          : {acc:.4f}")
        print(f"F1 Score          : {f1:.4f}")
        print(f"Precision         : {prec:.4f}")
        print(f"Recall            : {rec:.4f}")
        print("Retraining selesai!")

if __name__ == "__main__":
    run_retraining()