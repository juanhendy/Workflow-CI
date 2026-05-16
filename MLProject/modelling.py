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
    parser.add_argument("--n_estimators", type=int, default=100, help="Jumlah tree di RandomForest")
    parser.add_argument("--max_depth", type=int, default=10, help="Maksimal kedalaman tree")
    args = parser.parse_args()

    # --- 2. AUTENTIKASI & INISIALISASI DAGSHUB ---
    # Mengambil token DagsHub secara otomatis dari GitHub Secrets (Environment Variable)
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if dagshub_token:
        dagshub.auth.add_app_token(dagshub_token)

    # Inisialisasi tracking MLflow ke DagsHub
    repo_owner = 'juan10082002'
    repo_name = 'Membangun_model'
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    
    # Matikan autolog untuk memastikan kestabilan logging manual via CI
    mlflow.sklearn.autolog(disable=True)

    # --- 3. MEMBACA DATASET ---
    # Path relatif disesuaikan dengan struktur folder Workflow-CI Anda
    data_path = "MLProject/namadataset_preprocessing/cleaned_data.csv"
    if not os.path.exists(data_path):
        print(f"Error: File dataset tidak ditemukan di {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # Pisahkan Fitur dan Target (Pastikan kolom 'target' sesuai dengan nama kolom di dataset Anda)
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 4. EKSEKUSI TRAINING & MLFLOW LOGGING ---
    mlflow.set_experiment("CI_Retraining_Experiment")
    
    with mlflow.start_run(run_name="CI_Automated_Retrain"):
        # Inisialisasi model dengan parameter input dari GitHub Actions
        rf = RandomForestClassifier(
            n_estimators=args.n_estimators, 
            max_depth=args.max_depth, 
            random_state=42
        )
        rf.fit(X_train, y_train)
        
        # Evaluasi Model
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Mencatat Parameter & Metrik ke DagsHub
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("accuracy", acc)
        
        # Menyimpan Artefak Model Utama
        mlflow.sklearn.log_model(rf, "model")
        
        # --- LOG BUKTI UNTUK GITHUB ACTIONS (PENTING) ---
        # Mengambil Run ID yang aktif saat ini
        current_run_id = mlflow.active_run().info.run_id
        
        # Format print ini dibaca oleh perintah grep di ci_workflow.yml untuk build Docker
        print(f"TARGET_RUN_ID:{current_run_id}")
        print(f"Retraining Success! Model Accuracy: {acc:.4f}")

if __name__ == "__main__":
    run_retraining()