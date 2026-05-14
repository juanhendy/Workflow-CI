import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import os  # Tambahkan ini untuk membaca environment variables
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    # --- BAGIAN KOREKSI UTAMA: AUTO-AUTHENTICATION ---
    # Memastikan GitHub Actions tidak meminta login manual (OAuth)
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if dagshub_token:
        dagshub.auth.add_app_token(dagshub_token)

    # Inisialisasi DagsHub
    dagshub.init(repo_owner='juan10082002', repo_name='Membangun_model', mlflow=True)
    
    with mlflow.start_run():
        # Membaca dataset
        # Pastikan path ini sesuai dengan struktur folder di repo GitHub Anda
        data_path = "MLProject/namadataset_preprocessing/cleaned_data.csv"
        df = pd.read_csv(data_path)
        
        # Pisahkan fitur dan target
        # Pastikan kolom 'target' ada di dataset Anda, jika namanya berbeda silakan ganti
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Inisialisasi dan Training
        rf = RandomForestClassifier(
            n_estimators=args.n_estimators, 
            max_depth=args.max_depth,
            random_state=42 # Tambahkan random_state agar hasil konsisten
        )
        rf.fit(X_train, y_train)
        
        # Evaluasi
        acc = accuracy_score(y_test, rf.predict(X_test))
        
        # Logging ke MLflow/DagsHub
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("accuracy", acc)
        
        # Log model (Penting untuk Kriteria 4 nantinya)
        mlflow.sklearn.log_model(rf, "model")
        
        print(f"Success! Accuracy: {acc}")