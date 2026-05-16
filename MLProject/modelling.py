import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    # Integrasi Token DagsHub dari GitHub Secrets
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if dagshub_token:
        dagshub.auth.add_app_token(dagshub_token)

    dagshub.init(repo_owner='juan10082002', repo_name='Membangun_model', mlflow=True)
    
    with mlflow.start_run(run_name="CI_Automated_Retrain"):
        # Jalur relatif dari root repositori saat di-run via MLflow
        df = pd.read_csv("MLProject/namadataset_preprocessing/cleaned_data.csv")
        
        X = df.drop('target', axis=1) # Sesuaikan kolom target Anda
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
        rf.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, rf.predict(X_test))
        
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(rf, "model")
        
        print(f"Retraining Success! Accuracy: {acc}")