import argparse
import pandas as pd
import mlflow
import dagshub
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    dagshub.init(repo_owner='juan10082002', repo_name='Membangun_model', mlflow=True)
    
    with mlflow.start_run():
        # ... load data & train ...
        rf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)
        # ... log metrics & artifacts ...
        mlflow.sklearn.log_model(rf, "model")