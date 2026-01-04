#!/usr/bin/env python3
"""
訓練 + MLflow 註冊（最小可跑版）
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
import mlflow
import mlflow.sklearn

DATA_DIR = Path(__file__).parents[3] / "data"
PROC_DIR = DATA_DIR / "processed"

def main():
    print('[train] 載入資料 …')
    X = pd.read_csv(PROC_DIR / "cmip6_features.csv")
    y = pd.read_csv(PROC_DIR / "spot_price.csv").squeeze()

    print('[train] 訓練模型 …')
    model = LinearRegression()
    model.fit(X, y)
    score = model.score(X, y)

    # MLflow 註冊
    mlflow.set_tracking_uri("https://verbose-space-waffle-7v6wvqw4wxp5hrvw6-8000.app.github.dev/mlflow")
    mlflow.set_experiment("soybean-price")
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("r2", score)
        print(f'[train] 模型已落地 R²={score:.3f}')

if __name__ == '__main__':
    main()
