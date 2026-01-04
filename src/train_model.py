#!/usr/bin/env python3
"""
使用 MLflow 训练 CMIP6 农产品价格预测模型（含滞后特征）
- 输入: data/final/{crop}_dataset.csv（由 data_merge.py 生成）
- 特征: tas_avg, pr_sum + 滞后项 (lag1, lag7)
- 模型: LinearRegression, RandomForest, XGBoost
- 输出: MLflow 实验记录（http://localhost:8000）
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# === 配置 ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJECT_ROOT / "data" / "final"
CROPS = ["rice", "corn", "barley"]

# MLflow 设置
mlflow.set_tracking_uri("http://0.0.0.0:8000")
EXPERIMENT_NAME = "cmip6-crop-price-prediction"

# 特征列（必须与 data_merge.py 生成的一致）
FEATURE_COLS = [
    "tas_avg", "pr_sum",
    "tas_lag1", "pr_lag1", "price_lag1",
    "tas_lag7", "pr_lag7", "price_lag7"
]

def train_and_log(crop: str, model_name: str, model, X_train, X_test, y_train, y_test):
    """训练单个模型并记录到 MLflow"""
    with mlflow.start_run(run_name=f"{crop}-{model_name}"):
        # 标签
        mlflow.set_tag("crop", crop)
        mlflow.set_tag("model_type", model_name)

        # 训练
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)

        # 评估指标
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # 记录指标
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        # 记录超参数
        if model_name == "random_forest":
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("max_depth", model.max_depth)
        elif model_name == "xgboost":
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("learning_rate", model.learning_rate)

        # 保存模型
        mlflow.sklearn.log_model(model, "model")

        print(f"✅ {crop} - {model_name}: R²={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")

def main():
    # 创建或获取 MLflow 实验
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    for crop in CROPS:
        print(f"\n--- 训练 {crop} 模型 ---")
        dataset_path = FINAL_DIR / f"{crop}_dataset.csv"

        if not dataset_path.exists():
            print(f"⚠️  数据集不存在: {dataset_path}，跳过...")
            continue

        # 加载数据
        df = pd.read_csv(dataset_path, parse_dates=["date"])
        print(f"📊 加载 {len(df)} 行数据")

        # 提取特征和标签
        X = df[FEATURE_COLS].fillna(0)  # 处理可能的 NaN
        y = df["price"]

        # 时间序列分割（前80%训练，后20%测试）
        split_idx = int(0.8 * len(df))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # 定义模型
        models = {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            "xgboost": XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42,
                verbosity=0  # 静默模式
            ),
        }

        # 训练所有模型
        for name, model in models.items():
            train_and_log(crop, name, model, X_train, X_test, y_train, y_test)

    print("\n🎉 所有训练完成！查看 MLflow: http://localhost:8000")

if __name__ == "__main__":
    main()
