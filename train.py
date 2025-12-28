#!/usr/bin/env python3
"""
训练 + 注册 三合一 train.py
通过环境变量 COMMODITY 切换商品（rice / corn / barley）
- 读 data/processed/{commodity}_features.csv
- 记忆体训练 LinearRegression
- 自动注册最佳模型到 MLflow Registry
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import mlflow
import mlflow.sklearn
from pathlib import Path

# ---------- 参数 ----------
COMMODITY   = os.getenv("COMMODITY", "rice")        # 切换商品
DATA_FILE   = Path(__file__).with_name('data') / 'processed' / f"{COMMODITY}_features.csv"
MODEL_NAME  = f"{COMMODITY}-price-model"            # Registry 仓库名
R2_THRESHOLD = 0.0                                  # 推进 Stage 门槛（可调）
FEATURE_COLS = ['tasmax', 'pr']                     # 气象特征
TARGET_VAR   = f"{COMMODITY}_price"                 # 预测目标列

# ---------- 主流程 ----------
def main():
    print(f'[train] 商品 = {COMMODITY}')
    print('[train] 载入内存资料 …')
    df = pd.read_csv(DATA_FILE, parse_dates=['time'])
    X = df[FEATURE_COLS].values
    y = df[TARGET_VAR].values

    with mlflow.start_run(run_name=f"lr_{COMMODITY}") as run:
        # ① log 参数
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("feature_cnt", X.shape[1])
        mlflow.log_param("data_rows", len(df))

        # ② 训练
        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)
        mlflow.log_metric("r2", r2)
        print(f'[train] 模型已落地 R²={r2:.3f}')

        # ③ artifact（写到当前 run，避开 /mlflow 权限问题）
        mlflow.sklearn.log_model(model, "model")

        # ④ 注册到 Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri, MODEL_NAME)

        # ⑤ 版本描述
        client = mlflow.MlflowClient()
        client.update_model_version(
            name=MODEL_NAME,
            version=mv.version,
            description=f"R²={r2:.3f}  特征={X.shape[1]}  数据={DATA_FILE}"
        )

        # ⑥ 推进 Stage（只保留最佳 1 版）
        if r2 > R2_THRESHOLD:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=mv.version,
                stage="Staging",               # 也可 "Production"
                archive_existing_versions=True
            )
            print(f'✅ 最佳模型已推进 Staging：{MODEL_NAME} v{mv.version}')
        else:
            print(f'⚠️ R²={r2} 未达门槛，仅注册不推进 Stage')

# ---------- 入口 ----------
if __name__ == '__main__':
    main()
