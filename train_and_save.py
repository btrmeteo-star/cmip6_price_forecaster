#!/usr/bin/env python3
"""
使用真实 CMIP6 气候数据 + 历史价格训练并保存模型。
数据文件要求：
  - data/corn.csv
  - data/wheat.csv
  - data/soybean.csv

每份 CSV 必须包含以下列：
  pr, pr_lag1, pr_lag2, pr_std,
  price_lag1, price_lag2,
  tasmax, tasmax_lag1, tasmax_lag2, tasmax_mean,
  price  ← 目标变量（当前期价格）
"""

import os
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRegressor
# 或改用：from sklearn.ensemble import RandomForestRegressor

# 配置
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# 特征列（必须与 index.html 中的字段名完全一致！）
FEATURES = [
    "pr", "pr_lag1", "pr_lag2", "pr_std",
    "price_lag1", "price_lag2",
    "tasmax", "tasmax_lag1", "tasmax_lag2", "tasmax_mean"
]
TARGET = "price"

# 支持的作物
CROPS = ["corn", "wheat", "soybean"]


def train_model_for_crop(crop: str):
    """为指定作物训练模型"""
    data_path = DATA_DIR / f"{crop}.csv"
    
    if not data_path.exists():
        print(f"⚠️ 跳过 {crop}：未找到数据文件 {data_path}")
        return

    # 加载数据
    df = pd.read_csv(data_path)
    
    # 检查必要列
    missing_features = set(FEATURES + [TARGET]) - set(df.columns)
    if missing_features:
        raise ValueError(f"{crop}.csv 缺少列: {missing_features}")

    X = df[FEATURES]
    y = df[TARGET]

    print(f"📊 {crop}: 训练样本数 = {len(df)}")

    # 初始化模型（可替换为其他算法）
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    # 训练
    model.fit(X, y)
    
    # 保存为兼容格式
    model_package = {
        "model": model,
        "feature_names": FEATURES
    }
    
    joblib.dump(model_package, MODEL_DIR / f"{crop}.joblib")
    print(f"✅ 已保存 {crop}.joblib")


def main():
    """主函数：批量训练所有作物"""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"请创建 '{DATA_DIR}' 目录并放入 CSV 数据文件")

    for crop in CROPS:
        try:
            train_model_for_crop(crop)
        except Exception as e:
            print(f"❌ 训练 {crop} 失败: {e}")

    print("\n🎉 所有模型训练完成！")


if __name__ == "__main__":
    main()