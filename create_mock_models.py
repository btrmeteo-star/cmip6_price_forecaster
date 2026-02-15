#!/usr/bin/env python3
"""
生成兼容的 mock 模型文件（用于测试前端 + API 结构）
实际部署时请替换为真实训练的模型。
"""

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from pathlib import Path

# 创建 models 目录
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# 定义作物及其特征（必须与 index.html 中的字段一致！）
crops = {
    "corn": [
        "pr", "pr_lag1", "pr_lag2", "pr_std",
        "price_lag1", "price_lag2",
        "tasmax", "tasmax_lag1", "tasmax_lag2", "tasmax_mean"
    ],
    "wheat": [
        "pr", "pr_lag1", "pr_lag2", "pr_std",
        "price_lag1", "price_lag2",
        "tasmax", "tasmax_lag1", "tasmax_lag2", "tasmax_mean"
    ],
    "soybean": [
        "pr", "pr_lag1", "pr_lag2", "pr_std",
        "price_lag1", "price_lag2",
        "tasmax", "tasmax_lag1", "tasmax_lag2", "tasmax_mean"
    ]
}

# 为每种作物生成一个简单模型
for crop, features in crops.items():
    # 创建虚拟数据
    X = pd.DataFrame([{f: 1.0 for f in features}])
    y = [100.0]  # 虚拟目标值
    
    # 使用 DummyRegressor（总是预测固定值 186.88）
    model = DummyRegressor(strategy="constant", constant=186.88)
    model.fit(X, y)
    
    # 保存为字典格式，兼容 main.py
    model_package = {
        "model": model,
        "feature_names": features
    }
    
    joblib.dump(model_package, MODEL_DIR / f"{crop}.joblib")
    print(f"✅ 已生成 {crop}.joblib")

print("\n💡 提示：这些是模拟模型。请用你的真实训练代码替换它们！")