#!/usr/bin/env python3
"""
合并 CMIP6 气象特征与价格，并生成滞后特征（Lag Features）
- 输入: features.csv + price.csv
- 输出: dataset.csv with lag features
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_DIR = PROJECT_ROOT / "data" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

CROPS = ["rice", "corn", "barley"]

def add_lag_features(df: pd.DataFrame, lags=[1, 7]) -> pd.DataFrame:
    """为 tas, pr, price 添加滞后特征"""
    df = df.copy()
    for lag in lags:
        df[f"tas_lag{lag}"] = df["tas_avg"].shift(lag)
        df[f"pr_lag{lag}"] = df["pr_sum"].shift(lag)
        df[f"price_lag{lag}"] = df["price"].shift(lag)
    return df.dropna().reset_index(drop=True)

def main():
    for crop in CROPS:
        print(f"\n--- 处理 {crop} ---")
        
        feature_path = PROC_DIR / f"{crop}_features.csv"
        price_path = RAW_DIR / f"{crop}_price.csv"
        output_path = FINAL_DIR / f"{crop}_dataset.csv"

        # 加载气象特征
        if not feature_path.exists():
            raise FileNotFoundError(f"❌ 特征文件缺失: {feature_path}")
        df_feat = pd.read_csv(feature_path, parse_dates=["date"])

        # 加载或生成价格
        if not price_path.exists():
            print(f"⚠️  {price_path.name} 不存在，生成模拟价格...")
            dates = pd.date_range("2015-01-01", "2022-12-31", freq="D")
            np.random.seed(42)
            prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
            df_price = pd.DataFrame({"date": dates, "price": prices})
            df_price.to_csv(price_path, index=False)
        else:
            df_price = pd.read_csv(price_path, parse_dates=["date"])

        # 合并
        df = pd.merge(df_feat, df_price, on="date", how="inner")

        # 添加滞后特征
        df = add_lag_features(df, lags=[1, 7])
        print(f"✅ 添加滞后特征后: {len(df)} 行")

        # 保存
        df.to_csv(output_path, index=False)
        print(f"✅ 保存至 {output_path}")

    print("\n🎉 滞后特征已生成！")

if __name__ == "__main__":
    main()