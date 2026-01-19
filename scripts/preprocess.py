#!/usr/bin/env python3
"""
预处理脚本（适配 CSV mock 数据 + 强制标准特征顺序）
- 读取 data/raw/spot_price.csv（含 commodity 列）
- 读取 data/raw/cmip6_{crop}.csv
- 合并、对齐时间、生成滞后和滚动特征
- 按固定顺序保存特征列，确保 train.py 和 app.py 一致
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CROPS = ["rice", "wheat"]
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 🔑 标准特征顺序（必须与 train.py / app.py 一致）
FEATURE_ORDER = [
    "pr", "pr_lag1", "pr_lag2", "pr_std",
    "price_lag1", "price_lag2",
    "tasmax", "tasmax_lag1", "tasmax_lag2", "tasmax_mean"
]


def load_spot_prices(crop: str) -> pd.DataFrame:
    """从统一价格文件中加载指定作物的价格"""
    df = pd.read_csv(RAW_DATA_DIR / "spot_price.csv", parse_dates=["time"])
    df_crop = df[df["commodity"] == crop].copy()
    print(f"✅ Loaded {len(df_crop)} price records for {crop}")
    return df_crop[["time", "price"]]


def load_climate_data(crop: str) -> pd.DataFrame:
    """加载 CSV 格式的气候数据"""
    file_path = RAW_DATA_DIR / f"cmip6_{crop}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Climate file not found: {file_path}")
    
    df = pd.read_csv(file_path, parse_dates=["time"])
    print(f"✅ Loaded climate data for {crop} ({len(df)} days)")
    return df


def add_lag_features(df: pd.DataFrame, cols: list, lags: list = [1, 2]) -> pd.DataFrame:
    """为指定列添加滞后特征"""
    df = df.copy()
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """添加滚动统计特征"""
    df = df.copy()
    # pr 的 7 天滚动标准差
    df["pr_std"] = df["pr"].rolling(window=7, min_periods=1).std().fillna(0)
    # tasmax 的 7 天滚动均值
    df["tasmax_mean"] = df["tasmax"].rolling(window=7, min_periods=1).mean()
    return df


def main():
    for crop in CROPS:
        print(f"\n[preprocess] 商品 = {crop}")
        
        # 1. 加载价格和气候数据
        prices = load_spot_prices(crop)
        climate = load_climate_data(crop)
        
        # 2. 合并数据（按 time 对齐）
        df = pd.merge(climate, prices, on="time", how="inner")
        print(f"✅ 合并后数据量: {len(df)}")
        
        # 3. 添加滞后特征（价格 + 气候）
        df = add_lag_features(df, cols=["price", "tasmax", "pr"], lags=[1, 2])
        
        # 4. 添加滚动特征
        df = add_rolling_features(df, cols=["pr", "tasmax"])
        
        # 5. 删除包含 NaN 的行（因滞后产生）
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        print(f"✅ 去除 NaN 后: {len(df)} (丢弃 {initial_len - len(df)})")
        
        # 6. ✅ 关键：按标准顺序重排列，并确保包含所有必要字段
        expected_columns = ["time"] + FEATURE_ORDER + ["price"]
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"缺失列: {missing_cols}")
        
        df_ordered = df[expected_columns]
        
        # 7. 保存
        output_file = PROCESSED_DIR / f"{crop}_features.csv"
        df_ordered.to_csv(output_file, index=False)
        print(f"✅ 保存至 {output_file} (标准特征顺序)")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
