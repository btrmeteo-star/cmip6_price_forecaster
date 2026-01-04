#!/usr/bin/env python3
"""
rice 气象特征生成器
- 读取 CMIP6 NetCDF（模拟模式）
- 裁剪 & 日平均（此处用模拟数据代替）
- 输出 rice_features.csv
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- 路径 ----------
RAW_DIR  = Path(__file__).resolve().parents[2] / 'data' / 'raw'
PROC_DIR = Path(__file__).resolve().parents[2] / 'data' / 'processed'
PROC_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT   = PROC_DIR / "rice_features.csv"

# ---------- 参数 ----------
TARGET_LAT = 25.0
TARGET_LON = 121.0
DIST_DEG   = 2.0
TIME_SLICE = ('2015-01-01', '2022-12-31')

def main():
    print(f"[rice] 开始生成气象特征 -> {OUTPUT}")
    
    # ========== 模拟数据（后续替换为真实 CMIP6 读取） ==========
    dates = pd.date_range(start=TIME_SLICE[0], end=TIME_SLICE[1], freq='D')
    np.random.seed(42)  # 可复现
    df = pd.DataFrame({
        'date': dates,
        'tas_avg': np.random.uniform(18, 32, len(dates)),  # 日均温
        'pr_sum': np.random.exponential(4, len(dates))     # 日降水
    })
    df.to_csv(OUTPUT, index=False)
    print(f"[rice] 成功保存 {len(df)} 天数据到 {OUTPUT}")

if __name__ == '__main__':
    main()
