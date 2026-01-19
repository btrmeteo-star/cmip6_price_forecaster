#!/usr/bin/env python3
"""
生成模拟的原始数据（多商品统一格式）：
- data/raw/spot_price.csv: 包含 commodity 列
- data/raw/cmip6_rice.csv
- data/raw/cmip6_wheat.csv
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 创建目录
os.makedirs("data/raw", exist_ok=True)

# 时间范围
start_date = datetime(2015, 1, 1)
end_date = datetime(2025, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='D')
n = len(dates)

# 为每个商品生成价格
all_prices = []
np.random.seed(42)
for crop in ["rice", "wheat"]:
    # 基础价格不同
    base = 220 if crop == "rice" else 200
    prices = base + 50 * np.sin(np.arange(n) * 2 * np.pi / 365) + np.random.normal(0, 10, n)
    prices = np.clip(prices, 150, 300)
    
    crop_df = pd.DataFrame({
        "time": dates,
        "price": prices,
        "commodity": crop
    })
    all_prices.append(crop_df)

# 合并为一个文件
spot_df = pd.concat(all_prices, ignore_index=True)
spot_df.to_csv("data/raw/spot_price.csv", index=False)
print("✅ 已生成: data/raw/spot_price.csv (含 commodity 列)")

# 生成 CMIP6 气候数据（每个作物一个文件）
np.random.seed(99)
tasmax = 25 + 10 * np.sin(np.arange(n) * 2 * np.pi / 365) + np.random.normal(0, 3, n)
pr = np.maximum(0, 5 + 3 * np.sin(np.arange(n) * 2 * np.pi / 365 + np.pi/2) + np.random.normal(0, 1, n))

for crop in ["rice", "wheat"]:
    cmip_df = pd.DataFrame({
        "time": dates,
        "tasmax": tasmax,
        "pr": pr
    })
    cmip_df.to_csv(f"data/raw/cmip6_{crop}.csv", index=False)
    print(f"✅ 已生成: data/raw/cmip6_{crop}.csv")

print("\n🎉 模拟数据生成完成！")
