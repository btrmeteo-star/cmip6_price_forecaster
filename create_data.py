import pandas as pd
import xarray as xr

# 1. 加载玉米价格
price_df = pd.read_csv("data/raw/corn_prices.csv", parse_dates=["date"])
price_df.set_index("date", inplace=True)

# 2. 加载 CMIP6 数据（按区域平均）
tasmax_ds = xr.open_dataset("data/raw/tasmax_day.nc")
pr_ds = xr.open_dataset("data/raw/pr_day.nc")

# 提取玉米主产区（如中国 35°N–45°N, 110°E–130°E）
region = tasmax_ds.sel(lat=slice(35, 45), lon=slice(110, 130))
tasmax_monthly = region.tasmax.resample(time="1M").mean()
pr_monthly = pr_ds.pr.sel(lat=slice(35, 45), lon=slice(110, 130)).resample(time="1M").sum()

# 3. 合并气候与价格
climate_df = pd.DataFrame({
    "tasmax": tasmax_monthly.values.flatten(),
    "pr": pr_monthly.values.flatten()
}, index=tasmax_monthly.time.to_pandas())

# 对齐时间（确保月度一致）
merged = price_df.join(climate_df, how="inner")

# 4. 构造滞后特征（lag1, lag2...）
for col in ["price", "tasmax", "pr"]:
    for lag in [1, 2]:
        merged[f"{col}_lag{lag}"] = merged[col].shift(lag)
merged["tasmax_mean"] = merged[["tasmax", "tasmax_lag1", "tasmax_lag2"]].mean(axis=1)
merged["pr_std"] = merged[["pr", "pr_lag1", "pr_lag2"]].std(axis=1)

# 5. 保存为 corn_features.csv
features = merged.dropna()  # 去除滞后产生的 NaN
features.to_csv("data/processed/corn_features.csv", index=False)