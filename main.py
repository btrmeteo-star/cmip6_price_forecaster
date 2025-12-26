#!/usr/bin/env python3
"""
最小可跑 CMIP6 氣象特徵產生器
- 90 天小檔保證不爆記憶體
- 每步印維度，方便抓無限迴圈
- 輸出可直接餵給 train.py
"""
import os
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np

# ---------- 路徑 ----------
RAW_DIR   = Path(__file__).resolve().parents[2] / 'data' / 'raw'
PROC_DIR  = Path(__file__).resolve().parents[2] / 'data' / 'processed'
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 參數（可調） ----------
TARGET_LAT =  25.0          # 台中附近
TARGET_LON = 121.0          # 台中附近
DIST_DEG   =   2.0          # ±2° 矩形裁剪
TIME_SLICE = '2015-01-01','2015-04-01'  # 只取前 3 個月（90 天）

# ---------- 工具函式 ----------
def load_cmip6(nc_path):
    """強制 chunks 避免 OOM，印維度方便 debug"""
    print(f'[load] {nc_path.name}')
    ds = xr.open_dataset(nc_path, chunks={'time': 30})
    print(f'[load] 原始維度 {ds.dims}')
    return ds

def clip_region(ds, lat0, lon0, dist_deg):
    """矩形裁剪 + 印結果筆數"""
    lat_min, lat_max = lat0-dist_deg, lat0+dist_deg
    lon_min, lon_max = lon0-dist_deg, lon0+dist_deg
    out = ds.sel(lat=slice(lat_min, lat_max),
                 lon=slice(lon_min, lon_max))
    print(f'[clip] 裁剪後 {out.dims}')
    return out

def daily_agg(ds, var):
    """日平均（若已是日均值就回傳原值）"""
    if 'time' in ds[var].coords:
        agg = ds[var].resample(time='1D').mean()
    else:
        agg = ds[var]
    print(f'[agg ] {var} 日筆數 {len(agg.time)}')
    return agg

def to_feature_table(daily_ds, vars=['tasmax', 'pr']):
    """把 3D (time,lat,lon) 拉平成 2D DataFrame"""
    df = daily_ds[vars].to_dataframe().reset_index()  # 展平
    print(f'[table] 總列數 {len(df)}')
    return df

# ---------- 主流程 ----------
def main():
    print('[main] 1. 載入 NetCDF')
    tasmax_ds = load_cmip6(RAW_DIR / 'cmip6_tasmax.nc')
    pr_ds     = load_cmip6(RAW_DIR / 'cmip6_pr.nc')

    print('[main] 2. 時間裁剪')
    tasmax_ds = tasmax_ds.sel(time=slice(*TIME_SLICE))
    pr_ds     = pr_ds.sel(time=slice(*TIME_SLICE))

    print('[main] 3. 空間裁剪')
    tasmax_ds = clip_region(tasmax_ds, TARGET_LAT, TARGET_LON, DIST_DEG)
    pr_ds     = clip_region(pr_ds,     TARGET_LAT, TARGET_LON, DIST_DEG)

    print('[main] 4. 日平均（若需要）')
    tasmax_day = daily_agg(tasmax_ds, 'tasmax')
    pr_day     = daily_agg(pr_ds,     'pr')

    print('[main] 5. 合併 & 拉平成 DataFrame')
    daily_ds   = xr.merge([tasmax_day, pr_day])
    df_feat    = to_feature_table(daily_ds)

    print('[main] 6. 寫 CSV')
    out_file = PROC_DIR / 'cmip6_features.csv'
    df_feat.to_csv(out_file, index=False)
    print(f'[main] 特徵已寫出 -> {out_file}  列數={len(df_feat)}')

# ---------- 可被 import 也直接執行 ----------
if __name__ == '__main__':
    main()
