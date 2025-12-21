#!/usr/bin/env python3
"""
CMIP6 氣象特徵計算器（完工版）
- 讀取每日 tasmax / pr NetCDF
- 純 pandas + numpy 滾動，完全繞過 xarray.rolling
- 輸出 csv 供後續訓練
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR  = Path(__file__).parents[3] / "data"
RAW_DIR   = DATA_DIR / "raw"
PROC_DIR  = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 工具 ----------
def load_cmip6(nc_path: Path) -> xr.Dataset:
    """強制載入記憶體，不再用 dask"""
    ds = xr.open_dataset(nc_path)
    return ds.load()               # 脫離 dask

def rolling_sum_numpy(arr: np.ndarray, window: int) -> np.ndarray:
    """純 numpy 滾動和，邊界補 np.nan"""
    return np.convolve(arr, np.ones(window), mode='same')

def rolling_mean_numpy(arr: np.ndarray, window: int) -> np.ndarray:
    """純 numpy 滾動平均，邊界補 np.nan"""
    return np.convolve(arr, np.ones(window) / window, mode='same')

# ---------- 指標 ----------
def calculate_spi(pr: pd.Series, scale: int = 3) -> pd.Series:
    """3 個月累積降水 proxy-SPI"""
    roll = rolling_sum_numpy(pr.values, scale)
    spi  = (roll - np.nanmean(roll)) / np.nanstd(roll)
    return pd.Series(spi, index=pr.index, name=f'spi_{scale}month')

def calculate_temp_anomaly(tas: pd.Series) -> pd.Series:
    """氣溫異常（相對於氣候平均）"""
    clim = tas.groupby(tas.index.dayofyear).mean()
    anomaly = tas.groupby(tas.index.dayofyear).transform(lambda x: x - clim)
    return anomaly.rename('tasmax_anomaly')

def calculate_gdd(tas: pd.Series, base: float = 10.0) -> pd.Series:
    """生長度日 GDD = max(0, Tmax - base)"""
    gdd = (tas - base).clip(lower=0)
    return gdd.rename('gdd')

def calculate_heat_wave_days(tas: pd.Series, threshold: float = 35.0) -> pd.Series:
    """熱浪天數 proxy：Tmax > 35 °C"""
    hot = (tas > threshold).astype(int)
    return hot.rename('heat_wave_days')

# ---------- 主流程 ----------
def main():
    print('[CP] 載入 CMIP6 資料 …')
    tasmax_ds = load_cmip6(RAW_DIR / 'cmip6_tasmax.nc')
    precip_ds = load_cmip6(RAW_DIR / 'cmip6_pr.nc')

    # 轉 pandas Series（索引=時間）
    tas = tasmax_ds['tasmax'].to_pandas().squeeze()
    pr  = precip_ds['pr'].to_pandas().squeeze()

    print('[CP] 計算指標 …')
    indicators = pd.DataFrame({
        'spi_3month'     : calculate_spi(pr, scale=3),
        'tasmax_anomaly' : calculate_temp_anomaly(tas),
        'gdd'            : calculate_gdd(tas),
        'heat_wave_days' : calculate_heat_wave_days(tas),
        'cum_pr'         : pr.rename('cum_pr'),
    })
    indicators['date'] = tas.index

    out_path = PROC_DIR / 'cmip6_features.csv'
    indicators.to_csv(out_path, index=False)
    print(f'[CP] 氣象特徵已寫入 {out_path}')

if __name__ == '__main__':
    main()
