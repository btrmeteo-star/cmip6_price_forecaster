#!/usr/bin/env python3
"""
CMIP6 氣象特徵計算器
- 讀取每日 tasmax / pr  NetCDF
- 計算 5 項核心指標
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

class ClimateProcessor:
    """最小氣象指標計算類"""

    def load_cmip6_data(self, nc_path: Path) -> xr.Dataset:
        """讀取 NetCDF 並當場載入記憶體，避免 dask rolling 錯誤"""
        ds = xr.open_dataset(nc_path)
        ds = ds.load()               # 關鍵：脫離 dask
        return ds

    def calculate_spi(self, precip_ds: xr.Dataset, scale: int = 3) -> xr.DataArray:
    # 關鍵：當場脫離 dask
    precip_ds = precip_ds.load()
    precip_roll = precip_ds['pr'].rolling(time=scale, center=True).sum()
    # 簡易標準化
    spi = (roll - roll.mean()) / roll.std()
    return spi.rename(f'spi_{scale}month')

    def calculate_temp_anomaly(self, tasmax_ds: xr.Dataset) -> xr.DataArray:
        """氣溫異常（相對於氣候平均）"""
        tasmax_ds = tasmax_ds.load()
        clim = tasmax_ds['tasmax'].groupby('time.dayofyear').mean()
        anomaly = tasmax_ds['tasmax'].groupby('time.dayofyear') - clim
        return anomaly.rename('tasmax_anomaly')

    def calculate_gdd(self, tasmax_ds: xr.Dataset, base: float = 10.0) -> xr.DataArray:
        """生長度日 GDD = max(0, Tmax - base) 每日累積"""
        tasmax_ds = tasmax_ds.load()
        gdd = (tasmax_ds['tasmax'] - base).clip(min=0)
        return gdd.rename('gdd')

    def calculate_heat_wave_days(self, tasmax_ds: xr.Dataset, threshold: float = 35.0) -> xr.DataArray:
        """熱浪天數 proxy：Tmax > 35 °C 的日計"""
        tasmax_ds = tasmax_ds.load()
        hot = (tasmax_ds['tasmax'] > threshold).astype(int)
        return hot.rename('heat_wave_days')

    def calculate_cum_precip(self, precip_ds: xr.Dataset) -> xr.DataArray:
        """累積降水量（無滾動，僅單日值）"""
        precip_ds = precip_ds.load()
        return precip_ds['pr'].rename('cum_pr')


# -------------------- 主流程 --------------------
if __name__ == '__main__':
    processor = ClimateProcessor()

    print('[CP] 載入 CMIP6 資料 …')
    tasmax_ds = processor.load_cmip6_data(RAW_DIR / 'cmip6_tasmax.nc')
    precip_ds = processor.load_cmip6_data(RAW_DIR / 'cmip6_pr.nc')

    print('[CP] 計算指標 …')
    indicators = {}
    indicators['spi_3month']      = processor.calculate_spi(precip_ds, scale=3)
    indicators['tasmax_anomaly']  = processor.calculate_temp_anomaly(tasmax_ds)
    indicators['gdd']             = processor.calculate_gdd(tasmax_ds)
    indicators['heat_wave_days']  = processor.calculate_heat_wave_days(tasmax_ds)
    indicators['cum_pr']          = processor.calculate_cum_precip(precip_ds)

    # 合併成 DataFrame
    df = pd.concat([v.to_pandas().reset_index(drop=True) for v in indicators.values()], axis=1)
    df.columns = list(indicators.keys())
    df['date'] = precip_ds.time.to_pandas().index      # 保留時間索引

    out_path = PROC_DIR / 'cmip6_features.csv'
    df.to_csv(out_path, index=False)
    print(f'[CP] 氣象特徵已寫入 {out_path}')
