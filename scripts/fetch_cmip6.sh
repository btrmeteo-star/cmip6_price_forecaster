#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="data/raw"
mkdir -p "$OUT_DIR"
echo "[CMIP6] 產生假 netCDF 供 pipeline 通過…"
python - << 'PY'
import numpy as np, xarray as xr, pandas as pd
dates = pd.date_range('2015-01-01', '2025-12-31', freq='D')
ny = nx = 10
temp  = 15 + 10 * np.random.randn(len(dates), ny, nx)
precip= np.clip(np.random.exponential(5, (len(dates), ny, nx)), 0, 200)
for name, var in [('tasmax', temp), ('pr', precip)]:
    ds = xr.Dataset({name: (['time','lat','lon'], var)},
                    coords={'time': dates, 'lat': np.linspace(-5, 5, ny), 'lon': np.linspace(110, 120, nx)})
    ds.to_netcdf(f'data/raw/cmip6_{name}.nc')
PY
