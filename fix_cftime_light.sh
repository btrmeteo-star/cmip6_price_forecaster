#!/usr/bin/env bash
set -e
echo "=== 轻量装 cftime（pip 预编译）==="
pip install --user cftime
echo "=== 验证 ==="
python -c "import cftime, xarray as xr; ds=xr.open_dataset('data/raw/cmip6_tasmax.nc'); print('OK', ds.dims)"
echo "=== 跑处理器 ==="
python -m cmip6_commodity.climate_processor
