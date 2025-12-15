import xarray as xr
import pandas as pd
import click
from pathlib import Path

@click.command()
@click.option('--nc', default='data/cmip6.nc', help='raw CMIP6 netCDF')
@click.option('--spot', default='data/spot.csv', help='spot price csv')
@click.option('--out', default='data/features.parquet')
def main(nc, spot, out):
    ds = xr.open_dataset(nc)
    df_cmip = ds.to_dataframe().reset_index()
    df_spot = pd.read_csv(spot, parse_dates=['date'])

    # 36 個氣象衍生變數 + 3 月滯後
    df = feature_engineering(df_cmip, df_spot)
    Path(out).parent.mkdir(exist_ok=True, parents=True)
    df.to_parquet(out)
    print(f'saved {out}  rows={len(df)}')

def feature_engineering(df_cmip, df_spot):
    # TODO: 真實拼接邏輯
    df = df_spot.merge(df_cmip, left_on='date', right_on='time', how='inner')
    for lag in [1, 3]:
        df[f'price_lag_{lag}m'] = df['price'].shift(lag)
    return df.dropna()

if __name__ == '__main__':
    main()
