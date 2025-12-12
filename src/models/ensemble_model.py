"""
Ensemble price forecaster
Prophet + Climate regressor || LSTM || XGBoost
Author: CMIP6-Price-Forecaster
"""
import os
import joblib
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import logging

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------- 辅助 ----------
def xr_to_ts(ds: xr.Dataset, var_list, commodity, region_bbox) -> pd.DataFrame:
    """把 CMIP6 区域平均后转成月度序列"""
    if region_bbox:   # [lon_min, lon_max, lat_min, lat_max]
        ds = ds.sel(lon=slice(region_bbox[0], region_bbox[1]),
                    lat=slice(region_bbox[2], region_bbox[3]))
    df = ds[var_list].to_dataframe().reset_index()
    df = df.groupby([df.time.dt.year, df.time.dt.month]).mean().reset_index()
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    df = df[["date"] + var_list].rename(columns={v: f"cmip6_{v}" for v in var_list})
    return df.sort_values("date").reset_index(drop=True)


# ---------- LSTM 模块 ----------
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TSDataset(Dataset):
    def __init__(self, df, target_col, climate_cols, look_back=12):
        self.look_back = look_back
        self.y = df[target_col].values
        self.X = df[climate_cols + [target_col]].values
        self.scaler = StandardScaler().fit(self.X)
        self.X = self.scaler.transform(self.X)

    def __len__(self):
        return len(self.X) - self.look_back

    def __getitem__(self, idx):
        return (self.X[idx:idx + self.look_back],
                self.y[idx + self.look_back])


# ---------- 单模型封装 ----------
class ProphetClimate:
    """Prophet + 气候回归项"""
    def __init__(self, climate_vars):
        self.climate_vars = climate_vars
        self.model = None
        self.reg_scaler = StandardScaler()

    def fit(self, df: pd.DataFrame, target_col: str):
        # 构造 Prophet 数据框
        tmp = df[['date', target_col] + self.climate_vars].dropna()
        tmp = tmp.rename(columns={'date': 'ds', target_col: 'y'})
        climate = self.reg_scaler.fit_transform(tmp[self.climate_vars])
        for i, v in enumerate(self.climate_vars):
            tmp[f'cli_{i}'] = climate[:, i]

        self.model = Prophet(yearly_seasonality=True, weekly_seasonality=False)
        for i in range(len(self.climate_vars)):
            self.model.add_regressor(f'cli_{i}')
        self.model.fit(tmp)
        logger.info("ProphetClimate fit done")

    def predict(self, df: pd.DataFrame, horizon: int):
        tmp = df[['date'] + self.climate_vars].copy()
        future_dates = pd.date_range(tmp.date.max() + timedelta(days=1),
                                     periods=horizon, freq='MS')
        future = pd.DataFrame({'ds': future_dates})
        # 用最后 12 个月均值简单外填（生产可用更复杂时序模型）
        cli_last = tmp[self.climate_vars].tail(12).mean()
        for i, v in enumerate(self.climate_vars):
            future[f'cli_{i}'] = cli_last[v]
        fc = self.model.predict(future)
        return fc.yhat.values


class LSTMClimate:
    def __init__(self, climate_vars, look_back=12, device='cpu'):
        self.look_back = look_back
        self.climate_vars = climate_vars
        self.device = device
        self.model = None
        self.scaler = None

    def fit(self, df: pd.DataFrame, target_col: str, epochs=100, lr=1e-3):
        ds = TSDataset(df, target_col, self.climate_vars, self.look_back)
        loader = DataLoader(ds, batch_size=16, shuffle=True)
        input_dim = ds.X.shape[1]
        self.model = LSTMNet(input_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.float().to(self.device), yb.float().to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out.squeeze(), yb)
                loss.backward()
                optimizer.step()
        self.scaler = ds.scaler
        logger.info("LSTMClimate fit done")

    def predict(self, df: pd.DataFrame, horizon: int):
        self.model.eval()
        # 用最后窗口滚动预测
        tmp = df[self.climate_vars + ['price']].values
        tmp = self.scaler.transform(tmp)
        preds = []
        with torch.no_grad():
            for _ in range(horizon):
                x = torch.tensor(tmp[-self.look_back:]).unsqueeze(0).float()
                p = self.model(x).item()
                preds.append(p)
                # 把预测值拼回去继续滚
                new_row = np.append(tmp[-1, :-1], p)
                tmp = np.vstack([tmp, new_row])
        return np.array(preds)


# ---------- Ensemble ----------
class EnsembleModel:
    def __init__(self, climate_vars, device='cpu'):
        self.climate_vars = climate_vars
        self.device = device
        self.models = {
            'prophet': ProphetClimate(climate_vars),
            'lstm': LSTMClimate(climate_vars, device=device),
            'xgb': GradientBoostingRegressor(max_depth=4, n_estimators=300)
        }
        self.weights = None   # 等权 or 最优加权

    # ----- 训练 -----
    def fit(self, df: pd.DataFrame, target_col='price', horizon_month=6):
        # 1. Prophet
        self.models['prophet'].fit(df, target_col)
        # 2. LSTM
        self.models['lstm'].fit(df, target_col, epochs=150)
        # 3. XGBoost（基于滞后特征）
        self._fit_xgb(df, target_col, horizon_month)
        # 4. 计算最优加权（可关闭）
        self._tune_weights(df, target_col, horizon_month)
        logger.info("Ensemble fit all done")

    def _fit_xgb(self, df, target, horizon):
        tmp = df.copy()
        # 构造滞后+气候特征
        for lag in [1, 2, 3, 6, 12]:
            tmp[f'lag_{lag}'] = tmp[target].shift(lag)
        tmp = tmp.dropna()
        X = tmp[self.climate_vars + [c for c in tmp if c.startswith('lag_')]]
        y = tmp[target]
        self.models['xgb'].fit(X, y)
        self.xgb_cols = X.columns.tolist()

    def _tune_weights(self, df, target, horizon):
        # 用 12 个月滚动窗口生成验证集
        val_idx = -horizon
        val_df = df.iloc[val_idx - 12: val_idx]
        preds = {}
        for name in ['prophet', 'lstm', 'xgb']:
            preds[name] = self._single_pred(name, val_df, horizon=horizon)
        true = df[target].iloc[val_idx: val_idx + horizon].values
        # 最小化 MAE 的加权
        from scipy.optimize import minimize
        def error(w): return mean_absolute_error(true, sum(w[i]*preds[n] for i, n in enumerate(preds)))
        x0 = np.ones(len(preds)) / len(preds)
        cons = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        bounds = [(0, 1)] * len(preds)
        res = minimize(error, x0, bounds=bounds, constraints=cons)
        self.weights = res.x
        logger.info(f"Optimal weights: {dict(zip(preds, self.weights))}")

    def _single_pred(self, name, df, horizon):
        if name == 'prophet':
            return self.models['prophet'].predict(df, horizon)
        elif name == 'lstm':
            return self.models['lstm'].predict(df, horizon)
        else:  # xgb
            tmp = df.copy()
            for lag in [1, 2, 3, 6, 12]:
                tmp[f'lag_{lag}'] = tmp['price'].shift(lag)
            tmp = tmp.tail(horizon)   # 需要 horizon 行
            X = tmp[self.xgb_cols]
            return self.models['xgb'].predict(X)

    # ----- 预测 -----
    def predict(self, ds: xr.Dataset, commodity: str, horizon: int):
        # 1. 把 CMIP6 转为 DataFrame
        region_bbox = COMMODITY_REGION.get(commodity, None)
        var_list = self.climate_vars
        df_climate = xr_to_ts(ds, var_list, commodity, region_bbox)

        # 2. 各模型预测
        preds = {}
        for name in ['prophet', 'lstm', 'xgb']:
            preds[name] = self._single_pred(name, df_climate, horizon)

        # 3. 加权集成
        if self.weights is not None:
            w = self.weights
        else:
            w = np.ones(len(preds)) / len(preds)
        ensemble = sum(w[i] * preds[n] for i, n in enumerate(preds))
        ci_lower = np.percentile(list(preds.values()), 25, axis=0)
        ci_upper = np.percentile(list(preds.values()), 75, axis=0)
        return float(ensemble[-1]), float(ci_lower[-1]), float(ci_upper[-1])

    # ----- 序列化 -----
    def save(self, path: str):
        joblib.dump(self, path)
        logger.info(f"Ensemble saved to {path}")

    @staticmethod
    def load(path: str):
        obj = joblib.load(path)
        logger.info(f"Ensemble loaded from {path}")
        return obj


# ---------- 预设产区 ----------
COMMODITY_REGION = {
    "maize": [-100, -80, 35, 50],      # US Midwest
    "wheat": [30, 60, 45, 55],         # Russia + Ukraine
    "soybean": [-100, -50, -40, 10],   # Brazil + Argentina
    "copper": [-75, -65, -30, -15],    # Chile
    "crude": [30, 60, 25, 35],         # Middle-East
}
