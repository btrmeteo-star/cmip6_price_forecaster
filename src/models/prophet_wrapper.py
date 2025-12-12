"""
Thin wrapper for Prophet + optional climate regressors
save/load compatible with joblib
"""
import joblib
import pandas as pd
from prophet import Prophet
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProphetWrapper:
    def __init__(self, yearly_seasonality=True, weekly_seasonality=False,
                 daily_seasonality=False, climate_vars=None):
        self.climate_vars = climate_vars or []
        self.model = Prophet(yearly_seasonality=yearly_seasonality,
                             weekly_seasonality=weekly_seasonality,
                             daily_seasonality=daily_seasonality)
        # 动态添加气候回归器
        for v in self.climate_vars:
            self.model.add_regressor(v)

    def fit(self, df: pd.DataFrame, target_col='price', date_col='date'):
        tmp = df[[date_col, target_col] + self.climate_vars].dropna()
        tmp = tmp.rename(columns={date_col: 'ds', target_col: 'y'})
        self.model.fit(tmp)
        logger.info("ProphetWrapper fit complete")

    def predict(self, df_future: pd.DataFrame, date_col='date'):
        # 保证列名
        tmp = df_future[[date_col] + self.climate_vars].copy()
        tmp = tmp.rename(columns={date_col: 'ds'})
        forecast = self.model.predict(tmp)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def save(self, path: str):
        joblib.dump(self, path)
        logger.info(f"ProphetWrapper saved to {path}")

    @staticmethod
    def load(path: str):
        obj = joblib.load(path)
        logger.info(f"ProphetWrapper loaded from {path}")
        return obj
