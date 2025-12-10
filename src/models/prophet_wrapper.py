"""
整合气候因子到Prophet预测模型
"""
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class ClimateProphetModel:
    def __init__(self, growth='linear', yearly_seasonality=True):
        self.model = Prophet(
            growth=growth,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        self.scalers = {}
        
    def add_climate_regressors(self, climate_features):
        """添加气候因子作为外生变量"""
        for feature in climate_features:
            self.model.add_regressor(feature)
            self.scalers[feature] = StandardScaler()
    
    def prepare_data(self, price_df, climate_df):
        """
        准备Prophet所需数据格式
        price_df: 包含['ds'(日期), 'y'(价格)]的DataFrame
        climate_df: 包含气候指标的DataFrame，需有'date'列
        """
        # 合并数据
        climate_df['ds'] = pd.to_datetime(climate_df['date'])
        merged_df = pd.merge(price_df, climate_df, on='ds', how='inner')
        
        # 标准化气候特征
        for feature in self.scalers.keys():
            if feature in merged_df.columns:
                merged_df[feature] = self.scalers[feature].fit_transform(
                    merged_df[[feature]]
                )
        
        return merged_df
    
    def fit(self, price_df, climate_df):
        """训练模型"""
        train_df = self.prepare_data(price_df, climate_df)
        self.model.fit(train_df)
        
    def predict(self, future_climate_df, periods=365):
        """生成预测"""
        # 创建未来时间框架
        future_dates = self.model.make_future_dataframe(
            periods=periods, 
            include_history=True
        )
        
        # 合并未来气候数据
        future_climate_df['ds'] = pd.to_datetime(future_climate_df['date'])
        future_df = pd.merge(future_dates, future_climate_df, on='ds', how='left')
        
        # 填充缺失值（向前填充）
        for feature in self.scalers.keys():
            if feature in future_df.columns:
                # 标准化未来气候数据
                future_df[feature] = self.scalers[feature].transform(
                    future_df[[feature]]
                )
                future_df[feature] = future_df[feature].ffill()
        
        # 生成预测
        forecast = self.model.predict(future_df)
        
        return forecast
    
    def plot_components(self, forecast):
        """可视化预测组件"""
        fig = self.model.plot_components(forecast)
        return fig

# 使用示例
def example_usage():
    # 加载数据
    price_data = pd.read_csv("data/economic/corn_prices.csv")
    climate_data = pd.read_csv("data/processed/climate_indicators.csv")
    
    # 初始化模型
    model = ClimateProphetModel()
    
    # 添加气候回归因子
    climate_features = [
        'growing_season_mean_temp',
        'extreme_heat_days',
        'spi_3month'
    ]
    model.add_climate_regressors(climate_features)
    
    # 训练模型
    model.fit(price_data, climate_data)
    
    # 加载未来气候情景
    future_climate = pd.read_csv("data/processed/future_climate_ssp585.csv")
    
    # 生成预测（未来30年）
    forecast = model.predict(future_climate, periods=30*365)
    
    # 保存结果
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(
        "data/outputs/corn_price_forecast_ssp585.csv", 
        index=False
    )
    
    return forecast
