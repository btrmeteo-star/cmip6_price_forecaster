"""
经济数据获取与处理：价格、产量、库存等
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

class EconomicDataProcessor:
    def __init__(self, config=None):
        self.config = config
        
    def fetch_futures_data(self, symbol, start_date, end_date):
        """
        从雅虎财经获取期货数据
        符号示例: 'ZC=F' (玉米), 'ZS=F' (大豆), 'ZW=F' (小麦)
        """
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            data.reset_index(inplace=True)
            data.rename(columns={'Date': 'date', 'Close': 'price'}, inplace=True)
            return data[['date', 'price']]
        except Exception as e:
            print(f"获取期货数据失败: {e}")
            # 返回示例数据
            return self._create_sample_price_data(start_date, end_date)
    
    def fetch_usda_data(self, commodity, data_type='yield'):
        """获取USDA数据（简化版本，实际需要API密钥）"""
        # USDA数据源映射
        usda_sources = {
            'corn': {
                'yield': 'https://quickstats.nass.usda.gov/api/api_GET/',
                'production': 'https://quickstats.nass.usda.gov/api/api_GET/'
            },
            'soybeans': {
                'yield': 'https://quickstats.nass.usda.gov/api/api_GET/'
            }
        }
        
        print(f"注意: 需要USDA API密钥获取真实数据")
        print(f"请访问: https://quickstats.nass.usda.gov/api/")
        
        # 返回示例数据
        return self._create_sample_yield_data()
    
    def load_from_csv(self, filepath, date_col='date', value_col='value'):
        """从CSV文件加载数据"""
        try:
            df = pd.read_csv(filepath)
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
            return df
        except Exception as e:
            print(f"加载CSV文件失败: {e}")
            return None
    
    def process_price_data(self, price_df, frequency='M'):
        """处理价格数据：重采样、填充、计算回报"""
        if price_df is None or len(price_df) == 0:
            return None
        
        df = price_df.copy()
        
        # 确保日期格式
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
        
        # 重采样到月度频率
        if frequency == 'M':
            df_resampled = df['price'].resample('M').last()
        elif frequency == 'Q':
            df_resampled = df['price'].resample('Q').last()
        else:
            df_resampled = df['price']
        
        # 向前填充缺失值
        df_resampled = df_resampled.ffill()
        
        # 计算对数回报和波动率
        result = pd.DataFrame({
            'price': df_resampled,
            'log_return': np.log(df_resampled / df_resampled.shift(1)),
            'volatility_30d': df_resampled.rolling(30).std()
        })
        
        return result.reset_index()
    
    def create_economic_indicators(self, price_df, yield_df=None):
        """创建综合经济指标"""
        indicators = pd.DataFrame()
        
        # 基础价格指标
        indicators['price_level'] = price_df['price']
        indicators['price_momentum'] = price_df['price'].pct_change(12)  # 12个月动量
        
        # 如果有产量数据
        if yield_df is not None and 'yield' in yield_df.columns:
            indicators['yield'] = yield_df['yield']
            indicators['yield_growth'] = yield_df['yield'].pct_change()
            
            # 简单的供给压力指标
            indicators['supply_pressure'] = (
                indicators['yield'] / indicators['price_level']
            )
        
        # 季节性调整
        indicators = self._adjust_seasonality(indicators)
        
        return indicators
    
    def _create_sample_price_data(self, start_date, end_date):
        """创建示例价格数据（用于测试）"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)
        
        # 随机游走生成价格
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.02, n)
        price = 100 * np.exp(np.cumsum(returns))
        
        # 添加季节性和趋势
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 365)
        trend = 0.0003 * np.arange(n)
        
        price = price + seasonal + trend
        
        df = pd.DataFrame({
            'date': dates,
            'price': price
        })
        
        return df
    
    def _create_sample_yield_data(self):
        """创建示例产量数据（用于测试）"""
        years = np.arange(1980, 2024)
        n = len(years)
        
        # 趋势增长 + 随机波动
        np.random.seed(42)
        trend = 50 + 2 * (years - 1980)
        noise = np.random.normal(0, 5, n)
        
        yield_data = trend + noise
        
        df = pd.DataFrame({
            'year': years,
            'yield': yield_data
        })
        
        return df
    
    def _adjust_seasonality(self, df):
        """简单的季节性调整"""
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                # 12个月移动平均去除季节性
                df[f'{col}_sa'] = df[col] / df[col].rolling(12).mean()
        return df
    
    def save_processed_data(self, data_dict, output_dir):
        """保存处理后的数据"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, data in data_dict.items():
            if data is not None:
                filepath = os.path.join(output_dir, f'{name}.csv')
                data.to_csv(filepath, index=False)
                print(f"已保存: {filepath}")
