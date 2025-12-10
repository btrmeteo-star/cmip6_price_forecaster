"""
特征工程：创建用于价格预测的气候和经济特征
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self, config=None):
        self.config = config
        self.scalers = {}
        self.imputer = KNNImputer(n_neighbors=5)
        
    def create_climate_features(self, climate_df):
        """从基础气候数据创建高级特征"""
        features = climate_df.copy()
        
        # 1. 滞后特征 (过去1-3年的影响)
        for lag in [1, 2, 3]:
            for col in ['mean_temp', 'total_precip', 'heat_days']:
                if f'growing_season_{col}' in features.columns:
                    features[f'{col}_lag{lag}'] = features[f'growing_season_{col}'].shift(lag)
        
        # 2. 滚动统计特征
        window_sizes = [3, 5, 10]  # 年
        for col in ['growing_season_mean_temp', 'growing_season_total_precip']:
            for window in window_sizes:
                features[f'{col}_rolling_mean_{window}'] = (
                    features[col].rolling(window=window, min_periods=1).mean()
                )
                features[f'{col}_rolling_std_{window}'] = (
                    features[col].rolling(window=window, min_periods=1).std()
                )
        
        # 3. 交互特征（温度与降水的交互效应）
        if ('growing_season_mean_temp' in features.columns and 
            'growing_season_total_precip' in features.columns):
            features['temp_precip_interaction'] = (
                features['growing_season_mean_temp'] * 
                features['growing_season_total_precip']
            )
        
        # 4. 极端事件指标
        # 热浪指标：连续3天以上超过阈值
        if 'extreme_heat_days' in features.columns:
            features['heatwave_intensity'] = (
                features['extreme_heat_days'] / 92  # 生长季总天数
            )
        
        # 5. 干旱指标
        if 'spi_3month' in features.columns:
            features['drought_severity'] = (
                features['spi_3month'].apply(lambda x: -x if x < -1 else 0)
            )
        
        # 6. 季节特征
        features['year_sin'] = np.sin(2 * np.pi * features['year'] / 10)
        features['year_cos'] = np.cos(2 * np.pi * features['year'] / 10)
        
        return features
    
    def create_economic_features(self, price_df, yield_df=None):
        """创建经济特征"""
        features = pd.DataFrame()
        
        # 价格相关特征
        features['price'] = price_df['price'].values
        
        # 1. 价格变化率
        features['price_return'] = price_df['price'].pct_change()
        features['price_volatility'] = price_df['price'].rolling(12).std()
        
        # 2. 技术指标（简化版）
        features['price_ma_12m'] = price_df['price'].rolling(12).mean()
        features['price_ma_24m'] = price_df['price'].rolling(24).mean()
        features['price_momentum'] = price_df['price'] / price_df['price'].shift(12) - 1
        
        # 3. 如果有产量数据，添加供给特征
        if yield_df is not None:
            features['yield'] = yield_df['yield'].values
            features['yield_trend'] = (
                features['yield'].rolling(5).mean() / features['yield'].rolling(10).mean() - 1
            )
            # 库存消费比（简化）
            features['stock_to_use'] = features['yield'] / features['price']
        
        # 4. 宏观经济代理变量（使用价格本身衍生）
        features['market_trend'] = (features['price'] > features['price_ma_12m']).astype(int)
        
        return features
    
    def merge_features(self, climate_features, economic_features, 
                       price_df, common_key='year'):
        """合并气候和经济特征"""
        # 确保有共同的时间键
        if common_key not in climate_features.columns:
            climate_features[common_key] = climate_features.index
        
        if common_key not in economic_features.columns:
            economic_features[common_key] = economic_features.index
        
        # 合并
        merged = pd.merge(
            climate_features, 
            economic_features, 
            on=common_key, 
            how='inner'
        )
        
        # 添加目标变量：未来价格变化（1年领先）
        merged['target_price_1y'] = price_df['price'].shift(-12).values
        
        return merged
    
    def prepare_training_data(self, merged_features, target_col='target_price_1y'):
        """准备训练数据，处理缺失值并标准化"""
        # 分离特征和目标
        feature_cols = [col for col in merged_features.columns 
                       if col not in ['year', target_col, 'price']]
        
        X = merged_features[feature_cols].copy()
        y = merged_features[target_col].copy()
        
        # 处理缺失值
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # 标准化特征
        for col in X_imputed.columns:
            if X_imputed[col].std() > 0:  # 避免常数列
                scaler = StandardScaler()
                X_imputed[col] = scaler.fit_transform(X_imputed[[col]])
                self.scalers[col] = scaler
        
        # 创建时间序列分割索引
        train_size = int(len(X_imputed) * 0.7)
        val_size = int(len(X_imputed) * 0.15)
        
        splits = {
            'train': X_imputed.index[:train_size],
            'val': X_imputed.index[train_size:train_size+val_size],
            'test': X_imputed.index[train_size+val_size:],
            'X': X_imputed,
            'y': y,
            'feature_names': feature_cols
        }
        
        return splits
    
    def save_features(self, features, output_path):
        """保存特征到文件"""
        features.to_csv(output_path, index=False)
        print(f"特征已保存至: {output_path}")
