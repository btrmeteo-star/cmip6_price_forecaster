"""
测试特征工程模块
"""
import pytest
import pandas as pd
import numpy as np
from src.feature_engineer import FeatureEngineer

def test_create_climate_features():
    """测试气候特征创建"""
    engineer = FeatureEngineer()
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'year': range(2000, 2020),
        'growing_season_mean_temp': np.random.normal(20, 2, 20),
        'growing_season_total_precip': np.random.normal(500, 100, 20),
        'extreme_heat_days': np.random.randint(0, 30, 20)
    })
    
    features = engineer.create_climate_features(test_data)
    
    # 检查是否创建了新特征
    assert len(features.columns) > len(test_data.columns)
    assert 'mean_temp_lag1' in features.columns
    assert 'growing_season_mean_temp_rolling_mean_3' in features.columns
    
    # 检查没有NaN（除了滞后特征的前几行）
    assert features.iloc[3:].isnull().sum().sum() == 0

def test_prepare_training_data():
    """测试训练数据准备"""
    engineer = FeatureEngineer()
    
    # 创建测试数据
    n_samples = 100
    features = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples)
    })
    
    features.loc[10, 'feature1'] = np.nan  # 添加一个缺失值
    
    target = pd.Series(np.random.randn(n_samples), name='target')
    
    # 创建合并数据框
    merged = pd.concat([features, target], axis=1)
    merged['year'] = range(2000, 2000 + n_samples)
    
    splits = engineer.prepare_training_data(merged, target_col='target')
    
    # 检查返回的数据结构
    assert 'train' in splits
    assert 'val' in splits
    assert 'test' in splits
    assert 'X' in splits
    assert 'y' in splits
    
    # 检查数据形状
    assert splits['X'].shape[0] == n_samples
    assert splits['y'].shape[0] == n_samples
    
    # 检查缺失值处理
    assert not splits['X'].isnull().any().any()

def test_scaler_persistence():
    """测试标准化器的持久性"""
    engineer = FeatureEngineer()
    
    # 创建测试数据
    data1 = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
    data2 = pd.DataFrame({'feature': [6, 7, 8, 9, 10]})
    
    # 第一次拟合
    splits1 = engineer.prepare_training_data(
        pd.concat([data1, pd.Series([10, 20, 30, 40, 50], name='target')], axis=1)
    )
    
    # 检查标准化器已保存
    assert 'feature' in engineer.scalers
    assert engineer.scalers['feature'] is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
