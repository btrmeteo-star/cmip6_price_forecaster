"""
集成模型：结合多个预测模型的结果
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class EnsembleCommodityModel:
    def __init__(self, config=None):
        self.config = config or {}
        self.models = {}
        self.scaler = StandardScaler()
        self.meta_model = LinearRegression()
        self.feature_importance = {}
        
    def build_base_models(self):
        """构建基础模型集合"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0),
            'prophet_based': None  # 将使用Prophet模型的结果作为特征
        }
        
    def prepare_ensemble_features(self, X, y, prophet_predictions=None):
        """准备集成学习的特征"""
        # 基础特征
        ensemble_features = X.copy()
        
        # 如果提供Prophet预测，作为额外特征
        if prophet_predictions is not None:
            ensemble_features['prophet_pred'] = prophet_predictions
        
        # 添加时间特征
        if hasattr(X, 'index'):
            if hasattr(X.index, 'year'):
                ensemble_features['year'] = X.index.year
                ensemble_features['month'] = X.index.month if hasattr(X.index, 'month') else 1
        
        return ensemble_features, y
    
    def train_ensemble(self, X_train, y_train, X_val=None, y_val=None):
        """训练集成模型"""
        # 1. 训练所有基础模型
        base_predictions_train = {}
        
        for name, model in self.models.items():
            if model is not None:
                print(f"训练基础模型: {name}")
                model.fit(X_train, y_train)
                
                # 收集训练集预测
                y_pred_train = model.predict(X_train)
                base_predictions_train[name] = y_pred_train
                
                # 计算特征重要性（如果可用）
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
        
        # 2. 创建元特征矩阵（基础模型的预测）
        meta_features_train = pd.DataFrame(base_predictions_train)
        
        # 3. 训练元模型
        self.meta_model.fit(meta_features_train, y_train)
        
        # 4. 验证（如果提供验证集）
        if X_val is not None and y_val is not None:
            base_predictions_val = {}
            for name, model in self.models.items():
                if model is not None:
                    y_pred_val = model.predict(X_val)
                    base_predictions_val[name] = y_pred_val
            
            meta_features_val = pd.DataFrame(base_predictions_val)
            y_pred_val_ensemble = self.meta_model.predict(meta_features_val)
            
            # 计算验证集指标
            val_metrics = self._calculate_metrics(y_val, y_pred_val_ensemble)
            print(f"验证集指标: {val_metrics}")
            
            return val_metrics
        
        return None
    
    def predict(self, X, include_components=False):
        """生成集成预测"""
        # 获取所有基础模型的预测
        base_predictions = {}
        for name, model in self.models.items():
            if model is not None:
                base_predictions[name] = model.predict(X)
        
        # 创建元特征矩阵
        meta_features = pd.DataFrame(base_predictions)
        
        # 元模型集成预测
        ensemble_pred = self.meta_model.predict(meta_features)
        
        if include_components:
            return {
                'ensemble': ensemble_pred,
                'components': base_predictions,
                'weights': self.meta_model.coef_ if hasattr(self.meta_model, 'coef_') else None
            }
        else:
            return ensemble_pred
    
    def predict_future(self, future_features, steps=12):
        """预测未来多个时间步"""
        predictions = []
        components_history = []
        
        # 递归预测（简单实现）
        current_features = future_features.copy()
        
        for step in range(steps):
            # 生成当前步的预测
            pred_result = self.predict(current_features, include_components=True)
            
            predictions.append(pred_result['ensemble'][0])
            components_history.append(pred_result['components'])
            
            # 更新特征用于下一步预测（简化处理）
            # 在实际应用中，需要更复杂的状态更新逻辑
            if step < steps - 1:
                # 这里可以根据具体需求更新特征
                pass
        
        result = {
            'predictions': np.array(predictions),
            'components': components_history,
            'steps': steps
        }
        
        return result
    
    def _calculate_metrics(self, y_true, y_pred):
        """计算预测指标"""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'r2': r2_score(y_true, y_pred)
        }
        
        return metrics
    
    def get_feature_importance(self, top_n=10):
        """获取特征重要性汇总"""
        if not self.feature_importance:
            return None
        
        # 合并所有模型的特征重要性
        all_importance = []
        for model_name, importance_df in self.feature_importance.items():
            if importance_df is not None:
                importance_df['model'] = model_name
                all_importance.append(importance_df.head(top_n))
        
        if all_importance:
            combined = pd.concat(all_importance, ignore_index=True)
            return combined
        else:
            return None
    
    def save_model(self, filepath):
        """保存模型"""
        import joblib
        joblib.dump({
            'models': self.models,
            'meta_model': self.meta_model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance
        }, filepath)
        print(f"模型已保存至: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        import joblib
        data = joblib.load(filepath)
        self.models = data['models']
        self.meta_model = data['meta_model']
        self.scaler = data['scaler']
        self.feature_importance = data.get('feature_importance', {})
        print(f"模型已从 {filepath} 加载")
