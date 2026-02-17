# train_model.py
import joblib
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# 示例：训练玉米模型
X_train_corn = pd.DataFrame({
    'pr': [1.2, 1.1, 1.3],
    'pr_lag1': [0.8, 0.9, 0.7],
    'pr_lag2': [1.0, 1.1, 0.9],
    'pr_std': [0.5, 0.6, 0.4],
    'price_lag1': [105.0, 104.0, 106.0],
    'price_lag2': [102.0, 103.0, 101.0],
    'tasmax': [26.5, 26.3, 26.7],
    'tasmax_lag1': [26.0, 25.8, 26.1],
    'tasmax_lag2': [25.8, 25.7, 25.9],
    'tasmax_mean': [26.2, 26.1, 26.3]
})
y_train_corn = [186.88, 185.50, 187.20]

model_corn = RandomForestRegressor(n_estimators=100)
model_corn.fit(X_train_corn, y_train_corn)

# 保存为 dict 格式（兼容性更好）
joblib.dump({
    "model": model_corn,
    "feature_names": X_train_corn.columns.tolist()
}, "models/corn.joblib")