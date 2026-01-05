import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# 创建模型目录
os.makedirs("models", exist_ok=True)

# 加载并合并数据
processed_dir = "data/processed"
feature_files = [f for f in os.listdir(processed_dir) if f.endswith("_features.csv")]
df_list = []
for fname in feature_files:
    df_crop = pd.read_csv(os.path.join(processed_dir, fname))
    df_list.append(df_crop)
df = pd.concat(df_list, ignore_index=True)

# 排除非数值列，并显式排序（关键！）
exclude_cols = {'time', 'price'}
feature_cols = sorted([col for col in df.columns if col not in exclude_cols])  # ✅ 排序！
X = df[feature_cols]
y = df["price"]

best_r2 = -float("inf")
best_model_path = None

for name, model in [
    ("LinearRegression", LinearRegression()),
    ("RandomForest", RandomForestRegressor(n_estimators=50, random_state=42))
]:
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print(f"{name} | R²: {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model_path = f"models/best_model.joblib"
        joblib.dump(model, best_model_path)

print(f"\n🏆 最佳模型 R² = {best_r2:.4f}")
print(f"📁 模型已保存至: {best_model_path}")
print(f"🔍 训练特征顺序: {feature_cols}")

# 保存路径供 app.py 使用
with open("best_model_path.txt", "w") as f:
    f.write(best_model_path)
