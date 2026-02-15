import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

os.makedirs("models", exist_ok=True)
processed_dir = "data/processed"
feature_files = [f for f in os.listdir(processed_dir) if f.endswith("_features.csv")]

for fname in feature_files:
    crop = fname.replace("_features.csv", "")
    print(f"\n🌱 训练作物: {crop}")
    
    df = pd.read_csv(os.path.join(processed_dir, fname))
    
    # ✅ 关键修复：跳过空数据
    if df.empty or len(df) == 0:
        print(f"  ⚠️ 跳过 {crop}: 无有效数据")
        continue
    
    exclude_cols = {'time', 'price'}
    feature_cols = sorted([col for col in df.columns if col not in exclude_cols])
    X = df[feature_cols]
    y = df["price"]
    
    # ✅ 再次确保 X 非空
    if X.shape[0] == 0:
        print(f"  ⚠️ 跳过 {crop}: 特征矩阵为空")
        continue
    
    best_r2 = -float("inf")
    best_model = None
    
    for name, model in [
        ("LinearRegression", LinearRegression()),
        ("RandomForest", RandomForestRegressor(n_estimators=50, random_state=42))
    ]:
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))
        print(f"  {name} | R²: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
    
    model_path = f"models/{crop}.joblib"
    joblib.dump({
        'model': best_model,
        'feature_names': feature_cols,
        'r2_score': best_r2
    }, model_path)
    
    print(f"  📁 保存至: {model_path}")

print("\n✅ 训练完成！")
