from fastapi import FastAPI, HTTPException
import xarray as xr
import joblib
import os

app = FastAPI()

# 加载训练好的模型（joblib 保存）
MODEL = joblib.load("artifacts/ensemble.pkl")

@app.post("/predict")
def predict(cmip6_nc: str, commodity: str, horizon_month: int = 6):
    try:
        ds = xr.open_dataset(f"data/{cmip6_nc}")
        # 提取气候特征
        features = extract_features(ds, commodity)
        pred, lower, upper = MODEL.predict(features, horizon_month)
        return {"pred_price": pred, "ci_lower": lower, "ci_upper": upper}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def extract_features(ds, commodity):
    # 根据商品选择区域、变量，返回 DataFrame
    region = REGION_MAP[commodity]
    df = ds.sel(lat=slice(region[2], region[3]), lon=slice(region[0], region[1])).to_dataframe()
    return df.groupby(["time"]).mean().reset_index()
