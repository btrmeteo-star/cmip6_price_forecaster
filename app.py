from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xarray as xr
import joblib
import numpy as np
from typing import Literal

app = FastAPI(title="CMIP6-Price-Forecaster", version="1.0.0")

# 1. 加载 ensemble 模型（你之前保存的 joblib）
MODEL = joblib.load("artifacts/ensemble.pkl")

# 2. 商品 → 区域映射（与训练时一致）
COMMODITY_REGION = {
    "maize":  [-100, -80, 35, 50],
    "wheat":  [30, 60, 45, 55],
    "soybean":[-100, -50, -40, 10],
}

# 3. 请求体
class PredictIn(BaseModel):
    cmip6_nc: str          # 文件名，例如 "test.nc"
    commodity: Literal["maize", "wheat", "soybean"]
    horizon_month: int = 6


# 4. 特征工程（与训练时一致）
def extract_features(ds: xr.Dataset, commodity: str) -> np.ndarray:
    """返回 1 行 N 列的 numpy 向量，与训练时特征顺序一致"""
    region = COMMODITY_REGION[commodity]
    ds = ds.sel(lat=slice(region[2], region[3]),
                lon=slice(region[0], region[1]))

    # 示例：只用 3 个气候指标，按季度平均
    df = ds[["pr", "tasmax", "tasmin"]].to_dataframe().dropna()
    df = df.groupby(df.index.get_level_values('time').to_period('Q')).mean()
    # 展开成 1 维向量（长度 = 3 变量 × 季度数）
    return df.values.flatten()[:36]          # 截断/补零保持固定长度


# 5. 预测端点
@app.post("/predict", response_model=dict)
def predict(body: PredictIn):
    try:
        ds = xr.open_dataset(f"data/{body.cmip6_nc}")
        X = extract_features(ds, body.commodity).reshape(1, -1)
        pred, lower, upper = MODEL.predict(X, body.horizon_month)
        return {
            "pred_price": float(pred),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# 6. 健康检查
@app.get("/health")
def health():
    return {"status": "ok"}
