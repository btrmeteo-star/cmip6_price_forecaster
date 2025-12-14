from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, numpy as np
from typing import Literal

app = FastAPI(title="CMIP6-Price-Forecaster", version="1.0.0")

# 1. 占位模型（后续用真实模型覆盖即可）
MODEL = joblib.load("artifacts/ensemble.pkl")

# 2. 请求体（零文件依赖）
class PredictIn(BaseModel):
    commodity: Literal["maize", "wheat", "soybean"]
    horizon_month: int = 6


def extract_features(commodity: str) -> np.ndarray:
    """纯内存虚拟数据，后续换真实 CMIP6 即可"""
    np.random.seed(42)          # 固定结果便于调试
    return np.random.rand(36).reshape(1, -1)


@app.post("/predict")
def predict(body: PredictIn):
    try:
        X = extract_features(body.commodity)
        pred = float(MODEL.predict(X)[0])
        # 占位置信区间：±5 %
        lower = pred * 0.95
        upper = pred * 1.05
        return {"pred_price": pred, "ci_lower": lower, "ci_upper": upper}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
