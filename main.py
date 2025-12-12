from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, os, xarray as xr
from src.models.ensemble_model import EnsembleModel

app = FastAPI(title="CMIP6-Price-Forecaster", version="1.0.0")
model = EnsembleModel.load("artifacts/ensemble.pkl")

class PredictIn(BaseModel):
    cmip6_nc: str          # 已上傳 netCDF 路徑
    commodity: str
    horizon_month: int = 6

class PredictOut(BaseModel):
    pred_price: float
    ci_lower: float
    ci_upper: float

@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    try:
        ds = xr.open_dataset(body.cmip6_nc)
        pred, lower, upper = model.predict(ds, body.commodity, body.horizon_month)
        return PredictOut(pred_price=pred, ci_lower=lower, ci_upper=upper)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
