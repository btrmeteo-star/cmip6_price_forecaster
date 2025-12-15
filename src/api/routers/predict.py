import os, joblib, xarray as xr, pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

MODEL = joblib.load(os.getenv("MODEL_PATH"))

class PredictRequest(BaseModel):
    cmip6_nc: str          # 檔案路徑（掛載在 /app/data/...）
    commodity: str
    horizon_month: int

router = APIRouter()

@router.post("/predict")
def predict(req: PredictRequest):
    file_path = f"/app/{req.cmip6_nc}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="NetCDF not found")
    ds = xr.open_dataset(file_path)
    df = ds.to_dataframe().reset_index()
    # TODO: 與現貨表拼接 & 特徵工程
    X = df[['tas', 'pr']].mean().values.reshape(1, -1)  # demo
    price = float(MODEL.predict(X)[0])
    return {
        "commodity": req.commodity,
        "horizon_month": req.horizon_month,
        "predicted_price": price,
        "unit": "USD/tonne",
        "model": "xgb",
        "nc_source": req.cmip6_nc
    }
