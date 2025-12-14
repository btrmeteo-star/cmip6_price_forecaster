from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib,numpy as np,xarray as xr
from typing import Literal

app = FastAPI()

MODEL = joblib.load("artifacts/ensemble.pkl")
COMMODITY_REGION = {
    "maize":  [-100, -80, 35, 50],
    "wheat":  [30, 60, 45, 55],
    "soybean":[-100, -50, -40, 10],
}

class PredictIn(BaseModel):
    cmip6_nc: str
    commodity: Literal["maize", "wheat", "soybean"]
    horizon_month: int = 6

def extract_features(ds: xr.Dataset, commodity: str) -> np.ndarray:
    region = COMMODITY_REGION[commodity]
    ds = ds.sel(lat=slice(region[2], region[3]), lon=slice(region[0], region[1]))
    df = ds[["pr", "tasmax", "tasmin"]].to_dataframe().dropna()
    df = df.groupby(df.index.get_level_values('time').to_period('Q')).mean()
    return df.values.flatten()[:36]

@app.post("/predict")
def predict(body: PredictIn):
    try:
        ds = xr.open_dataset(f"data/{body.cmip6_nc}")
        X = extract_features(ds, body.commodity).reshape(1, -1)
        # 占位预测：用 RidgeCV .predict
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
