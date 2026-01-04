#!/usr/bin/env python3
"""
部署 MLflow 模型为 REST API
- 自动加载最新 best run 的模型
- 支持 POST /predict
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
from typing import List

app = FastAPI(title="CMIP6 Crop Price Predictor", version="1.0")

# 全局模型变量
model = None
feature_names = None

class PredictionRequest(BaseModel):
    crop: str  # rice, corn, barley
    features: List[float]  # 按顺序: [tas_avg, pr_sum, tas_lag1, ..., price_lag7]

@app.on_event("startup")
async def load_model():
    global model, feature_names
    try:
        # 从 MLflow 加载最新 best model（按 R² 最高）
        experiment_name = "cmip6-crop-price-prediction"
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            order_by=["metrics.r2 DESC"],
            max_results=1
        )
        if runs.empty:
            raise RuntimeError("No runs found in MLflow")
        
        best_run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{best_run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        feature_names = [
            "tas_avg", "pr_sum",
            "tas_lag1", "pr_lag1", "price_lag1",
            "tas_lag7", "pr_lag7", "price_lag7"
        ]
        print(f"✅ Loaded best model from run {best_run_id}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if len(request.features) != len(feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_names)} features, got {len(request.features)}"
        )
    
    # 预测
    df = pd.DataFrame([request.features], columns=feature_names)
    prediction = model.predict(df)[0]
    
    return {
        "crop": request.crop,
        "predicted_price": float(prediction),
        "model_used": str(type(model).__name__)
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}