#!/usr/bin/env python3
"""
CMIP6 農產品價格預測 API - v2.0.3 (結構優化版)
修復：中文亂碼、類型轉換、目錄結構
"""

import os
import io
import base64
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from loguru import logger

# ==================== Matplotlib 中文配置 (關鍵修復) ====================
import matplotlib
matplotlib.use('Agg')  # 非互動式後端
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 設置中文字體 (嘗試常見字體，防止亂碼)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# ==================== FastAPI 初始化 ====================
app = FastAPI(title="CMIP6 農產品價格預測 API", version="2.0.3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

MODEL_DIR = "models"
DATA_DIR = "data"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ==================== Pydantic 模型 ====================
class PredictionRequest(BaseModel):
    data: Dict[str, float]

class PredictionResponse(BaseModel):
    predicted_price: float
    feature_importance: Dict[str, float] = {}

class BatchPredictionResponse(BaseModel):
    results: List[Dict]
    success_count: int
    error_count: int

# ==================== 路由 ====================

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"error": "Frontend not found"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(crop: str = "corn", request: PredictionRequest = None):
    if not request or not request.data:
        raise HTTPException(status_code=400, detail="缺少 'data' 字段")
    try:
        model_path = Path(MODEL_DIR) / f"{crop}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"模型 '{crop}' 不存在")
        
        model_package = joblib.load(model_path)
        model = model_package.get("model")
        feature_names = model_package.get("feature_names", [])
        
        if model is None:
            raise ValueError("模型為空")
        
        input_data = request.data
        missing = set(feature_names) - set(input_data.keys())
        if missing:
            raise HTTPException(status_code=400, detail=f"缺少特徵：{sorted(missing)}")
        
        X = pd.DataFrame([input_data], columns=feature_names)
        prediction = model.predict(X)[0]
        
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            for name, importance in zip(feature_names, model.feature_importances_):
                feature_importance[name] = round(float(importance), 4)
        
        return PredictionResponse(predicted_price=float(prediction), feature_importance=feature_importance)
    except Exception as e:
        logger.exception("預測失敗")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/historical-price")
async def get_historical_price(crop: str = "corn", periods: str = "12"):
    try:
        n_periods = int(periods) if periods else 12
        dates = []
        prices = []
        # 模擬數據邏輯
        base_price = {"corn": 180, "wheat": 160, "soybean": 200}.get(crop, 180)
        for i in range(n_periods):
            days_ago = (n_periods - 1 - i) * 30
            d = datetime.now() - timedelta(days=days_ago)
            dates.append(d.strftime('%Y-%m'))
            prices.append(round(base_price + np.random.uniform(-20, 20), 2))
        return {"dates": dates, "prices": prices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/historical-price-chart")
async def get_historical_price_chart(crop: str = "corn", periods: str = "12"):
    try:
        n_periods = int(periods) if periods else 12
        data = await get_historical_price(crop, str(n_periods))
        
        plt.figure(figsize=(10, 5))
        plt.plot(data['dates'], data['prices'], marker='o', linewidth=2, color='#2563eb', label='價格')
        plt.fill_between(range(len(data['dates'])), data['prices'], alpha=0.3, color='#2563eb')
        plt.title(f'{crop.upper()} 歷史價格趨勢', fontsize=14, fontweight='bold')
        plt.xlabel('月份', fontsize=12)
        plt.ylabel('價格', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return {"chart": f"data:image/png;base64,{img_base64}"}
    except Exception as e:
        logger.error(f"圖表錯誤：{e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/climate-data")
async def get_climate_data(crop: str = "corn", periods: str = "12"):
    try:
        n_periods = int(periods) if periods else 12
        dates = []
        precip = []
        temp = []
        for i in range(n_periods):
            days_ago = (n_periods - 1 - i) * 30
            d = datetime.now() - timedelta(days=days_ago)
            dates.append(d.strftime('%Y-%m'))
            precip.append(round(np.random.uniform(0.5, 2.5), 2))
            temp.append(round(np.random.uniform(22, 32), 2))
        return {"dates": dates, "precipitation": precip, "temperature": temp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/climate-chart")
async def get_climate_chart(crop: str = "corn", periods: str = "12"):
    try:
        n_periods = int(periods) if periods else 12
        data = await get_climate_data(crop, str(n_periods))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.bar(data['dates'], data['precipitation'], color='#3b82f6', alpha=0.7, label='降雨量')
        ax1.set_ylabel('降雨量 (mm)', fontsize=11, color='#3b82f6')
        ax1.set_title('氣候數據變化', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#3b82f6')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        ax2.plot(data['dates'], data['temperature'], marker='s', linewidth=2, color='#ef4444', label='最高溫')
        ax2.set_ylabel('最高溫 (°C)', fontsize=11, color='#ef4444')
        ax2.tick_params(axis='y', labelcolor='#ef4444')
        ax2.set_xlabel('月份', fontsize=11)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return {"chart": f"data:image/png;base64,{img_base64}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feature-importance")
async def get_feature_importance(crop: str = "corn"):
    try:
        model_path = Path(MODEL_DIR) / f"{crop}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"模型 '{crop}' 不存在")
        model_package = joblib.load(model_path)
        model = model_package.get("model")
        feature_names = model_package.get("feature_names", [])
        if not hasattr(model, 'feature_importances_'):
            raise HTTPException(status_code=400, detail="不支持特徵重要性")
        importance_dict = {name: round(float(imp), 4) for name, imp in zip(feature_names, model.feature_importances_)}
        return {"feature_importance": dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feature-importance-chart")
async def get_feature_importance_chart(crop: str = "corn"):
    try:
        data = await get_feature_importance(crop)
        features = list(data['feature_importance'].keys())
        importances = list(data['feature_importance'].values())
        
        feature_labels = {
            'pr': '當前降雨量', 'pr_lag1': '前一期降雨量', 'pr_lag2': '前兩期降雨量',
            'pr_std': '降雨標準差', 'price_lag1': '前一期價格', 'price_lag2': '前兩期價格',
            'tasmax': '當前最高溫', 'tasmax_lag1': '前一期最高溫',
            'tasmax_lag2': '前兩期最高溫', 'tasmax_mean': '最高溫平均值'
        }
        labels = [feature_labels.get(f, f) for f in features]
        
        plt.figure(figsize=(10, 6))
        plt.barh(labels, importances, color='#667eea')
        plt.xlabel('重要性', fontsize=12)
        plt.title(f'{crop.upper()} 模型特徵重要性', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return {"chart": f"data:image/png;base64,{img_base64}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(crop: str = "corn", file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        model_path = Path(MODEL_DIR) / f"{crop}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"模型 '{crop}' 不存在")
        model_package = joblib.load(model_path)
        model = model_package.get("model")
        feature_names = model_package.get("feature_names", [])
        missing = set(feature_names) - set(df.columns)
        if missing:
            raise HTTPException(status_code=400, detail=f"CSV 缺少列：{sorted(missing)}")
        X = df[feature_names]
        predictions = model.predict(X)
        results = [{"row": i+1, "predicted_price": round(float(p), 2), "status": "success"} for i, p in enumerate(predictions)]
        return BatchPredictionResponse(results=results, success_count=len(results), error_count=0)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "2.0.3"}

@app.get("/api/crops")
async def get_supported_crops():
    crops = [f.stem for f in Path(MODEL_DIR).glob("*.joblib")]
    return {"crops": crops}