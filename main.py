#!/usr/bin/env python3
"""
CMIP6 农产品价格预测 API - 增强版 v2.0
支持：预测、历史图表、气候可视化、模型解释、批量预测
"""

import os
import io
import json
import base64
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from loguru import logger
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# 初始化 FastAPI 应用
app = FastAPI(
    title="CMIP6 农产品价格预测 API",
    description="基于 CMIP6 气候数据与历史价格预测农产品未来价格",
    version="2.0.0"
)

# 添加 CORS 中间件（允许跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置路径
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

# ==================== 前端页面 ====================

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """提供前端页面"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"error": "Frontend index.html not found"}

# ==================== 核心预测功能 ====================

@app.post("/predict", response_model=PredictionResponse)
async def predict(crop: str = "corn", request: PredictionRequest = None):
    """预测指定作物的价格"""
    if not request or not request.data:
        raise HTTPException(status_code=400, detail="请求体中缺少 'data' 字段")

    try:
        model_path = Path(MODEL_DIR) / f"{crop}.joblib"
        if not model_path.exists():
            available = [f.stem for f in Path(MODEL_DIR).glob("*.joblib")]
            raise HTTPException(status_code=404, detail=f"模型 '{crop}' 不存在。可用模型：{available}")

        model_package = joblib.load(model_path)
        model = model_package.get("model")
        feature_names = model_package.get("feature_names", [])

        if model is None:
            raise ValueError("模型对象为空")

        input_data = request.data
        missing = set(feature_names) - set(input_data.keys())
        if missing:
            raise HTTPException(status_code=400, detail=f"缺少必要特征：{sorted(missing)}")

        X = pd.DataFrame([input_data], columns=feature_names)
        prediction = model.predict(X)[0]

        # 获取特征重要性（模型解释）
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            for name, importance in zip(feature_names, model.feature_importances_):
                feature_importance[name] = round(float(importance), 4)

        logger.info(f"✅ 预测成功 | crop={crop} | price={prediction:.2f}")
        return PredictionResponse(
            predicted_price=float(prediction),
            feature_importance=feature_importance
        )

    except Exception as e:
        logger.exception("❌ 预测过程中发生错误")
        raise HTTPException(status_code=500, detail=f"预测失败：{str(e)}")

# ==================== 📈 历史价格图表 ====================

@app.get("/api/historical-price")
async def get_historical_price(crop: str = "corn", periods: int = 12):
    """获取历史价格数据（用于图表）"""
    try:
        # ✅ 关键修复：确保 periods 是整数
        periods = int(periods)
        
        data_path = DATA_DIR / f"{crop}.csv"
        
        if data_path.exists():
            df = pd.read_csv(data_path)
            if 'price' in df.columns:
                prices = df['price'].tail(periods).tolist()
                dates = [(datetime.now() - timedelta(days=i*30)).strftime('%Y-%m') 
                        for i in range(periods-1, -1, -1)]
                return {
                    "dates": dates, 
                    "prices": [float(p) for p in prices]
                }
        
        # 生成模拟数据（如果没有真实数据）
        np.random.seed(42)
        base_price = {"corn": 180, "wheat": 160, "soybean": 200}.get(crop, 180)
        prices = [base_price + np.random.uniform(-20, 20) for _ in range(periods)]
        dates = [(datetime.now() - timedelta(days=i*30)).strftime('%Y-%m') 
                for i in range(periods-1, -1, -1)]
        
        return {
            "dates": dates, 
            "prices": [round(float(p), 2) for p in prices]
        }
    
    except Exception as e:
        logger.error(f"获取历史价格失败：{e}")
        raise HTTPException(status_code=500, detail=f"获取历史数据失败：{str(e)}")

@app.get("/api/historical-price-chart")
async def get_historical_price_chart(crop: str = "corn", periods: int = 12):
    """获取历史价格图表（Base64 图片）"""
    try:
        # ✅ 关键修复：确保 periods 是整数
        periods = int(periods)
        
        data = await get_historical_price(crop, periods)
        
        plt.figure(figsize=(10, 5))
        plt.plot(data['dates'], data['prices'], marker='o', linewidth=2, 
                markersize=8, color='#2563eb')
        plt.fill_between(range(len(data['dates'])), data['prices'], alpha=0.3, color='#2563eb')
        plt.title(f'{crop.upper()} 历史价格趋势', fontsize=14, fontweight='bold')
        plt.xlabel('月份', fontsize=12)
        plt.ylabel('价格', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return {"chart": f"data:image/png;base64,{img_base64}"}
    
    except Exception as e:
        logger.error(f"生成历史价格图表失败：{e}")
        raise HTTPException(status_code=500, detail=f"生成图表失败：{str(e)}")

# ==================== 🌡️ 气候数据可视化 ====================

@app.get("/api/climate-data")
async def get_climate_data(crop: str = "corn", periods: int = 12):
    """获取气候数据（降雨和温度）"""
    try:
        # ✅ 关键修复：确保 periods 是整数
        periods = int(periods)
        
        data_path = DATA_DIR / f"{crop}.csv"
        
        if data_path.exists():
            df = pd.read_csv(data_path)
            if 'pr' in df.columns and 'tasmax' in df.columns:
                pr_data = df['pr'].tail(periods).tolist()
                tasmax_data = df['tasmax'].tail(periods).tolist()
                dates = [(datetime.now() - timedelta(days=i*30)).strftime('%Y-%m') 
                        for i in range(periods-1, -1, -1)]
                return {
                    "dates": dates,
                    "precipitation": [float(p) for p in pr_data],
                    "temperature": [float(t) for t in tasmax_data]
                }
        
        # 生成模拟数据
        np.random.seed(42)
        periods_list = [(datetime.now() - timedelta(days=i*30)).strftime('%Y-%m') 
                       for i in range(periods-1, -1, -1)]
        
        return {
            "dates": periods_list,
            "precipitation": [round(float(np.random.uniform(0.5, 2.5)), 2) for _ in range(periods)],
            "temperature": [round(float(np.random.uniform(22, 32)), 2) for _ in range(periods)]
        }
    
    except Exception as e:
        logger.error(f"获取气候数据失败：{e}")
        raise HTTPException(status_code=500, detail=f"获取气候数据失败：{str(e)}")

@app.get("/api/climate-chart")
async def get_climate_chart(crop: str = "corn", periods: int = 12):
    """获取气候数据图表（Base64 图片）"""
    try:
        # ✅ 关键修复：确保 periods 是整数
        periods = int(periods)
        
        data = await get_climate_data(crop, periods)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 降雨量图表
        ax1.bar(data['dates'], data['precipitation'], color='#3b82f6', alpha=0.7)
        ax1.set_ylabel('降雨量 (mm)', fontsize=11, color='#3b82f6')
        ax1.tick_params(axis='y', labelcolor='#3b82f6')
        ax1.set_title('降雨量变化', fontsize=12, fontweight='bold')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 温度图表
        ax2.plot(data['dates'], data['temperature'], marker='s', linewidth=2, 
                markersize=6, color='#ef4444')
        ax2.set_ylabel('最高温 (°C)', fontsize=11, color='#ef4444')
        ax2.tick_params(axis='y', labelcolor='#ef4444')
        ax2.set_title('最高温度变化', fontsize=12, fontweight='bold')
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
        logger.error(f"生成气候图表失败：{e}")
        raise HTTPException(status_code=500, detail=f"生成气候图表失败：{str(e)}")

# ==================== 📊 模型解释 ====================

@app.get("/api/feature-importance")
async def get_feature_importance(crop: str = "corn"):
    """获取特征重要性（模型解释）"""
    try:
        model_path = Path(MODEL_DIR) / f"{crop}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"模型 '{crop}' 不存在")

        model_package = joblib.load(model_path)
        model = model_package.get("model")
        feature_names = model_package.get("feature_names", [])

        if not hasattr(model, 'feature_importances_'):
            raise HTTPException(status_code=400, detail="该模型不支持特征重要性分析")

        importance_dict = {}
        for name, importance in zip(feature_names, model.feature_importances_):
            importance_dict[name] = round(float(importance), 4)
        
        # 按重要性排序
        sorted_importance = dict(sorted(importance_dict.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return {"feature_importance": sorted_importance}
    
    except Exception as e:
        logger.error(f"获取特征重要性失败：{e}")
        raise HTTPException(status_code=500, detail=f"获取特征重要性失败：{str(e)}")

@app.get("/api/feature-importance-chart")
async def get_feature_importance_chart(crop: str = "corn"):
    """获取特征重要性图表（Base64 图片）"""
    try:
        data = await get_feature_importance(crop)
        
        features = list(data['feature_importance'].keys())
        importances = list(data['feature_importance'].values())
        
        # 中文特征名映射
        feature_labels = {
            'pr': '当前降雨量',
            'pr_lag1': '前一期降雨量',
            'pr_lag2': '前两期降雨量',
            'pr_std': '降雨标准差',
            'price_lag1': '前一期价格',
            'price_lag2': '前两期价格',
            'tasmax': '当前最高温',
            'tasmax_lag1': '前一期最高温',
            'tasmax_lag2': '前两期最高温',
            'tasmax_mean': '最高温平均值'
        }
        
        labels = [feature_labels.get(f, f) for f in features]
        
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
        bars = plt.barh(labels, importances, color=colors)
        
        plt.xlabel('重要性', fontsize=12)
        plt.title(f'{crop.upper()} 模型特征重要性', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # 在柱子上添加数值
        for bar, val in zip(bars, importances):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return {"chart": f"data:image/png;base64,{img_base64}"}
    
    except Exception as e:
        logger.error(f"生成特征重要性图表失败：{e}")
        raise HTTPException(status_code=500, detail=f"生成特征重要性图表失败：{str(e)}")

# ==================== 📥 批量预测 ====================

@app.post("/api/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(crop: str = "corn", file: UploadFile = File(...)):
    """批量预测（上传 CSV 文件）"""
    try:
        # 读取上传的 CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # 加载模型
        model_path = Path(MODEL_DIR) / f"{crop}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"模型 '{crop}' 不存在")
        
        model_package = joblib.load(model_path)
        model = model_package.get("model")
        feature_names = model_package.get("feature_names", [])
        
        # 验证列
        missing = set(feature_names) - set(df.columns)
        if missing:
            raise HTTPException(status_code=400, detail=f"CSV 缺少列：{sorted(missing)}")
        
        # 批量预测
        results = []
        success_count = 0
        error_count = 0
        
        X = df[feature_names]
        predictions = model.predict(X)
        
        for i, pred in enumerate(predictions):
            results.append({
                "row": i + 1,
                "predicted_price": round(float(pred), 2),
                "status": "success"
            })
            success_count += 1
        
        logger.info(f"✅ 批量预测成功 | crop={crop} | count={success_count}")
        return BatchPredictionResponse(
            results=results,
            success_count=success_count,
            error_count=error_count
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ 批量预测失败")
        raise HTTPException(status_code=500, detail=f"批量预测失败：{str(e)}")

# ==================== 健康检查 ====================

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "2.0.0"}

@app.get("/api/crops")
async def get_supported_crops():
    """获取支持的作物列表"""
    crops = [f.stem for f in Path(MODEL_DIR).glob("*.joblib")]
    return {"crops": crops}

# ==================== 启动日志 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)