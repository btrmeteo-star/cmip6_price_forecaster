#!/usr/bin/env python3
"""
CMIP6 农产品价格预测 API
支持作物：corn, wheat, soybean
模型路径：models/{crop}.joblib
前端入口：index.html
"""

import os
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
from loguru import logger

# 初始化 FastAPI 应用
app = FastAPI(
    title="CMIP6 农产品价格预测 API",
    description="基于 CMIP6 气候数据与历史价格预测玉米、小麦、大豆未来价格",
    version="1.0.0"
)

# 配置路径
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


class PredictionRequest(BaseModel):
    data: Dict[str, float]


class PredictionResponse(BaseModel):
    predicted_price: float


@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """提供前端页面"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"error": "Frontend index.html not found"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(crop: str = "corn", request: PredictionRequest = None):
    """
    预测指定作物的价格
    - crop: corn / wheat / soybean
    - request.data: 包含所有必要特征的字典
    """
    if not request or not request.data:
        raise HTTPException(status_code=400, detail="请求体中缺少 'data' 字段")

    try:
        # 构建模型路径
        model_path = Path(MODEL_DIR) / f"{crop}.joblib"

        # 检查模型是否存在
        if not model_path.exists():
            available = [f.stem for f in Path(MODEL_DIR).glob("*.joblib")]
            raise HTTPException(
                status_code=404,
                detail=f"模型 '{crop}' 不存在。可用模型: {available}"
            )

        # 加载模型包
        model_package = joblib.load(model_path)
        model = model_package.get("model")
        feature_names = model_package.get("feature_names", [])

        if model is None:
            raise ValueError("模型对象为空，请检查模型保存格式")

        input_data = request.data

        # 验证输入特征完整性
        missing = set(feature_names) - set(input_data.keys())
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"缺少必要特征: {sorted(missing)}"
            )

        # ✅ 关键修复：使用 columns=feature_names 确保 XGBoost 兼容
        X = pd.DataFrame([input_data], columns=feature_names)

        # 执行预测
        prediction = model.predict(X)[0]

        logger.info(f"✅ 预测成功 | crop={crop} | price={prediction:.2f}")
        return PredictionResponse(predicted_price=float(prediction))

    except Exception as e:
        logger.exception("❌ 预测过程中发生错误")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")