# main.py
import os
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from loguru import logger

app = FastAPI(
    title="CMIP6 农产品价格预测 API",
    description="基于 CMIP6 气候数据与历史价格预测玉米、小麦等作物未来价格",
    version="1.0.0"
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

class PredictionRequest(BaseModel):
    data: Dict[str, float]

class PredictionResponse(BaseModel):
    predicted_price: float

@app.get("/")
async def root():
    return {"message": "CMIP6 Price Forecaster API is running!"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(crop: str = "corn", request: PredictionRequest = None):
    if not request or not request.data:
        raise HTTPException(status_code=400, detail="缺少预测数据")

    try:
        model_path = Path(MODEL_DIR) / f"{crop}.joblib"

        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"未找到 {crop} 的预测模型")

        # 加载模型包（可能是一个字典）
        model_package = joblib.load(model_path)

        # ✅ 关键修复：从字典中提取实际模型对象
        if isinstance(model_package, dict):
            model = model_package.get("model")
            feature_names = model_package.get("feature_names", [])
        else:
            model = model_package
            feature_names = []

        if model is None:
            raise ValueError("模型对象为空")

        # 准备输入数据
        input_data = request.data

        # 验证特征完整性
        missing_features = set(feature_names) - set(input_data.keys())
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"缺少必要特征: {missing_features}"
            )

        # 按训练顺序排列特征
        X = pd.DataFrame([{k: input_data[k] for k in feature_names}])

        # 执行预测
        prediction = model.predict(X)[0]

        logger.success(f"预测成功! {crop} 价格: {prediction:.2f}")
        return PredictionResponse(predicted_price=float(prediction))

    except Exception as e:
        logger.exception("预测过程中发生未知错误")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")
