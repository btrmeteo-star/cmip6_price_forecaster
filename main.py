# main.py
import os
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
from loguru import logger

app = FastAPI(
    title="CMIP6 农产品价格预测 API",
    description="基于真实气候与价格数据预测玉米、小麦、大豆等作物未来价格",
    version="1.0.0"
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

class PredictionRequest(BaseModel):
    data: Dict[str, float]

class PredictionResponse(BaseModel):
    predicted_price: float

@app.get("/")
async def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Frontend not found"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(crop: str = "corn", request: PredictionRequest = None):
    if not request or not request.data:
        raise HTTPException(status_code=400, detail="缺少预测数据")

    try:
        model_path = Path(MODEL_DIR) / f"{crop}.joblib"

        if not model_path.exists():
            available = [f.stem for f in Path(MODEL_DIR).glob("*.joblib")]
            raise HTTPException(
                status_code=404,
                detail=f"模型 '{crop}' 不存在。可用模型: {available}"
            )

        model_package = joblib.load(model_path)

        if isinstance(model_package, dict):
            model = model_package.get("model")
            feature_names = model_package.get("feature_names", [])
        else:
            model = model_package
            feature_names = []

        if model is None:
            raise ValueError("模型对象为空")

        input_data = request.data

        missing = set(feature_names) - set(input_data.keys())
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"缺少必要特征: {sorted(missing)}"
            )

        X = pd.DataFrame([{k: input_data[k] for k in feature_names}])
        prediction = model.predict(X)[0]

        logger.info(f"✅ 预测成功 | crop={crop} | price={prediction:.2f}")
        return PredictionResponse(predicted_price=float(prediction))

    except Exception as e:
        logger.exception("❌ 预测失败")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")