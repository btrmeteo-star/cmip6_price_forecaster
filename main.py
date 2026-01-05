import os
import joblib
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

# ==============================
# 配置
# ==============================

MODEL_PATH = "models/best_model.joblib"
FEATURES = [
    'pr', 'pr_lag1', 'pr_lag2', 'pr_std',
    'price_lag1', 'price_lag2',
    'tasmax', 'tasmax_lag1', 'tasmax_lag2', 'tasmax_mean'
]

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# 加载模型
# ==============================

if not os.path.exists(MODEL_PATH):
    logger.error(f"❌ 模型文件未找到: {MODEL_PATH}")
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    logger.info("✅ 模型已加载: %s", MODEL_PATH)
    logger.info("🔍 特征列: %s", FEATURES)
except Exception as e:
    logger.error("❌ 模型加载失败: %s", str(e))
    raise RuntimeError("Failed to load model") from e

# ==============================
# Pydantic 模型
# ==============================

class PredictionRequest(BaseModel):
    pr: float = Field(..., description="当前降水量 (mm)")
    pr_lag1: float = Field(..., description="前1期降水量")
    pr_lag2: float = Field(..., description="前2期降水量")
    pr_std: float = Field(..., description="降水量标准差")
    price_lag1: float = Field(..., description="前1期价格")
    price_lag2: float = Field(..., description="前2期价格")
    tasmax: float = Field(..., description="当前最高气温 (°C)")
    tasmax_lag1: float = Field(..., description="前1期最高气温")
    tasmax_lag2: float = Field(..., description="前2期最高气温")
    tasmax_mean: float = Field(..., description="最高气温均值")

class PredictionResponse(BaseModel):
    crop: str = Field("generic_crop", description="农产品名称")
    predicted_price: float = Field(..., description="预测价格")
    status: str = Field("success", description="请求状态")

class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = True
    feature_count: int = len(FEATURES)

# ==============================
# FastAPI App
# ==============================

app = FastAPI(
    title="CMIP6 Price Forecaster",
    description="基于 CMIP6 气候数据和历史价格预测农产品价格",
    version="1.0.0",
    contact={
        "name": "Your Team",
        "email": "team@example.com"
    }
)

# ==============================
# 路由
# ==============================

@app.get("/health", response_model=HealthResponse, tags=["健康检查"])
def health_check():
    """服务健康检查"""
    return HealthResponse()

@app.post("/predict", response_model=PredictionResponse, tags=["预测"])
def predict(request: PredictionRequest):
    """
    根据气候与价格特征预测未来价格
    """
    try:
        # 构造特征向量（顺序必须与训练一致！）
        features = np.array([[
            request.pr,
            request.pr_lag1,
            request.pr_lag2,
            request.pr_std,
            request.price_lag1,
            request.price_lag2,
            request.tasmax,
            request.tasmax_lag1,
            request.tasmax_lag2,
            request.tasmax_mean
        ]], dtype=np.float32)

        # 模型预测
        prediction = model.predict(features)[0]

        # 确保是 float（避免 numpy 类型问题）
        predicted_price = float(prediction)

        logger.info("📈 预测成功: %.2f", predicted_price)

        return PredictionResponse(
            crop="corn",  # 可根据需求改为动态作物名
            predicted_price=predicted_price,
            status="success"
        )

    except Exception as e:
        logger.error("❌ 预测失败: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

# ==============================
# 可选：仪表盘（可扩展）
# ==============================

@app.get("/dashboard", tags=["监控"])
def dashboard():
    """简单状态页面（可返回指标或重定向到 Grafana）"""
    return {
        "message": "Dashboard placeholder. Consider integrating with monitoring tools.",
        "uptime": "available"
    }
