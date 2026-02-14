# main.py
import os
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from loguru import logger

# 初始化 FastAPI
app = FastAPI(
    title="CMIP6 农产品价格预测 API",
    description="基于 CMIP6 气候数据与历史价格预测玉米、小麦等作物未来价格",
    version="1.0.0"
)

# 配置路径
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Pydantic 模型定义
class PredictionRequest(BaseModel):
    data: Dict[str, float]

class PredictionResponse(BaseModel):
    predicted_price: float

@app.get("/")
async def root():
    return {"message": "CMIP6 Price Forecaster API is running!"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(crop: str = "corn", request: PredictionRequest = None):
    """
    预测指定作物价格
    
    Args:
        crop: 作物名称 (e.g., 'corn', 'wheat')
        request: 包含10个特征的字典
        
    Returns:
        预测价格
    """
    if not request or not request.data:
        raise HTTPException(status_code=400, detail="缺少预测数据")
    
    try:
        # 构建模型路径
        model_path = Path(MODEL_DIR) / f"{crop}.joblib"
        
        # 检查模型是否存在
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            raise HTTPException(status_code=404, detail=f"未找到 {crop} 的预测模型")
        
        # 加载模型包（包含模型+元数据）
        logger.info(f"正在加载模型: {model_path}")
        model_package = joblib.load(model_path)
        
        # ✅ 关键修复：从字典中提取实际模型
        if isinstance(model_package, dict):
            model = model_package.get("model")
            feature_names = model_package.get("feature_names", [])
        else:
            # 兼容旧版直接保存的模型
            model = model_package
            feature_names = []
        
        if model is None:
            raise ValueError("模型对象为空")
        
        # 准备输入数据
        input_data = request.data
        
        # 如果有特征顺序信息，按顺序排列
        if feature_names:
            # 验证所有必需特征都存在
            missing_features = set(feature_names) - set(input_data.keys())
            if missing_features:
                raise HTTPException(
                    status_code=400,
                    detail=f"缺少必要特征: {missing_features}"
                )
            # 按训练时的顺序排列
            X = pd.DataFrame([{k: input_data[k] for k in feature_names}])
        else:
            # 无特征顺序信息时，使用任意顺序（不推荐）
            X = pd.DataFrame([input_data])
        
        # 执行预测
        logger.info(f"执行预测，输入特征: {list(X.columns)}")
        prediction = model.predict(X)[0]
        
        logger.success(f"预测成功! {crop} 价格: {prediction:.2f}")
        return PredictionResponse(predicted_price=float(prediction))
        
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {str(e)}")
        raise HTTPException(status_code=404, detail="模型文件缺失")
    except ValueError as e:
        logger.error(f"数据错误: {str(e)}")
        raise HTTPException(status_code=400, detail=f"输入数据无效: {str(e)}")
    except Exception as e:
        logger.exception("预测过程中发生未知错误")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

# 启动日志
logger.info("API 服务启动完成")
