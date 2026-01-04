# src/auto_deploy_best.py (FIXED VERSION)

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
EXPERIMENT_NAME = "cmip6-crop-price-prediction"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://0.0.0.0:5000")
MODEL = None


def load_best_model_from_mlflow():
    global MODEL
    print("🚀 启动自动部署服务...")
    print(f"   MLflow URI: {MLFLOW_TRACKING_URI}")
    print(f"   实验名称: {EXPERIMENT_NAME}")

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            raise RuntimeError(f"实验 '{EXPERIMENT_NAME}' 不存在！")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.r2 DESC"],
            max_results=1
        )

        if not runs:
            raise RuntimeError("未找到任何模型 run！")

        best_run = runs[0]
        print(f"🏆 使用 Run ID: {best_run.info.run_id} (R²={best_run.data.metrics.get('r2', 'N/A')})")

        model_uri = f"runs:/{best_run.info.run_id}/model"
        print(f"📥 加载模型: {model_uri}")
        
        # 关键：捕获加载异常
        MODEL = mlflow.sklearn.load_model(model_uri)
        print("✅ 模型加载成功！类型:", type(MODEL))

    except Exception as e:
        print(f"💥 模型加载失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # 不 exit，让 FastAPI 启动但标记为不可用
        MODEL = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_best_model_from_mlflow()
    print("🟢 FastAPI 生命周期启动完成")
    yield
    print("🔴 FastAPI 生命周期结束")


app = FastAPI(title="CMIP6 Predictor", lifespan=lifespan)


class PredictionRequest(BaseModel):
    crop: str
    features: List[float]


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "model_type": str(type(MODEL)) if MODEL else None
    }


@app.post("/predict")
async def predict(request: PredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="模型未加载，请查看启动日志")
    if len(request.features) != 8:
        raise HTTPException(status_code=400, detail="需要 8 个特征值")
    try:
        pred = MODEL.predict([request.features])[0]
        return {"predicted_price": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测错误: {str(e)}")


@app.get("/")
async def root():
    return {"message": "CMIP6 Price API is running!", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    print("🔧 启动 FastAPI 服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
