import os
import io
import random
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from pathlib import Path
from fpdf import FPDF

# ======================
# 配置
# ======================
MODEL_DIR = "models"
SUPPORTED_CROPS = ["corn", "wheat", "rice"]

# 特征列顺序（必须与训练一致）
FEATURE_COLUMNS = [
    "pr", "pr_lag1", "pr_lag2", "pr_std",
    "price_lag1", "price_lag2",
    "tasmax", "tasmax_lag1", "tasmax_lag2", "tasmax_mean"
]

# ======================
# Mock 模型（兜底）
# ======================
class MockModel:
    def predict(self, X: pd.DataFrame) -> List[float]:
        return [round(random.uniform(100.0, 300.0), 2) for _ in range(len(X))]

# ======================
# 模型缓存
# ======================
model_cache: Dict[str, Any] = {}

def get_model(crop: str):
    if crop in model_cache:
        return model_cache[crop]

    model_path = Path(MODEL_DIR) / f"{crop}.joblib"
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            model_cache[crop] = model
            print(f"✅ 加载模型: {model_path}")
            return model
        except Exception as e:
            print(f"⚠️ 模型加载失败 {model_path}: {e}")
    
    print(f"🔄 使用 Mock 模型: {crop}")
    model_cache[crop] = MockModel()
    return model_cache[crop]

# ======================
# FastAPI App
# ======================
app = FastAPI(
    title="CMIP6 农产品价格预测 API",
    description="基于气候与历史价格的多作物价格预测服务",
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")

# ======================
# 数据模型
# ======================
class PredictionRequest(BaseModel):
    crop: str = Field(..., description="作物名称")
    pr: float
    pr_lag1: float
    pr_lag2: float
    pr_std: float
    price_lag1: float
    price_lag2: float
    tasmax: float
    tasmax_lag1: float
    tasmax_lag2: float
    tasmax_mean: float

class PredictionResponse(BaseModel):
    crop: str
    predicted_price: float
    status: str = "success"

# ======================
# 路由
# ======================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """单预测页面（含图表）"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/batch", response_class=HTMLResponse)
async def batch_page(request: Request):
    """批量预测页面"""
    return templates.TemplateResponse("batch_prediction.html", {"request": request})

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """单样本预测"""
    if request.crop not in SUPPORTED_CROPS:
        raise HTTPException(status_code=400, detail=f"不支持的作物。支持: {SUPPORTED_CROPS}")

    try:
        model = get_model(request.crop)
        features = [[
            request.pr, request.pr_lag1, request.pr_lag2, request.pr_std,
            request.price_lag1, request.price_lag2,
            request.tasmax, request.tasmax_lag1, request.tasmax_lag2, request.tasmax_mean
        ]]
        df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        pred = model.predict(df)[0]
        return PredictionResponse(crop=request.crop, predicted_price=float(pred))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(crop: str, file: UploadFile = File(...)):
    """批量预测（接收 CSV 文件）"""
    if crop not in SUPPORTED_CROPS:
        raise HTTPException(status_code=400, detail=f"不支持的作物。支持: {SUPPORTED_CROPS}")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="仅支持 CSV 文件")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # 验证列名是否匹配
        if list(df.columns) != FEATURE_COLUMNS:
            raise HTTPException(
                status_code=400,
                detail=f"CSV 列必须严格为: {FEATURE_COLUMNS}"
            )

        model = get_model(crop)
        predictions = model.predict(df)
        results = [
            {"crop": crop, "predicted_price": float(p), "status": "success"}
            for p in predictions
        ]
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")

@app.post("/report")
async def generate_report(request: PredictionRequest):
    """生成 PDF 预测报告（兼容 fpdf2）"""
    if request.crop not in SUPPORTED_CROPS:
        raise HTTPException(status_code=400, detail=f"不支持的作物。支持: {SUPPORTED_CROPS}")

    try:
        model = get_model(request.crop)
        features = [[
            request.pr, request.pr_lag1, request.pr_lag2, request.pr_std,
            request.price_lag1, request.price_lag2,
            request.tasmax, request.tasmax_lag1, request.tasmax_lag2, request.tasmax_mean
        ]]
        df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        pred = model.predict(df)[0]

        # === 使用 fpdf2 正确生成 PDF (返回 bytes) ===
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "CMIP6 农产品价格预测报告", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"作物: {request.crop.title()}", ln=True)
        pdf.cell(0, 10, f"当前降水量 (pr): {request.pr} mm", ln=True)
        pdf.cell(0, 10, f"前1期价格: {request.price_lag1} 元", ln=True)
        pdf.cell(0, 10, f"当前最高气温: {request.tasmax} °C", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"预测价格: ¥{pred:.2f}", ln=True)

        # ✅ fpdf2 的 output() 默认返回 bytes
        pdf_bytes = pdf.output()

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=price_prediction_report.pdf"}
        )
    except Exception as e:
        print(f"❌ PDF 生成错误: {e}")  # 调试日志
        raise HTTPException(status_code=500, detail=f"PDF 生成失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "ok",
        "service": "cmip6-price-forecaster",
        "loaded_models": list(model_cache.keys())
    }
