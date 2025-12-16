from fastapi import FastAPI
from src.api.routers import health, predict
import os
import joblib
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型（容错）
    global MODEL
    model_path = os.getenv("MODEL_PATH", "/app/models/xgb.pkl")
    if os.path.exists(model_path):
        MODEL = joblib.load(model_path)
    else:
        # 缺失模型时注入空回归器，保证服务不崩溃
        from sklearn.linear_model import LinearRegression
        MODEL = LinearRegression()
    yield
    # 关闭时释放（如有需要）
    MODEL = None

app = FastAPI(
    title="CMIP6 大宗预报",
    version="0.1.0",
    lifespan=lifespan
)

# ① 健康检查根路由（必须返回 200）
@app.get("/", tags=["health"])
def root():
    return {
        "status": "ok",
        "service": "cmip6-commodity",
        "version": "0.1.0",
        "mlflow": "/mlflow/",
        "docs": "/docs"
    }

# ② 健康探针（可选）
@app.get("/health", tags=["health"])
def health_check():
    return {"status": "healthy"}

# ③ 业务路由
app.include_router(health.router)
app.include_router(predict.router, prefix="/v1")
