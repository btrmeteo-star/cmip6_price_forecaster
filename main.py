from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

# 提供静态文件（确保 index.html 在项目根目录）
if os.path.exists("index.html"):
    @app.get("/")
    async def serve_frontend():
        return FileResponse("index.html")

@app.post("/predict")
async def mock_predict(crop: str = "corn", payload: dict = None):
    # 模拟预测结果（不加载真实模型）
    return {"predicted_price": 186.88}