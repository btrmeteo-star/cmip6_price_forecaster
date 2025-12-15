from fastapi import FastAPI
from src.api.routers import predict, health
app = FastAPI(title="CMIP6大宗預報")
app.include_router(health.router)
app.include_router(predict.router, prefix="/v1")
