import joblib, pandas as pd, os
from fastapi import APIRouter
from pydantic import BaseModel

MODEL = joblib.load(os.getenv("MODEL_PATH", "models/xgb.pkl"))
router = APIRouter()

class Req(BaseModel):
    crop: str = "soybean"
    lead_time: int = 3

@router.post("/forecasts")
def predict(req: Req):
    # 即時補一期特徵（demo 用隨機）
    X = pd.DataFrame([{"lead_time": req.lead_time, "crop": req.crop}])
    price = float(MODEL.predict(X)[0])
    return {"crop": req.crop, "lead_time": req.lead_time,
            "price": price, "ci_lower": price*0.9, "ci_upper": price*1.1}
