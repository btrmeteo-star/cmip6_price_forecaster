from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict():
    # TODO: 接入 CMIP6 与模型
    return {"pred_price": 312.45, "ci_lower": 308.12, "ci_upper": 316.78}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
