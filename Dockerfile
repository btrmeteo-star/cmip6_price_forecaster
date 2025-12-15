# ---- build ----
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --user -r requirements-api.txt

# ---- runtime ----
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY src/ ./src
COPY models/xgb.pkl ./models/
ENV MODEL_PATH=models/xgb.pkl
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
