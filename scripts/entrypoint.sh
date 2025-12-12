#!/bin/bash
set -e
# 若掛載 /app/data 沒模型，先離線訓練一次
if [ ! -f "artifacts/ensemble.pkl" ]; then
    echo "No model found, start training..."
    python cli.py train
fi
# 正式啟動 API
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
