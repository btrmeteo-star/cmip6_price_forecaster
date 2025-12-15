#!/usr/bin/env bash
set -e
if [ ! -f "$MODEL_PATH" ]; then
  echo "🚀 未检测到模型，执行首次训练 ..."
  dvc repro train
fi
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
