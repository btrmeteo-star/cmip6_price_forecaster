#!/usr/bin/env bash
set -e
# 1. 初始化 DVC
if [ ! -d .dvc ]; then
  echo "🔧 初始化 DVC 仓库"
  dvc init --no-scm
fi
# 2. 数据存在才训练
if [ ! -f "$MODEL_PATH" ] && [ -f "/app/data/cmip6.nc" ]; then
  echo "🚀 未检测到模型，执行首次训练 ..."
  dvc repro train
fi
# 3. 训练失败也继续（容错）
if [ ! -f "$MODEL_PATH" ]; then
  echo "⚠️  训练失败或无数据，启动空模型"
  python -c "import joblib, sklearn.linear_model as lm; joblib.dump(lm.LinearRegression(), '/app/models/xgb.pkl')"
fi
# 4. 启动 FastAPI
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
