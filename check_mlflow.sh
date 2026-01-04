#!/usr/bin/env bash
set -e
PORT=${1:-8000}
HOST=${2:-localhost}
echo "⏳ 探测 MLflow UI http://$HOST:$PORT ..."
for i in {1..30}; do
  if curl -f "http://$HOST:$PORT" > /dev/null 2>&1; then
    echo "✅ MLflow UI 已就绪"
    exit 0
  fi
  echo "⏳ 第 $i/30 秒重试..."
  sleep 1
done
echo "❌ 无法连接，查看日志："
cat mlflow.log
exit 1
