FROM python:3.11-slim
WORKDIR /app

# 系統相依
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Python 相依
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 原始碼
COPY src/ ./src
COPY dvc.yaml pyproject.toml ./
# 若映像內找不到模型，則在啟動時自動重訓
RUN mkdir -p /app/models

# 入口腳本
COPY scripts/startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh
CMD ["/app/startup.sh"]
