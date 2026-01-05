# 改为 Python 3.11（稳定且支持 sklearn 1.8.0）
FROM python:3.11-slim

WORKDIR /app

# 安装编译依赖（sklearn 需要 gcc）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# 替换原来的 CMD 行
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
