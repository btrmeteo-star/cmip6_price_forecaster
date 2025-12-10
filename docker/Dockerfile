# 使用官方Python镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    libnetcdf-dev \
    libhdf5-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY pyproject.toml .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY notebooks/ ./notebooks/
COPY data/ ./data/
COPY config.yaml .

# 创建数据目录
RUN mkdir -p /app/data/raw /app/data/processed /app/data/outputs

# 设置环境变量
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# 暴露Jupyter端口（如果需要）
EXPOSE 8888

# 默认命令
CMD ["python", "scripts/03_train_model.py"]
