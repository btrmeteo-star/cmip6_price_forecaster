# ---- 1. 建置階段 ----
FROM python:3.11-slim as builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cdo build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp
RUN pip install --user --no-cache-dir -r /tmp/requirements.txt

# ---- 2. 執行階段 ----
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    cdo && rm -rf /var/lib/apt/lists/*
COPY --from=builder /root/.local /usr/local
WORKDIR /app
COPY . .
RUN chmod +x scripts/entrypoint.sh
ENTRYPOINT ["./scripts/entrypoint.sh"]
