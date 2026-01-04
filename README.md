# 🌾 CMIP6 Crop Price Forecaster

> **End-to-end MLOps pipeline** that predicts agricultural commodity prices using climate projections from CMIP6 models and historical market data.

[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20+%20Model%20Registry-blue)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/btrmeteo-star/cmip6_price_forecaster/mlflow.yml?label=Train%20&%20Log)](https://github.com/btrmeteo-star/cmip6_price_forecaster/actions)
[![License](https://img.shields.io/github/license/btrmeteo-star/cmip6_price_forecaster)](LICENSE)

---

## 🌍 Overview

This project leverages **CMIP6 climate model outputs** (e.g., temperature, precipitation) to forecast future prices of key crops like barley, wheat, and maize. The system:

- Automatically fetches CMIP6 NetCDF data
- Engineers climate-agronomic features
- Trains ML models with **MLflow experiment tracking**
- Deploys the best-performing model via a **FastAPI REST endpoint**

Ideal for climate risk assessment, agricultural planning, and food security research.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
