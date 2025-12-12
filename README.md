# CMIP6-Price-Forecaster

## Quick Start
```bash
git clone https://github.com/yourname/cmip6_price_forecaster
cd cmip6_price_forecaster
# 本地開發
docker-compose up
# 呼叫 API
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"cmip6_nc":"data/test.nc","commodity":"maize","horizon_month":6}'
