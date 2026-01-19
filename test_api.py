# test_api.py
import requests

url = "http://localhost:8081/predict"

data = {
    "pr": 1.2,
    "pr_lag1": 0.8,
    "pr_lag2": 1.0,
    "pr_std": 0.5,
    "price_lag1": 105.0,
    "price_lag2": 102.0,
    "tasmax": 26.5,
    "tasmax_lag1": 26.0,
    "tasmax_lag2": 25.8,
    "tasmax_mean": 26.2
}

response = requests.post(url, json=data)
print(response.json())