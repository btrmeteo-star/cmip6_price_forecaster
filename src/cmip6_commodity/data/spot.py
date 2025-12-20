"""
現貨資料抓取範例
"""
import sys
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parents[3] / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("[spot] 開始抓取現貨報價…")
    # 這裡先放「假資料」讓流程跑通，之後再換成真 API
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=30),
        "price": 100 + pd.Series(range(30)) * 0.5
    })
    out_file = DATA_DIR / "spot_price.csv"
    df.to_csv(out_file, index=False)
    print(f"[spot] 已寫入 {out_file.resolve()}")

if __name__ == "__main__":
    main()
