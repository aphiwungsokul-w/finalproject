# data.py
import pandas as pd
from functools import lru_cache

FUND_PATH = "fundamentals_sp500.csv"   # วาง .csv ไว้โฟลเดอร์เดียวกับ main.py

@lru_cache(maxsize=1)
def load_fundamentals(path: str = FUND_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ticker"] = df["ticker"].str.upper().str.strip()
    return df

def load_tickers() -> list[str]:
    return load_fundamentals()["ticker"].dropna().unique().tolist()

@lru_cache(maxsize=1)
def load_sector_map() -> dict[str, str]:
    df = load_fundamentals()
    if "gics_sector" in df.columns:
        return dict(zip(df["ticker"], df["gics_sector"].fillna("Unknown")))
    # เผื่อไม่มีคอลัมน์ sector
    return {t: "Unknown" for t in df["ticker"].unique()}
