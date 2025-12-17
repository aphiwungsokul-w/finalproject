# fundamental.py
from typing import List
import numpy as np, pandas as pd
import data  # ← ใช้ CSV ผ่าน data.py

# ——— เกณฑ์คุณภาพ (Damodaran/McKinsey แนวปฏิบัติง่าย ๆ) ———
def _quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ค่ามาตรฐาน/เผื่อว่าง
    pe  = df.get("pe_ttm").astype(float)
    peg = df.get("peg_5y").astype(float)
    roic = df.get("roic").astype(float)
    d_ebitda = df.get("debt_to_ebitda").astype(float)
    ebitda = df.get("ebitda_ttm").astype(float)
    eps_cagr = df.get("eps_cagr_5y").astype(float)

    ok = (
        (roic.fillna(0) >= 0.08) &                  # ROIC ≥ 8%
        (ebitda.fillna(0) > 0) &                    # EBITDA บวก
        ((d_ebitda.isna()) | (d_ebitda <= 3.5)) &   # Debt/EBITDA ≤ 3.5
        ((pe.isna())  | ((pe >= 5) & (pe <= 40))) & # P/E 5–40 (ถ้ามี)
        ((peg.isna()) | ((peg > 0) & (peg <= 2.0))) # PEG ≤ 2 (ถ้ามี)
    )
    df = df[ok].copy()

    # คะแนนคุณภาพ (ถ่วงน้ำหนักแบบเรียบง่าย)
    df["score"] = (
        roic.fillna(0).clip(0, 0.5) * 2.0
        + eps_cagr.fillna(0).clip(0, 0.5) * 1.5
        - d_ebitda.fillna(3).clip(0, 6) * 0.5
        - np.log1p(peg.fillna(1.5)) * 1.0
    )
    return df.sort_values("score", ascending=False)

def screen_universe(raw: List[str]) -> List[str]:
    # ใช้ universe ที่อยู่ใน CSV + ตัดให้เหลือเฉพาะสัญลักษณ์ที่อยู่ใน raw
    fund = data.load_fundamentals()
    if raw:
        fund = fund[fund["ticker"].isin([r.upper() for r in raw])]
    if fund.empty:
        return raw

    filt = _quality_filter(fund)
    return filt["ticker"].head(200).tolist()

def pass_filters(ticker: str) -> bool:
    return True

def score(ticker: str) -> float:
    return 0.0
