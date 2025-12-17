from __future__ import annotations

"""Main FastAPI entry‑point – Black‑Litterman + ML Robo‑Advisor (robust).

🔄 2025‑05‑20
    • volatility filter per risk_mode (previous commit)
    • robust optimiser – graceful fallback when CVXPY reports infeasible:
        1. Sharpe max → MinRisk → Equal‑weight fallback
        2. Automatically relax sector caps / soft cap isn’t needed here – equal weight always feasible
"""

import hashlib
import os, json, time
from typing import Dict, List, Tuple
import re
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import fundamental as fun
from pathlib import Path
# project modules
import bl  # Black‑Litterman helper
from bl import _make_cov_pd  # PD‑fix helper
import data
import ml
import plotting_utils as pu

# ───────────────────────── Questionnaire scoring ──────────────────────────
BANDS = [
    (0, 14, 1, "เสี่ยงต่ำ"),
    (15, 21, 2, "เสี่ยงปานกลางค่อนข้างต่ำ"),
    (22, 29, 3, "เสี่ยงปานกลางค่อนข้างสูง"),
    (30, 36, 4, "เสี่ยงสูง"),
    (37, 40, 5, "เสี่ยงสูงมาก"),
]
LAMBDA_MAP = {1: 10.0, 2: 6.0, 3: 4.0, 4: 2.5, 5: 1.2}
MAX_SECTOR_FETCH = 5  # ดึงสดไม่เกิน 5 ตัวต่อ 1 request
_SECTOR_CACHE = "sector_cache.json"



def load_sp500_constituents() -> List[Dict[str, str]]:
    """
    โหลดรายชื่อหุ้น S&P500
      1) ถ้ามี cache แล้ว ใช้จากไฟล์เลย
      2) ถ้าไม่มี cache → พยายามดึงจาก Wikipedia (ผ่าน requests + user-agent)
      3) ถ้า error (เช่น 403) → fallback ไปใช้ data.load_tickers() + data.load_sector_map()
    """
    SP500_CACHE.parent.mkdir(parents=True, exist_ok=True)

    # ใช้ cache ก่อนถ้ามี
    if SP500_CACHE.exists():
        try:
            with open(SP500_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass  # ถ้าไฟล์เสียให้ไปสร้างใหม่ด้านล่าง

    records: List[Dict[str, str]] = []

    # 1) พยายามดึงจาก Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        resp.raise_for_status()
        df_list = pd.read_html(resp.text, header=0)
        if df_list:
            df = df_list[0]
            for _, row in df.iterrows():
                sym = str(row["Symbol"]).strip().upper().replace(".", "-")
                records.append(
                    {
                        "ticker": sym,
                        "name": str(row["Security"]).strip(),
                        "sector": str(row["GICS Sector"]).strip(),
                        "sub_industry": str(row.get("GICS Sub-Industry", "")).strip(),
                    }
                )
    except Exception:
        # 2) Fallback – ใช้ universe ที่มีอยู่ในโปรเจกต์
        try:
            tickers = data.load_tickers()
            sector_map = data.load_sector_map()
            for t in tickers:
                records.append(
                    {
                        "ticker": t,
                        "name": t,  # ถ้าไม่มีชื่อบริษัทใน local data ก็ใช้ ticker ไปก่อน
                        "sector": sector_map.get(t, ""),
                        "sub_industry": "",
                    }
                )
        except Exception:
            # ถ้า fallback ก็ยังพัง ให้คืนลิสต์ว่าง (ฝั่ง frontend จะโชว์ "ไม่พบหุ้น")
            return []

    if not records:
        return []

    # เขียน cache ไว้ใช้ครั้งถัดไป
    try:
        with open(SP500_CACHE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return records


def get_sp500_ticker_set() -> set:
    data = load_sp500_constituents()
    return {x["ticker"] for x in data}

def score_questionnaire(raw: str):
    """
    รับสตริงเช่น  '1,4,3,1|3|5,...'  → คืน
    (total_score, level 1-5, lambda_ra, explanation_str)
    """
    nums = list(map(int, re.findall(r"\d+", raw)))  # ดึงทุกเลข
    pts = sum(nums)  # 10–40

    for lo, hi, lvl, desc in BANDS:
        if lo <= pts <= hi:
            expl = f"คุณได้ {pts} คะแนน → ระดับ {lvl} ({desc})"
            return pts, lvl, LAMBDA_MAP[lvl], expl

    # fallback (กรณีคะแนนนอกขอบเขต)
    return pts, 3, LAMBDA_MAP[3], "ไม่พบคะแนนแบบสอบถาม"


# ───────────────────────── Risk parameters ────────────────────────────────
SECTOR_CAP = 0.25  # 25 % per sector
RISK_PARAM = {
    # level 1 – เสี่ยงต่ำมาก
    "level1": {"soft": 0.02, "hard": 0.06, "vol": 0.20},
    # level 2 – เสี่ยงปานกลางค่อนข้างต่ำ
    "level2": {"soft": 0.04, "hard": 0.08, "vol": 0.30},
    # level 3 – เสี่ยงปานกลางค่อนข้างสูง
    "level3": {"soft": 0.06, "hard": 0.10, "vol": 0.40},
    # level 4 – เสี่ยงสูง
    "level4": {"soft": 0.08, "hard": 0.12, "vol": 0.50},
    # level 5 – เสี่ยงสูงมาก
    "level5": {"soft": 0.10, "hard": 0.15, "vol": 0.60},
}

TF_ALIAS = {"short": "3mo", "medium": "1y", "long": "5y"}
TF_CFG: Dict[str, Dict[str, float]] = {
    "3mo": {"boost": 0.90, "ann": 0.25},
    "1y": {"boost": 1.00, "ann": 1.00},
    "5y": {"boost": 1.50, "ann": 5.00},
}

# ───────────────────────── FastAPI init ───────────────────────────────────
app = FastAPI(title="Robo‑Advisor BL+ML API")
app.mount("/static", StaticFiles(directory="static"), name="static")
BASE_DIR = Path(__file__).resolve().parent
app.mount("/pic", StaticFiles(directory=str(BASE_DIR / "pic")), name="pic")
templates = Jinja2Templates("templates")
SP500_CACHE = BASE_DIR / "cache" / "sp500_constituents.json"

# ───────────────────────── Helpers ────────────────────────────────────────
def _load_sector_cache():
    if os.path.exists(_SECTOR_CACHE):
        try:
            with open(_SECTOR_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_sector_cache(cache: dict):
    try:
        with open(_SECTOR_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception:
        pass


def get_sectors(tickers: list[str]) -> dict[str, str]:
    cache = _load_sector_cache()
    out = {}
    for t in tickers:
        if t in cache and cache[t]:
            out[t] = cache[t]
            continue
        sec = "Unknown"
        try:
            info = {}
            try:
                info = yf.Ticker(t).get_info()  # new API
            except Exception:
                info = yf.Ticker(t).info  # fallback
            sec = info.get("sector") or info.get("industry") or "Unknown"
        except Exception:
            sec = "Unknown"
        cache[t] = sec
        out[t] = sec
        time.sleep(0.15)  # ลดโอกาสโดน rate-limit
    _save_sector_cache(cache)
    return out


def _user_offset(uid: str, tic: str) -> float:
    h = hashlib.md5(f"{uid}_{tic}".encode()).digest()
    return (int.from_bytes(h[:4], "little") / 2**32 - 0.5) * 0.05


def _sector_of(tic: str) -> str:
    # ไม่เรียก Yahoo ระหว่าง request – ให้ Unknown ไปก่อน
    return "Unknown"


def _sample_ef_points(
    mu: pd.Series | pd.DataFrame, cov: pd.DataFrame, n: int = 300
) -> pd.DataFrame:
    # mu อนุญาตทั้ง Series (ยาว n) หรือ DataFrame 1×n
    if isinstance(mu, pd.DataFrame):
        mu = mu.iloc[0]
    mu = pd.Series(mu).astype(float)

    cov = cov.loc[mu.index, mu.index].astype(float)
    n_assets = len(mu)
    rs, rets = [], []
    for _ in range(n):
        w = np.random.rand(n_assets)
        w /= w.sum()
        rs.append(float(np.sqrt(np.dot(w, np.dot(cov.values, w)))))
        rets.append(float(np.dot(w, mu.values)))
    return pd.DataFrame({"risk": rs, "ret": rets})


# ───────────────────────── Core route ─────────────────────────────────────


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/favorites")
def favorites_page(request: Request):
    return templates.TemplateResponse("favorites.html", {"request": request})

@app.get("/portfolio")
def portfolio(
    user_id: str = Query("guest"),
    capital: float = Query(100_000, ge=0),
    timeframe: str = Query("1y"),
    questionnaire: str = Query(""),
):
    # 1) Risk params
    risk_score, risk_level, lambda_ra, explanation = score_questionnaire(questionnaire)
    risk_key = f"level{risk_level}"
    params = RISK_PARAM[risk_key]
    soft_cap, hard_cap, max_vol = params["soft"], params["hard"], params["vol"]

    # 2) Timeframe mapping
    timeframe = TF_ALIAS.get(timeframe, timeframe)
    cfg = TF_CFG.get(timeframe, TF_CFG["1y"])

    # 3) Universe
    raw = data.load_tickers()
    tickers = fun.screen_universe(raw)

    sector_all = data.load_sector_map()
    sector_map = {t: sector_all.get(t, "Unknown") for t in tickers}
    unique_secs = {sec for sec in sector_map.values() if sec != "Unknown"}
    use_sector_caps = len(unique_secs) >= 3

    # 4) Prices & volatility filter
    price = yf.download(
        tickers, period=timeframe, progress=False, threads=True, auto_adjust=False
    )["Close"].ffill()
    if price.empty:
        return Response("{}", media_type="application/json")
    vol = price.pct_change().std() * np.sqrt(252)
    tickers = [t for t in tickers if vol[t] <= max_vol]
    if len(tickers) < 5:
        tickers = vol.sort_values().index[:20].tolist()  # fallback smallest vol
    price = price[tickers]
    sector_map = {t: sector_map[t] for t in tickers}

    # 5) Market returns
    market_mu = {}
    for t in tickers:
        s = price[t].dropna()
        if len(s) < 2:
            continue
        r_tot = s.iloc[-1] / s.iloc[0] - 1
        market_mu[t] = (1 + r_tot) ** (1 / cfg["ann"]) - 1
    if not market_mu:
        return Response("{}", media_type="application/json")

    # 6) ML views
    ml_tf = "1y" if timeframe == "3mo" else timeframe
    ticker_list = list(market_mu.keys())

    use_ml = ml.models_ready(ml_tf)
    if use_ml:
        mu_map, sig_map = ml.predict_views(ticker_list, timeframe=ml_tf, risk_score=risk_score)
    else:
        mu_map, sig_map = {}, {}
    views = {}
    for t in ticker_list:
        v = mu_map.get(t, 0.0)
        u = sig_map.get(t, 0.50)
        views[t] = (v * cfg["boost"] + _user_offset(user_id, t), u)

    # 7) Build BL portfolio
    weights, port = bl.optimize_bl(
        market_mu=market_mu,
        views=views,
        risk_aversion=lambda_ra,
        period=timeframe,
        tau=0.05,
        prices=price,
        rf=0.02,
        obj="Sharpe",
        rm="MV",
        bounds=(0, 1),
    )

    # ---------------- sector inequality caps ------------------
    # ดึงชื่อสินทรัพย์จาก DataFrame / Series ใด ๆ
    assets = (
        list(port.mu.columns)
        if isinstance(port.mu, pd.DataFrame)
        else list(port.mu.index)
    )
    n = len(assets)

    # 8) Optimisation attempts
    def _optim(obj: str):
        try:
            return port.optimization(model="Classic", rm="CVaR", obj=obj, rf=0.02, l=lambda_ra)
        except Exception:
            return None

    # อย่าใช้ "or" กับ DataFrame (pandas ห้าม truth-test)
    w_df = _optim("Sharpe")
    if w_df is None or getattr(w_df, "empty", True):
        w_df = _optim("MinRisk")


    # ----- 10) Prepare response (ต่อ) -----
    top = weights.copy()
    top = pd.Series(top).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    top = top[top > 1e-6]
    top = top.sort_values(ascending=False)
    top = top / top.sum()  # normalize ให้รวมเป็น 1
    assets = list(top.index)


    # รวบรวมราคาย้อนหลังเฉพาะสินทรัพย์ที่อยู่ในพอร์ต
    px = price[assets].dropna(how="all")

    # 10.1 มูลค่าพอร์ตเทียบ S&P500 (เริ่มต้นทุน = capital แบ่งตาม weight)
    aum = (px.pct_change().fillna(0).add(1).cumprod())
    aum_port = (aum * top).sum(axis=1)
    aum_port = capital * aum_port / aum_port.iloc[0]

    sp500 = yf.download("^GSPC", period=timeframe, progress=False)["Close"].ffill()  # ใช้ S&P500
    sp500 = sp500.loc[aum_port.index]
    sp500_aum = capital * (sp500 / sp500.iloc[0])

    # 10.2 ผลตอบแทนสะสม (%)
    port_ret = (aum_port / aum_port.iloc[0] - 1.0) * 100.0
    sp500_ret = (sp500_aum / sp500_aum.iloc[0] - 1.0) * 100.0

    # 10.3 หุ้นรายตัวผลตอบแทนสะสม (%)
    indiv = {}
    for t in assets:
        s = px[t].dropna()
        if s.empty: 
            continue
        indiv[t] = list(((s / s.iloc[0] - 1.0) * 100.0).round(4))

    # 10.4 การ์ดสรุป
    initial_val = capital
    ending_val = float(aum_port.iloc[-1])
    sp500_ending = float(sp500_aum.iloc[-1])
    alpha = (ending_val / initial_val - sp500_ending / initial_val) * 100.0

    # drawdown (สูงสุด)
    roll_max = aum_port.cummax()
    drawdown = float(((aum_port / roll_max - 1.0).min()) * 100.0)

    # best / worst (จาก % รายตัวล่าสุด)
    latest_pct = {t: v[-1] if v else 0.0 for t, v in indiv.items()}
    best_tic = max(latest_pct, key=latest_pct.get)
    worst_tic = min(latest_pct, key=latest_pct.get)

    # 10.5 ตาราง holdings + ราคาปิดล่าสุด
    table = []

    # ดาวน์โหลดราคาครั้งเดียวพอ
    tick_info = yf.download(assets, period="5d", progress=False)["Close"].ffill()

    def _domain_from_url(url: str | None):
        if not url:
            return None
        u = url.strip().lower()
        u = re.sub(r"^https?://", "", u)
        u = re.sub(r"^www\.", "", u)
        return u.split("/")[0]

    for t in assets:
        # ราคาวันล่าสุด + % เปลี่ยนแปลง
        s = tick_info[t].dropna() if t in tick_info else pd.Series(dtype=float)
        last = float(s.iloc[-1]) if not s.empty else None
        prev = float(s.iloc[-2]) if len(s) > 1 else last
        chg = None if (last is None or prev is None or prev == 0) else (last/prev - 1.0) * 100.0

        # ข้อมูลบริษัท/ตลาด/โลโก้
        meta = {}
        try:
            tk = yf.Ticker(t)
            try:
                meta = tk.get_info() or {}
            except Exception:
                meta = tk.info or {}
        except Exception:
            meta = {}

        cname = meta.get("shortName") or meta.get("longName") or t
        exch  = (meta.get("fullExchangeName") or meta.get("exchange") or "-")
        logo  = meta.get("logo_url")
        if not logo:
            dom = _domain_from_url(meta.get("website"))
            if dom:
                logo = f"https://logo.clearbit.com/{dom}?size=64"

        table.append({
            "ticker": t,
            "company": cname,
            "exchange": exch,
            "weight": float(top[t]) * 100.0,
            "last_close": last,
            "pct_change": chg,
            "logo": logo,
        })

    # แพ็ก Highcharts options ที่มีอยู่แล้ว
    weights_cfg = pu.weights_bar_hc(top.to_dict(), "Portfolio Weights")
    cov_cfg = pu.cov_heatmap_hc(port.cov.values, list(price.columns))
    ef_cfg = pu.efficient_frontier_hc(_sample_ef_points(port.mu, port.cov))

    # 10.6 ส่ง response
    return JSONResponse({
        "risk_explantion": explanation,   # <— อธิบายคะแนนความเสี่ยง (สตริง)
        "risk_score": int(risk_score),    # <— คะแนนรวม
        "risk_level": int(risk_level),
        "summary": {
            "initial": initial_val,
            "ending": ending_val,
            "sp500_initial": capital,
            "sp500_ending": sp500_ending,
            "alpha_pct": round(alpha, 2),
            "drawdown_pct": round(drawdown, 2),
            "best": {"ticker": best_tic, "perf_pct": round(latest_pct[best_tic], 2)},
            "worst": {"ticker": worst_tic, "perf_pct": round(latest_pct[worst_tic], 2)}
        },
        "series": {
            "aum_dates": [d.strftime("%Y-%m-%d") for d in aum_port.index],
            "aum_port": [float(v) for v in aum_port.values],
            "aum_sp500":  [float(v) for v in sp500_aum.values],
            "ret_dates": [d.strftime("%Y-%m-%d") for d in port_ret.index],
            "ret_port": [float(v) for v in port_ret.values],
            "ret_sp500":  [float(v) for v in sp500_ret.values],
            "indiv": indiv  # {ticker: [%, %, ...]} บน same index ของ indiv_dates
        },
        "indiv_dates": [d.strftime("%Y-%m-%d") for d in px.dropna(how="all").index],
        "weights_hc": weights_cfg,
        "holdings": table
    })

@app.get("/stock")
def stock_page(request: Request, view: str = Query("all")):
    tab = "favorites" if view == "favorites" else "all"
    return templates.TemplateResponse(
        "stock.html",
        {"request": request, "tab": tab},
    )


@app.get("/api/sp500")
def api_sp500_universe():
    data = load_sp500_constituents()
    return JSONResponse(data)


@app.get("/api/stock/{ticker}")
def api_stock_detail(
    ticker: str,
    timeframe: str = Query("5y"),
):
    t = ticker.upper()
    allowed = get_sp500_ticker_set()
    if t not in allowed:
        raise HTTPException(status_code=404, detail="Ticker is not in S&P500")

    tf_map = {
        "7d": ("1mo", "1d"),
        "1m": ("1mo", "1d"),
        "3m": ("3mo", "1d"),
        "6m": ("6mo", "1d"),
        "1y": ("1y", "1wk"),
        "3y": ("3y", "1wk"),
        "5y": ("5y", "1wk"),
    }
    period, interval = tf_map.get(timeframe, ("5y", "1wk"))

    df = yf.download(t, period=period, interval=interval, progress=False)
    if df.empty:
        raise HTTPException(status_code=404, detail="No price data")

    close = df["Close"].ffill().dropna()

    # แปลง index ให้เป็น datetime เสมอ กันเคสที่เป็น string
    idx = pd.to_datetime(close.index)
    values = close.values

    chart: List[List[float]] = []
    for ts, v in zip(idx, values):
        try:
            ts_ms = int(pd.Timestamp(ts).timestamp() * 1000)
            chart.append([ts_ms, float(v)])
        except Exception:
            continue

    if not chart:
        raise HTTPException(status_code=404, detail="No chart data")

    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) >= 2 else None

    tk = yf.Ticker(t)
    info = {}
    try:
        info = tk.get_info() or {}
    except Exception:
        try:
            info = tk.info or {}
        except Exception:
            info = {}

    name = info.get("shortName") or info.get("longName") or t
    exch = info.get("fullExchangeName") or info.get("exchange") or "-"
    currency = info.get("currency") or "USD"
    sector = info.get("sector") or ""

    payload: Dict[str, object] = {
        "ticker": t,
        "name": name,
        "exchange": exch,
        "currency": currency,
        "sector": sector,
        "last": last,
        "prev_close": prev,
        "chart": chart,
    }

    if prev:
        change = last - prev
        payload["change"] = change
        payload["change_pct"] = change / prev * 100.0

    return JSONResponse(payload)

# ── Local dev run ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", port=8000)
