# build_fundamentals_csv.py
# SEC EDGAR → fundamentals CSV (EPS/EBITDA/ROIC/PEG)
# ใช้: python build_fundamentals_csv.py --tickers AAPL,MSFT --out fundamentals.csv --user-agent "YourApp/1.0 [email protected]" [--prices prices.csv]

import argparse, csv, json, math, os, time, sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests
from io import StringIO
import pandas as pd
from urllib.parse import urlparse
import json
import zipfile,io

SEC_BASE = "https://data.sec.gov"
TICKER_LIST_URL = "https://www.sec.gov/files/company_tickers.json"  # ticker→CIK mapping (ทางการ)
CACHE_DIR = "sec_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

REQ_TIMEOUT = 6  # seconds
SLEEP_BETWEEN = 0.4  # ~4 req/s เพื่อความปลอดภัย (<10 req/s ตาม SEC)

USGAAP = {
    # Income statement (duration)
    "NI": ["NetIncomeLoss"],
    "OI": ["OperatingIncomeLoss"],
    "DA": ["DepreciationAndAmortization", "DepreciationDepletionAndAmortization"],

    # Balance sheet (instant)
    "CASH": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndShortTermInvestments"],
    "EQUITY": ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "StockholdersEquity"],
    # Debt components (instant) — จะรวมกันเป็น total_debt
    "DEBT_COMP": [
        "LongTermDebtNoncurrent", "LongTermDebtCurrent",
        "ShortTermBorrowings", "ShortTermDebt", "CommercialPaper",
        "DebtCurrent", "DebtNoncurrent", "LongTermDebt", "LongTermBorrowings"
    ],

    # Shares / EPS
    "EPS_DIL": ["EarningsPerShareDiluted"],
    "SH_DIL": ["WeightedAverageNumberOfDilutedSharesOutstanding",
               "WeightedAverageNumberDilutedSharesOutstanding",
               "WeightedAverageNumberOfDilutedSharesOutstandingBasicAndDiluted"],
    # Tax
    "TAX": ["IncomeTaxExpenseBenefit", "IncomeTaxExpense"]
}


WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def load_sp500_constituents(user_agent: str):
    """
    return: tickers(list[str]), wiki_cik_map({sym: CIK(10d)}), sector_map({sym: sector})
    """
    resp = requests.get(WIKI_SP500, headers={"User-Agent": user_agent}, timeout=REQ_TIMEOUT)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))

    df = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if "symbol" in cols and "cik" in cols:
            df = t; break
    if df is None:
        # โครงสร้างหน้าเปลี่ยนอาจไม่เจอคอลัมน์ cik — กลับไปใช้ตารางแรกแล้วไม่มี CIK ก็ได้
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if "symbol" in cols:
                df = t; break
    if df is None:
        raise RuntimeError("Cannot find S&P 500 table on Wikipedia")

    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper().str.replace(" ", "", regex=False)

    tickers = df["Symbol"].tolist()
    wiki_cik_map = {}
    if "CIK" in df.columns:
        for sym, cik in zip(df["Symbol"], df["CIK"]):
            if pd.notna(cik):
                try:
                    wiki_cik_map[sym] = f"{int(cik):010d}"
                except Exception:
                    pass

    sector_map = {}
    col_sector = next((c for c in df.columns if str(c).lower().startswith("gics")), None)
    if col_sector is not None:
        for sym, sec in zip(df["Symbol"], df[col_sector]):
            sector_map[sym] = str(sec)

    return tickers, wiki_cik_map, sector_map

def log(msg):
    print(msg, file=sys.stderr)

def get_companyfacts_offline(cik: str, zip_path: str = None, dir_path: str = None) -> dict:
    fname = f"CIK{cik}.json"
    if zip_path:
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(fname) as fh:
                return json.load(io.TextIOWrapper(fh, encoding="utf-8"))
    if dir_path:
        with open(os.path.join(dir_path, fname), "r", encoding="utf-8") as fh:
            return json.load(fh)
    raise FileNotFoundError(fname)

def check_submissions(cik, ua):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    return http_get(url, ua, use_cache=True)

def http_get(url, ua, params=None, use_cache=True):
    key = url.replace("/", "_").replace("?", "_") + ("_" + json.dumps(params, sort_keys=True) if params else "")
    path = os.path.join(CACHE_DIR, f"{key}.json")

    # ถ้ามีไฟล์แคชแต่เสีย (ไม่ใช่ JSON) ให้ลบทิ้งแล้วดึงใหม่
    if use_cache and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            try: os.remove(path)
            except: pass  # ลบไม่ได้ก็ไปดึงใหม่ต่อ

    netloc = urlparse(url).netloc
    headers = {
        "User-Agent": ua,                           # ต้องมีอีเมลติดต่อจริง
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Referer": f"https://{netloc}/",
        "Host": netloc,
        "Cache-Control": "no-cache",
    }

    # retry + backoff กัน rate filter
    backoff = [0.5, 1.0, 2.0, 4.0]
    last = ""
    for wait in backoff:
        r = requests.get(url, headers=headers, params=params, timeout=REQ_TIMEOUT)
        ctype = r.headers.get("Content-Type","")
        if r.status_code == 200 and "application/json" in ctype.lower():
            data = r.json()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            time.sleep(SLEEP_BETWEEN)
            return data
        last = r.text[:400]
        time.sleep(wait)

    # fallback เฉพาะไฟล์ company_tickers.json (เผื่อโหลดมาใส่เอง)
    if "company_tickers.json" in url:
        local = os.path.join(CACHE_DIR, "company_tickers.json")
        if os.path.exists(local):
            with open(local, "r", encoding="utf-8") as f:
                return json.load(f)

    raise RuntimeError(f"GET {url} -> {r.status_code} {last}")


def load_ticker_map(ua) -> Dict[str, str]:
    """return {ticker_upper: 10-digit CIK string}"""
    data = http_get(TICKER_LIST_URL, ua, use_cache=True)
    # company_tickers.json เป็น dict ที่ key เป็น running index: {"0": {...}, "1": {...}}
    mapping = {}
    for _, row in data.items():
        tkr = (row.get("ticker") or "").upper()
        cik_int = int(row.get("cik_str"))
        mapping[tkr] = f"{cik_int:010d}"
    return mapping

def get_companyfacts(cik: str, ua) -> dict:
    url = f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    return http_get(url, ua)

def _extract_facts(facts: dict, tag: str, unit="USD") -> List[dict]:
    """get fact list for us-gaap:tag with the given unit"""
    node = (facts.get("facts") or {}).get("us-gaap", {}).get(tag)
    if not node:
        return []
    units = node.get("units") or {}
    arr = units.get(unit) or []
    return [x for x in arr if isinstance(x, dict) and "val" in x and x.get("val") is not None]

def _latest_instant(facts: dict, tags: List[str], unit="USD") -> Optional[Tuple[float, str]]:
    """pick most recent 'instant' value (balance sheet like) by end date"""
    cand = []
    for tg in tags:
        arr = _extract_facts(facts, tg, unit)
        for a in arr:
            # instant facts usually have only 'end' (no 'start')
            if a.get("end") and not a.get("start"):
                cand.append((a["end"], float(a["val"])))
    if not cand:
        # fallback: allow any with end, pick latest
        for tg in tags:
            arr = _extract_facts(facts, tg, unit)
            for a in arr:
                if a.get("end"):
                    cand.append((a["end"], float(a["val"])))
    if not cand:
        return None
    cand.sort(key=lambda x: x[0])  # by end date
    end = cand[-1][0]
    val = cand[-1][1]
    return val, end

def _last_quarters_sum(facts: dict, tags: List[str], quarters=4, unit="USD") -> Optional[Tuple[float, str]]:
    """sum last N quarter values (duration facts)"""
    rows = []
    for tg in tags:
        arr = _extract_facts(facts, tg, unit)
        for a in arr:
            fp = (a.get("fp") or "").upper()  # 'Q1','Q2','Q3','FY'...
            if fp.startswith("Q") and a.get("end") and a.get("val") is not None:
                try:
                    rows.append((a["end"], float(a["val"])))
                except Exception:
                    pass
    if not rows:
        # fallback: use FY (annual) last one
        for tg in tags:
            arr = _extract_facts(facts, tg, unit)
            for a in arr:
                fp = (a.get("fp") or "").upper()
                if fp == "FY" and a.get("end"):
                    rows.append((a["end"], float(a["val"])))
        if not rows:
            return None
        rows.sort(key=lambda x: x[0])
        end = rows[-1][0]
        val = float(rows[-1][1])
        return val, end

    rows.sort(key=lambda x: x[0])
    lastN = rows[-quarters:]
    total = sum(v for _, v in lastN)
    end = lastN[-1][0]
    return total, end

def _eps_ttm(facts: dict) -> Optional[Tuple[float, str]]:
    out = _last_quarters_sum(facts, USGAAP["EPS_DIL"], 4, unit="USD/shares")
    if out:
        return out
    # fallback: NI / diluted shares (TTM approximated)
    ni = _last_quarters_sum(facts, USGAAP["NI"], 4, unit="USD")
    sh = _last_quarters_sum(facts, USGAAP["SH_DIL"], 4, unit="shares")
    if ni and sh and sh[0] > 0:
        return (ni[0] / sh[0], ni[1])
    return None

def _eps_cagr_5y(facts: dict) -> Optional[float]:
    """CAGR from FY EPS diluted (last ~5 FY)"""
    rows = []
    for tg in USGAAP["EPS_DIL"]:
        arr = _extract_facts(facts, tg, unit="USD/shares")
        for a in arr:
            if (a.get("fp") or "").upper() == "FY" and a.get("end"):
                rows.append((a["end"], float(a["val"])))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda x: x[0])
    # ใช้ 6 ปีย้อนหลังถ้ามี เพื่อประมาณ 5y CAGR
    if len(rows) >= 6:
        base, last = rows[-6][1], rows[-1][1]
        years = 5.0
    else:
        base, last = rows[0][1], rows[-1][1]
        # คำนวณปีจากวันที่
        d0 = datetime.fromisoformat(rows[0][0])
        d1 = datetime.fromisoformat(rows[-1][0])
        years = max(1.0, (d1 - d0).days / 365.25)
    if base is None or base <= 0 or last is None or last <= 0:
        return None
    try:
        return (last / base) ** (1.0 / years) - 1.0
    except Exception:
        return None

def compute_metrics_from_companyfacts(facts: dict) -> dict:
    # TTM (duration)
    oi = _last_quarters_sum(facts, USGAAP["OI"], 4, unit="USD")
    da = _last_quarters_sum(facts, USGAAP["DA"], 4, unit="USD")
    ebitda_ttm = None
    asof_dur = None
    if oi or da:
        oi_v = oi[0] if oi else 0.0
        da_v = da[0] if da else 0.0
        ebitda_ttm = oi_v + da_v
        asof_dur = (oi or da)[1]

    # instant (balance sheet)
    cash = _latest_instant(facts, USGAAP["CASH"], unit="USD")
    equity = _latest_instant(facts, USGAAP["EQUITY"], unit="USD")

    # total debt = sum of known components
    debt_total, debt_asof = 0.0, None
    for tg in USGAAP["DEBT_COMP"]:
        val = _latest_instant(facts, [tg], unit="USD")
        if val:
            debt_total += float(val[0])
            debt_asof = max(debt_asof or val[1], val[1])

    eps_ttm = _eps_ttm(facts)
    eps_cagr5 = _eps_cagr_5y(facts)

    # tax rate approx from TTM Net Income + Income Tax (duration)
    ni = _last_quarters_sum(facts, USGAAP["NI"], 4, unit="USD")
    tax = _last_quarters_sum(facts, USGAAP["TAX"], 4, unit="USD")
    tax_rate = None
    if ni and tax and ni[0] > 0:
        tr = max(0.0, min(0.5, tax[0] / (ni[0] + 1e-9)))
        tax_rate = tr

    # NOPAT ~ EBIT*(1-tax); โดยใช้ OI เป็นตัวแทน EBIT
    nopat, roic = None, None
    if ebitda_ttm is not None and da:
        ebit_ttm = ebitda_ttm - da[0]
        if tax_rate is None:
            tax_rate = 0.21  # สมมุติฐาน
        nopat = ebit_ttm * (1.0 - tax_rate)

    invested_capital = None
    if equity and (debt_total is not None) and cash:
        invested_capital = (debt_total or 0.0) + (equity[0] or 0.0) - (cash[0] or 0.0)
        if invested_capital and invested_capital > 0 and nopat is not None:
            roic = nopat / invested_capital

    # debt/ebitda
    debt_to_ebitda = None
    if ebitda_ttm and ebitda_ttm > 0:
        debt_to_ebitda = (debt_total or 0.0) / ebitda_ttm

    # asof = เลือกวันที่ล่าสุดที่มี (instant/ duration)
    dates = [d for d in [asof_dur, (cash or (None, None))[1], (equity or (None, None))[1], debt_asof, (eps_ttm or (None, None))[1]] if d]
    asof = max(dates) if dates else None

    return {
        "asof": asof,
        "eps_ttm": None if not eps_ttm else float(eps_ttm[0]),
        "ebitda_ttm": None if ebitda_ttm is None else float(ebitda_ttm),
        "total_debt": None if debt_total is None else float(debt_total),
        "equity": None if not equity else float(equity[0]),
        "cash": None if not cash else float(cash[0]),
        "invested_capital": None if invested_capital is None else float(invested_capital),
        "nopat": None if nopat is None else float(nopat),
        "roic": None if roic is None else float(roic),
        "debt_to_ebitda": None if debt_to_ebitda is None else float(debt_to_ebitda),
        "eps_cagr_5y": None if eps_cagr5 is None else float(eps_cagr5),
    }

def load_prices(prices_csv: str) -> Dict[str, Tuple[float, str]]:
    """returns {ticker_upper: (price, asof)}"""
    out = {}
    with open(prices_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = (row.get("ticker") or "").upper().strip()
            p = row.get("price")
            d = row.get("asof") or ""
            if t and p:
                try:
                    out[t] = (float(p), d)
                except Exception:
                    pass
    return out

def main():
    ap = argparse.ArgumentParser()

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--sp500", action="store_true",
                    help="ดึง universe ทั้งชุดจาก S&P 500 (Wikipedia)")
    grp.add_argument("--tickers", type=str,
                    help="comma-separated tickers หรือ path ไฟล์ .txt (หนึ่งสัญลักษณ์ต่อบรรทัด)")

    ap.add_argument("--out", type=str, default="fundamentals.csv")
    ap.add_argument("--user-agent", type=str, required=True,
                    help='เช่น "YourApp/1.0 [email protected]" (SEC บังคับให้ใส่)')
    ap.add_argument("--prices", type=str, default=None,
                    help="ออปชัน: ไฟล์ราคา ticker,price,asof สำหรับคำนวณ P/E และ PEG")
    ap.add_argument("--no-sec-map", action="store_true",
                help="Skip downloading SEC ticker map (use Wikipedia CIKs only)")
    ap.add_argument("--limit", type=int, default=0,
                help="ประมวลผลสูงสุด N ตัว (0 = ทั้งหมด)")
    ap.add_argument("--start", type=int, default=0,
                    help="เริ่มจาก index ที่ระบุ (0-based) ใช้สำหรับ resume")
    ap.add_argument("--facts-zip", type=str, default=None,
                help="ใช้ companyfacts.zip ที่ดาวน์โหลดมา (อ่านออฟไลน์)")
    ap.add_argument("--facts-dir", type=str, default=None,
                    help="ใช้โฟลเดอร์ที่แตกไฟล์ companyfacts (มีไฟล์ CIK##########.json)")
    args = ap.parse_args()

    # เตรียมรายชื่อ
    # เตรียมรายชื่อ
    if args.sp500:
        log("Loading S&P 500 symbols from Wikipedia…")
        tickers, wiki_cik_map, sector_map = load_sp500_constituents(args.user_agent)
    else:
        sector_map = {}
        wiki_cik_map = {}
        if os.path.isfile(args.tickers):
            with open(args.tickers, "r", encoding="utf-8") as f:
                tickers = [ln.strip().upper() for ln in f if ln.strip()]
        else:
            tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if not tickers:
        print("no tickers", file=sys.stderr); sys.exit(1)

    # โหลดแผนที่ Ticker→CIK
    tmap = {}
    # 2.1 ใช้แมพจาก Wikipedia ก่อน
    tmap.update({k: v for k, v in wiki_cik_map.items() if v})

    # 2.2 พยายามโหลดจาก SEC เป็นตัวเสริม (fallback)
    if not args.no_sec_map:
        try:
            log("Loading SEC ticker map…")
            tmap.update(load_ticker_map(args.user_agent))
        except Exception as e:
                log(f"[WARN] SEC ticker map failed: {e} — continue with Wikipedia CIKs only.")
    else:
        log("Skip SEC ticker map (per --no-sec-map).")

    use_offline = bool(args.facts_zip or args.facts_dir)
    prices = load_prices(args.prices) if args.prices else {}

    out_rows = []
    for i, t in enumerate(tickers, 1):
        log(f"[{i}/{len(tickers)}] {t}")
        cik = tmap.get(t) or tmap.get(t.replace(".", "-")) or tmap.get(t.replace("-", "."))
        if not cik:
            log(f"[WARN] No CIK for {t}, skip"); continue

        try:
            if use_offline:
                facts = get_companyfacts_offline(cik, args.facts_zip, args.facts_dir)
            else:
                facts = get_companyfacts(cik, args.user_agent)  # ตัวเดิม (HTTP)
            m = compute_metrics_from_companyfacts(facts)
            ...
        except Exception as e:
            log(f"[ERROR] {t}: {e}")

        try:
            facts = get_companyfacts(cik, args.user_agent)
            m = compute_metrics_from_companyfacts(facts)

            # เติมราคา (ถ้ามี) → P/E & PEG(5y)
            pe, peg = None, None
            px_asof = None
            if t in prices and m.get("eps_ttm"):
                price, px_asof = prices[t]
                if m["eps_ttm"] and m["eps_ttm"] != 0:
                    pe = price / m["eps_ttm"]
                if pe and m.get("eps_cagr_5y") and m["eps_cagr_5y"] and m["eps_cagr_5y"] > 0:
                    peg = pe / (m["eps_cagr_5y"] * 100.0) if m["eps_cagr_5y"] > 1 else pe / (m["eps_cagr_5y"] * 100.0)

            out_rows.append({
                "ticker": t,
                "asof": m.get("asof") or px_asof,
                "eps_ttm": m.get("eps_ttm"),
                "ebitda_ttm": m.get("ebitda_ttm"),
                "total_debt": m.get("total_debt"),
                "equity": m.get("equity"),
                "cash": m.get("cash"),
                "invested_capital": m.get("invested_capital"),
                "nopat": m.get("nopat"),
                "roic": m.get("roic"),
                "debt_to_ebitda": m.get("debt_to_ebitda"),
                "pe_ttm": pe,
                "eps_cagr_5y": m.get("eps_cagr_5y"),
                "peg_5y": peg,
                "gics_sector": sector_map.get(t),   # ← เพิ่มบรรทัดนี้ได้
                "source": "sec_edgar_companyfacts",
            })
        except Exception as e:
            log(f"[ERROR] {t}: {e}")

    # เขียน CSV
    cols = ["ticker","asof","eps_ttm","ebitda_ttm","total_debt","equity","cash",
                "invested_capital","nopat","roic","debt_to_ebitda","pe_ttm",
                "eps_cagr_5y","peg_5y","gics_sector","source"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    log(f"Done. Wrote {len(out_rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
