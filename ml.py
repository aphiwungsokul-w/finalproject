""" 
ml.py  –  model-zoo (XGB + LSTM, ทั้ง regression และ classification) สำหรับ Black–Litterman views
-----------------------------------------------------------------------------------
• ใช้ FORECAST_HORIZON จาก config เป็น target (ผลตอบแทนล่วงหน้า H วัน)
• แยก train / test แบบ time-series 80/20
• Regression: ทำนายค่าผลตอบแทนล่วงหน้า (ใช้ใน predict_view)
• Classification: ทำนายทิศทาง (up/down) + prob (ใช้ใน predict_view_cls)
""" 

from __future__ import annotations
from data import load_tickers, load_fundamentals
import fundamental as fun
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from joblib import dump, load
from typing import Dict, List

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    roc_auc_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor, XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

from config import LOOKBACK, TRAIN_SUBSET, TIMEFRAMES, FORECAST_HORIZON
from indicators import sma, rsi, macd
from yfinance.exceptions import YFRateLimitError

BASE_DIR = Path(__file__).parent
CACHE = BASE_DIR / "models"
CACHE.mkdir(parents=True, exist_ok=True)

MAX_SCORE = 40

FUND = load_fundamentals()
FUND["ticker"] = FUND["ticker"].str.upper().str.strip()
FUND = FUND.set_index("ticker")


_SECTOR_NAME_COL = None
for _c in ("gics_sector", "sector", "gicsSector"):
    if _c in FUND.columns:
        _SECTOR_NAME_COL = _c
        break

if "sector_code" in FUND.columns:
    _SECTOR_CODE_MAP = None
elif _SECTOR_NAME_COL:
    _SECTOR_CODE_MAP = {
        s: i for i, s in enumerate(sorted(FUND[_SECTOR_NAME_COL].dropna().unique().tolist()))
    }
else:
    _SECTOR_CODE_MAP = {}

def _fund_row(ticker: str) -> dict:
    t = (ticker or "").upper().strip()
    if not t or t not in FUND.index:
        return {}
    r = FUND.loc[t]
    if isinstance(r, pd.DataFrame):
        r = r.iloc[0]
    if isinstance(r, pd.Series):
        return r.to_dict()
    return {}

def _safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        x = float(v)
        if np.isfinite(x):
            return x
        return default
    except Exception:
        return default

def _sector_code_from_row(row: dict) -> float:
    if not row:
        return 0.0
    if "sector_code" in row:
        return _safe_float(row.get("sector_code"), 0.0)
    if _SECTOR_NAME_COL and _SECTOR_CODE_MAP is not None:
        sec = row.get(_SECTOR_NAME_COL)
        return float(_SECTOR_CODE_MAP.get(sec, 0))
    return 0.0

def _expected_feature_names(model) -> list[str] | None:
    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        return list(cols)
    try:
        cols = model.get_booster().feature_names
        if cols:
            return list(cols)
    except Exception:
        pass
    return None

def _align_X_to_expected(X: pd.DataFrame, expected: list[str] | None) -> pd.DataFrame:
    if not expected:
        return X
    X2 = X.copy()
    for c in expected:
        if c not in X2.columns:
            X2[c] = 0.0
    X2 = X2[expected]
    return X2

def models_ready(timeframe: str) -> bool:
    """เช็คว่าโมเดลพื้นฐาน (XGB + LSTM regression) สำหรับ timeframe นี้ มีอยู่ใน cache หรือยัง"""
    return all((CACHE / f"{n}_{timeframe}.joblib").exists() for n in ("xgb", "lstm"))


def _features(
    s: pd.Series,
    risk_norm: float = 0.0,
    for_train: bool = True,
    ticker: str | None = None,
) -> pd.DataFrame:
    """สร้าง feature จากราคาปิดของหุ้นตัวเดียว

    หมายเหตุ: โมเดลบางชุดถูกเทรนรวม fundamental feature ไว้ด้วย
    (roic / debt_to_ebitda / eps_cagr_5y / sector_code) จึงต้องสร้างคอลัมน์ให้ครบ
    """

    mom = s.pct_change(LOOKBACK)  # momentum ระยะ LOOKBACK วัน
    vol = s.pct_change().rolling(LOOKBACK).std() * np.sqrt(252)  # annualised vol

    feat = pd.concat(
        [
            mom,
            vol,
            sma(s, 14) / s - 1,      # distance จาก 14-day SMA
            rsi(s, 14) / 100.0,      # scale 0–1
            macd(s) / s,             # MACD as fraction of price
        ],
        axis=1,
    )
    feat.columns = ["mom", "vol", "sma", "rsi", "macd"]
    feat["risk"] = float(risk_norm)

    row = _fund_row(ticker) if ticker else {}
    feat["roic"] = _safe_float(row.get("roic"), 0.0)
    feat["debt_to_ebitda"] = _safe_float(row.get("debt_to_ebitda"), 0.0)
    feat["eps_cagr_5y"] = _safe_float(row.get("eps_cagr_5y"), 0.0)
    feat["sector_code"] = _sector_code_from_row(row)

    feat = feat.replace([np.inf, -np.inf], np.nan)

    if for_train:
        fwd_ret = s.pct_change(FORECAST_HORIZON).shift(-FORECAST_HORIZON)
        feat["y"] = fwd_ret
        feat = feat.dropna(subset=[*feat.columns])
    else:
        feat = feat.dropna()

    return feat


def _get_prices(tic: str, period: str = "5y") -> pd.Series:
    """ดาวน์โหลดราคาปิดจาก yfinance – ถ้า error คืน series ว่าง"""
    try:
        df = yf.download(tic, period=period, progress=False, auto_adjust=False)
    except YFRateLimitError:
        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)

    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)

    clos = df["Close"]
    if isinstance(clos, pd.DataFrame):
        clos = clos.iloc[:, 0]

    return clos.dropna()


def _train_models_reg(X: pd.DataFrame, y: pd.Series) -> Dict[str, object]:
    """เทรน XGB + LSTM (regression) พร้อม scaler สำหรับ LSTM"""

    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    X_lstm = np.expand_dims(X_scaled, 1)  # (n_samples, 1, n_features)

    lstm = Sequential(
        [
            Input((1, X.shape[1])),
            LSTM(32, activation="tanh"),
            Dense(1),  # regression head
        ]
    )
    lstm.compile(optimizer="adam", loss="mse")
    lstm.fit(X_lstm, y.values, epochs=40, batch_size=64, verbose=0)

    return {"xgb": xgb, "lstm": lstm, "scaler": scaler}


def _train_models_cls(X: pd.DataFrame, y: pd.Series) -> Dict[str, object]:
    """เทรน XGB + LSTM (classification: ทำนายขึ้น/ลง)"""

    y_cls = (y.values > 0).astype(int)

    xgb_cls = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        eval_metric="logloss",
    )
    xgb_cls.fit(X, y_cls)

    scaler_cls = StandardScaler()
    X_scaled = scaler_cls.fit_transform(X.values)
    X_lstm = np.expand_dims(X_scaled, 1)

    lstm_cls = Sequential(
        [
            Input((1, X.shape[1])),
            LSTM(32, activation="tanh"),
            Dense(1, activation="sigmoid"),  # binary prob
        ]
    )
    lstm_cls.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    lstm_cls.fit(X_lstm, y_cls, epochs=40, batch_size=64, verbose=0)

    return {"xgb_cls": xgb_cls, "lstm_cls": lstm_cls, "scaler_cls": scaler_cls}

raw = load_tickers()
tickers = fun.screen_universe(raw)
def train_for_timeframe(tickers: List[str], timeframe: str, with_classification: bool = True):
    """เทรนโมเดลสำหรับ period ที่กำหนด (เช่น '3mo', '6mo', '1y')

    • รวมข้อมูลจากหลาย ticker เข้าด้วยกันแบบ panel แล้ว sort ตามเวลา
    • แบ่ง train/test ตามเวลา (80/20)
    • ประเมิน baseline + regression models + (option) classification models
    • เซฟโมเดลลง CACHE
    """
    X_parts: List[pd.DataFrame] = []
    y_parts: List[pd.Series] = []

    for t in tickers[:TRAIN_SUBSET]:
        prices = _get_prices(t, period="5y")
        if prices.empty:
            continue

        df = _features(prices, risk_norm=0.0, for_train=True, ticker=t)
        if df.empty:
            continue

        X_parts.append(df.drop(columns=["y"]))
        y_parts.append(df["y"])

    if not X_parts:
        raise RuntimeError("No training data collected")

    X = pd.concat(X_parts, axis=0)
    y = pd.concat(y_parts, axis=0)

    X = X.sort_index()
    y = y.sort_index()

    split = int(len(X) * 0.8)
    if split <= 0 or split >= len(X):
        X_train, y_train = X, y
        X_test, y_test = X.iloc[0:0], y.iloc[0:0]
    else:
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

    models_reg = _train_models_reg(X_train, y_train)
    models_cls: Dict[str, object] = {}
    if with_classification:
        models_cls = _train_models_cls(X_train, y_train)

    models = {**models_reg, **models_cls}

    if len(X_test) > 0:
        y_pred0 = np.zeros_like(y_test.values)
        mae0 = mean_absolute_error(y_test, y_pred0)
        rmse0 = float(np.sqrt(mean_squared_error(y_test, y_pred0)))
        print(f"[BASE][{timeframe}] zero: MAE={mae0:.6f} RMSE={rmse0:.6f}")

        y_pred_mean = np.full_like(y_test.values, y_train.mean())
        mae_mean = mean_absolute_error(y_test, y_pred_mean)
        rmse_mean = float(np.sqrt(mean_squared_error(y_test, y_pred_mean)))
        print(f"[BASE][{timeframe}] mean: MAE={mae_mean:.6f} RMSE={rmse_mean:.6f}")
        
        scaler_reg = models_reg.get("scaler")
        if scaler_reg is not None:
            X_test_scaled = scaler_reg.transform(X_test.values)
            X_test_lstm = np.expand_dims(X_test_scaled, 1)
        else:
            X_test_lstm = np.expand_dims(X_test.values, 1)

        for name, m in models_reg.items():
            if name == "scaler":
                continue

            if name == "xgb":
                y_pred = m.predict(X_test)
            elif name == "lstm":
                y_pred = m.predict(X_test_lstm)[:, 0]
            else:
                continue

            mae = mean_absolute_error(y_test, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            y_true_cls = (y_test.values > 0).astype(int)
            y_pred_cls = (y_pred > 0).astype(int)
            dir_acc = float((y_true_cls == y_pred_cls).mean())

            print(
                f"[ML-REG][{timeframe}] {name}: "
                f"MAE={mae:.6f} RMSE={rmse:.6f} DirAcc={dir_acc:.3f}"
            )

        if with_classification and models_cls:
            y_test_cls = (y_test.values > 0).astype(int)
            y_train_cls = (y_train.values > 0).astype(int)

            majority = 1 if y_train_cls.mean() >= 0.5 else 0
            y_pred_base_cls = np.full_like(y_test_cls, majority)
            acc_base = accuracy_score(y_test_cls, y_pred_base_cls)
            print(f"[BASE_CLS][{timeframe}] majority: Acc={acc_base:.3f}")

            xgb_cls = models_cls.get("xgb_cls")
            if xgb_cls is not None:
                prob = xgb_cls.predict_proba(X_test)[:, 1]
                pred = (prob >= 0.5).astype(int)
                acc = accuracy_score(y_test_cls, pred)
                try:
                    auc = roc_auc_score(y_test_cls, prob)
                except ValueError:
                    auc = float("nan")
                print(
                    f"[ML-CLS][{timeframe}] xgb_cls: "
                    f"Acc={acc:.3f} AUC={auc:.3f}"
                )

            lstm_cls = models_cls.get("lstm_cls")
            scaler_cls = models_cls.get("scaler_cls")
            if lstm_cls is not None and scaler_cls is not None:
                X_test_scaled_cls = scaler_cls.transform(X_test.values)
                X_lstm_test = np.expand_dims(X_test_scaled_cls, 1)
                prob_lstm = lstm_cls.predict(X_lstm_test).ravel()
                pred_lstm = (prob_lstm >= 0.5).astype(int)
                acc_lstm = accuracy_score(y_test_cls, pred_lstm)
                try:
                    auc_lstm = roc_auc_score(y_test_cls, prob_lstm)
                except ValueError:
                    auc_lstm = float("nan")
                print(
                    f"[ML-CLS][{timeframe}] lstm_cls: "
                    f"Acc={acc_lstm:.3f} AUC={auc_lstm:.3f}"
                )

    for name, m in models.items():
        dump(m, CACHE / f"{name}_{timeframe}.joblib")

    return models


def _load(name: str, tf: str):
    return load(CACHE / f"{name}_{tf}.joblib")


def predict_view(tic: str, timeframe: str = "1y", risk_score: float = 0.0):
    """คืนค่า (mu, sigma) เป็น view สำหรับ Black–Litterman (จาก regression)

    mu   = ค่าเฉลี่ยของ prediction จาก XGB + LSTM (regression)
    sigma = ส่วนเบี่ยงเบนมาตรฐานของ prediction (ใช้แทนความไม่แน่นอนของ view)
    """
    prices = _get_prices(tic, timeframe)
    if len(prices) < LOOKBACK + FORECAST_HORIZON + 2:
        return np.nan, np.nan

    risk_norm = float(risk_score) / MAX_SCORE
    X_all = _features(prices, risk_norm=risk_norm, for_train=False, ticker=tic)
    if X_all.empty:
        return np.nan, np.nan

    X_last = X_all.iloc[[-1]]

    xgb_model = _load("xgb", timeframe)
    xgb_cols = _expected_feature_names(xgb_model)
    X_xgb = _align_X_to_expected(X_last, xgb_cols)
    xgb_pred = float(xgb_model.predict(X_xgb)[0])

    try:
        scaler = _load("scaler", timeframe)
        s_cols = _expected_feature_names(scaler)
        X_s = _align_X_to_expected(X_last, s_cols) if s_cols else X_xgb
        X_scaled = scaler.transform(X_s.values)
    except Exception:
        X_scaled = X_xgb.values

    X_lstm = np.expand_dims(X_scaled, 1)
    lstm_model = _load("lstm", timeframe)
    lstm_pred = float(lstm_model.predict(X_lstm)[0][0])

    preds = np.array([xgb_pred, lstm_pred])
    return float(preds.mean()), float(preds.std())


def predict_views(tickers: List[str], timeframe: str = "1y", risk_score: int = 0):
    """คำนวณ views (regression) สำหรับหลาย ๆ ticker พร้อมกัน"""
    mu: Dict[str, float] = {}
    sig: Dict[str, float] = {}

    for t in tickers:
        m, s = predict_view(t, timeframe=timeframe, risk_score=risk_score)
        if np.isfinite(m) and np.isfinite(s):
            mu[t] = float(m)
            sig[t] = float(s)

    return mu, sig


def predict_view_cls(
    tic: str,
    timeframe: str = "1y",
    risk_score: float = 0.0,
    up_ret: float = 0.03,
    down_ret: float = -0.03,
):
    """
    ใช้โมเดล classification (xgb_cls + lstm_cls) เพื่อสร้าง view แบบ probabilistic

    - ทำนาย P(up) จากทั้ง XGB classifier + LSTM classifier
    - หาค่าเฉลี่ยของ probability เหล่านี้
    - แปลงเป็น expected return:
        mu = p_up * up_ret + (1 - p_up) * down_ret
    - ประมาณ sigma จาก p*(1-p) และระยะระหว่าง up_ret กับ down_ret
    """
    prices = _get_prices(tic, timeframe)
    if len(prices) < LOOKBACK + FORECAST_HORIZON + 2:
        return np.nan, np.nan

    risk_norm = float(risk_score) / MAX_SCORE
    X_all = _features(prices, risk_norm=risk_norm, for_train=False, ticker=tic)
    if X_all.empty:
        return np.nan, np.nan

    X_last = X_all.iloc[[-1]]

    X_aligned = X_last

    probs: List[float] = []

    try:
        xgb_cls = _load("xgb_cls", timeframe)
        c_cols = _expected_feature_names(xgb_cls)
        X_aligned = _align_X_to_expected(X_last, c_cols)
        p_up_xgb = float(xgb_cls.predict_proba(X_aligned)[0, 1])
        probs.append(p_up_xgb)
    except Exception:
        pass

    try:
        scaler_cls = _load("scaler_cls", timeframe)
        s_cols = _expected_feature_names(scaler_cls)
        X_s = _align_X_to_expected(X_aligned, s_cols) if s_cols else X_aligned
        X_scaled = scaler_cls.transform(X_s.values)
    except Exception:
        X_scaled = X_aligned.values

    try:
        lstm_cls = _load("lstm_cls", timeframe)
        X_lstm = np.expand_dims(X_scaled, 1)
        p_up_lstm = float(lstm_cls.predict(X_lstm)[0][0])  # sigmoid output
        probs.append(p_up_lstm)
    except Exception:
        pass

    if not probs:
        return np.nan, np.nan

    p_up = float(np.clip(np.mean(probs), 0.0, 1.0))
    mu = p_up * up_ret + (1.0 - p_up) * down_ret

    var_ret = p_up * (1.0 - p_up) * (up_ret - down_ret) ** 2
    sigma = float(np.sqrt(var_ret))

    return mu, sigma


def predict_views_cls(
    tickers: List[str],
    timeframe: str = "1y",
    risk_score: int = 0,
    up_ret: float = 0.03,
    down_ret: float = -0.03,
):
    """คำนวณ views (จาก classification) สำหรับหลาย ๆ ticker"""
    mu: Dict[str, float] = {}
    sig: Dict[str, float] = {}

    for t in tickers:
        m, s = predict_view_cls(
            t,
            timeframe=timeframe,
            risk_score=risk_score,
            up_ret=up_ret,
            down_ret=down_ret,
        )
        if np.isfinite(m) and np.isfinite(s):
            mu[t] = float(m)
            sig[t] = float(s)

    return mu, sig
