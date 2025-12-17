# bl.py  – Black‑Litterman helper (Riskfolio‑Lib version – patched)
# -----------------------------------------------------------------
import numpy as np, pandas as pd, yfinance as yf, riskfolio as rp
from typing import Dict, Tuple, List

# ───────────────── helper ─────────────────
def _prices(tickers: List[str], period: str = "5y") -> pd.DataFrame:
    return yf.download(tickers, period=period, progress=False,)["Close"].ffill()

# ───────────────── make Σ positive‑definite ─────────────────
def _make_cov_pd(cov: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    cov = cov.copy()
    k = 0
    while True:
        try:
            np.linalg.cholesky(cov.values)
            break
        except np.linalg.LinAlgError:
            cov.values[np.diag_indices_from(cov)] += eps * (10 ** k)
            k += 1
    return cov

# ───────────────── manual Black‑Litterman ─────────────────
def _black_litterman(
    mu_prior: pd.Series,
    Sigma: pd.DataFrame,
    views: Dict[str, Tuple[float, float]],
    tau: float = 0.05,
) -> Tuple[pd.Series, pd.DataFrame]:
    assets = mu_prior.index.tolist()
    n      = len(assets)

    # 1) Identity view matrix  (หนึ่ง view ต่อหุ้น)
    P = np.eye(n)
    Q = np.array([views[t][0] for t in assets])

    # 2) Ω = diag(uncertainty²)
    omega = np.diag([views[t][1] ** 2 for t in assets])

    # 3) Posterior moments
    tau_S  = tau * Sigma.values
    invtauS = np.linalg.inv(tau_S)
    invO    = np.linalg.inv(omega)

    inv_term = invtauS + P.T @ invO @ P
    Sigma_bl = np.linalg.inv(inv_term)
    mu_bl    = Sigma_bl @ (invtauS @ mu_prior.values + P.T @ invO @ Q)

    mu_bl = pd.Series(mu_bl, index=assets)
    Sigma_bl = pd.DataFrame(Sigma_bl, index=assets, columns=assets)
    return mu_bl, Sigma_bl

# ── core API ─────────────────
def build_bl_port(
    market_mu: Dict[str, float],
    views: Dict[str, Tuple[float, float]],
    risk_aversion: float = 2.5,
    sector: Dict[str, str] | None = None,
    period: str = "5y",
    tau: float = 0.05,
    prices: pd.DataFrame | None = None,   # << เพิ่ม
) -> rp.Portfolio:

    tickers = list(market_mu)
    prices  = prices if prices is not None else _prices(tickers, period)
    if prices.empty:
        raise RuntimeError("No price history downloaded.")

    # ── initialise portfolio ──
    port = rp.Portfolio(returns=prices.pct_change().dropna())
    port.assets_stats(method_mu="hist", method_cov="hist")

    assets = prices.columns.tolist()           # ← ตรงกับ returns เสมอ
    port.market_caps = np.repeat(1 / len(assets), len(assets))
    port.mu_prior    = pd.Series(market_mu).reindex(assets).fillna(0)

    # ── Black‑Litterman posterior ─────────────────────────────
    mu_bl, Sigma_bl = _black_litterman(port.mu_prior, port.cov, views, tau)

    # 1) ให้ mu_bl เป็น Series ยาว n
    if isinstance(mu_bl, pd.DataFrame):
        mu_bl = mu_bl.iloc[0]

    mu_bl = mu_bl.reindex(assets)

    # 2) เติมค่า 'epsilon' ให้ทุกช่องที่ยังว่างหรือเป็น 0
    eps = 1e-6
    mu_bl = mu_bl.fillna(eps)
    mu_bl[mu_bl == 0] = eps

    # 3) เก็บเป็น "แถว" 1×n ตามที่ Riskfolio คาด
    port.mu  = pd.DataFrame([mu_bl.values], columns=assets)
    port.cov = _make_cov_pd(Sigma_bl.loc[assets, assets])

    return port

def optimize_bl(
    market_mu: Dict[str, float],
    views: Dict[str, Tuple[float, float]],
    risk_aversion: float = 2.5,
    period: str = "5y",
    tau: float = 0.05,
    prices: pd.DataFrame | None = None,
    rf: float = 0.0,
    obj: str = "Sharpe",
    rm: str = "MV",
    bounds: tuple = (0, 1),
):
    port = build_bl_port(
        market_mu=market_mu,
        views=views,
        risk_aversion=risk_aversion,
        period=period,
        tau=tau,
        prices=prices,
    )
    port.cov = _make_cov_pd(port.cov)
    try:
        w_df = port.optimization(model="Classic", rm=rm, obj=obj, rf=rf, l=risk_aversion)
        if w_df is None or getattr(w_df, "empty", True):
            raise ValueError("empty")
    except Exception:
        w_df = port.optimization(model="Classic", rm=rm, obj="MinRisk", rf=rf, l=risk_aversion)
    weights = w_df["weights"].sort_values(ascending=False)
    return weights, port