# plotting_utils.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd

# ---------- 1) Weights bar (Highcharts: type 'bar') ----------
def weights_bar_hc(weights: dict[str, float], title: str = "Portfolio Weights") -> dict:
    if not weights:
        return {"title": {"text": title}}

    s = pd.Series(weights, dtype=float)
    # auto-scale ถ้าเผลอส่งมาเป็นเปอร์เซ็นต์ (10 = 10%)
    if s.max() > 1.5:
        s = s / 100.0
    s = s[s > 1e-9].sort_values(ascending=False)  # กรอง ~0

    categories = list(s.index)
    data = [round(float(v * 100), 6) for v in s.values]  # ส่งเป็น %

    return {
        "chart": {"type": "bar"},
        "title": {"text": title},
        "xAxis": {"categories": categories, "title": {"text": "Ticker"}},
        "yAxis": {"min": 0, "title": {"text": "Weight (%)"}, "labels": {"format": "{value}%"}},
        "legend": {"enabled": False},
        "tooltip": {"pointFormat": "<b>{point.y:.2f}%</b>"},
        "plotOptions": {"series": {"dataLabels": {"enabled": True, "format": "{y:.2f}%"}}},
        "series": [{"name": "Weight", "data": data}],
        "credits": {"enabled": False},
    }

# ---------- 2) Covariance heatmap ----------
def cov_heatmap_hc(cov: np.ndarray, assets: list[str] | None = None, title: str = "Covariance Heat-map") -> dict:
    if cov is None or len(np.shape(cov)) != 2 or min(cov.shape) == 0:
        return {"title": {"text": title}}

    n = cov.shape[0]
    if assets is None or len(assets) != n:
        assets = [str(i) for i in range(n)]

    # แปลงเป็น [x, y, value]
    data = []
    for i in range(n):
        for j in range(n):
            data.append([i, j, float(cov[i, j])])

    vmax = max(abs(float(np.nanmax(cov))), abs(float(np.nanmin(cov)))) or 1.0

    return {
        "chart": {"type": "heatmap"},
        "title": {"text": title},
        "xAxis": {"categories": assets},
        "yAxis": {"categories": assets, "title": {"text": "Assets"}, "reversed": True},
        "colorAxis": {"min": -vmax, "max": vmax},
        "legend": {"align": "right", "layout": "vertical", "verticalAlign": "middle"},
        "series": [{"name": "Cov", "data": data, "borderWidth": 0}],
        "tooltip": {"pointFormat": "({point.x},{point.y}): <b>{point.value:.4f}</b>"},
        "credits": {"enabled": False},
    }

# ---------- 3) Efficient frontier scatter ----------
def efficient_frontier_hc(points: pd.DataFrame | None, title: str = "Efficient Frontier (random sample)") -> dict:
    """
    points: DataFrame มีคอลัมน์ ['risk','ret'] (std, mean)
    """
    if points is None or points.empty:
        return {"title": {"text": title}}

    data = [{"x": float(r), "y": float(m)} for r, m in zip(points["risk"], points["ret"])]
    return {
        "chart": {"type": "scatter", "zoomType": "xy"},
        "title": {"text": title},
        "xAxis": {"title": {"text": "Risk (σ)"}},
        "yAxis": {"title": {"text": "Return (μ)"}},
        "legend": {"enabled": False},
        "tooltip": {"pointFormat": "Risk: <b>{point.x:.3f}</b><br>Return: <b>{point.y:.3f}</b>"},
        "series": [{"name": "Candidates", "data": data}],
        "credits": {"enabled": False},
    }
