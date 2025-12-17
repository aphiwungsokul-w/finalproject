
import pandas as pd
import requests, io


# ───────────────────────── market lists ──────────────────────────
def load_tickers() -> list[str]:
    html = requests.get(
        "https://www.slickcharts.com/sp500",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    ).text
    syms = pd.read_html(io.StringIO(html))[0]["Symbol"].tolist()
    return [s.replace(".", "-") for s in syms]