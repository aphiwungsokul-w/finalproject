from data import load_tickers
from config import TIMEFRAMES
from ml import train_for_timeframe

tickers = load_tickers()

for tf in TIMEFRAMES:
    train_for_timeframe(tickers, tf, with_classification=True)
