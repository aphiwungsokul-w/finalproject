def sma(s, n=14): return s.rolling(n).mean()
def ema(s, n=14): return s.ewm(span=n).mean()
def rsi(s, n=14):
    delta = s.diff(); up = delta.clip(lower=0); dn = -delta.clip(upper=0)
    rs = up.rolling(n).mean() / dn.rolling(n).mean()
    return 100 - 100/(1+rs)
def macd(s, slow=26, fast=12): return ema(s, fast) - ema(s, slow)
def bbands(s, n=20):
    ma = sma(s, n); std = s.rolling(n).std()
    return ma + 2*std, ma - 2*std
