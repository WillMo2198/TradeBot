"""
alpaca_econometrics_metrics.py

Goal: "each formula -> its own Python function" using alpaca-py data fetches.

pip install alpaca-py pandas numpy statsmodels scipy

Optional (only if you want PCA convenience):
pip install scikit-learn

Notes / reality checks (Alpaca limits):
- CBOE indexes like VIX are NOT equities, so Alpaca stock data won’t give true VIX or VIX futures term structure. Alpaca staff have said this historically.  [oai_citation:0‡Alpaca Community Forum](https://forum.alpaca.markets/t/how-to-get-volatility-data-vix-vxtlt-etc/2907?utm_source=chatgpt.com)
  -> I provide *proxy* term-structure functions using tradable VIX ETPs (VXX/VIXY/etc) that Alpaca *can* serve as equities.
- Alpaca option snapshots include latest implied vol + greeks, but historical IV is not directly in historical option bars; you either:
  (a) use snapshots (latest), or
  (b) compute IV yourself from historical option prices (Black-Scholes inversion).
  Alpaca docs: options snapshots include implied vol.  [oai_citation:1‡Alpaca](https://alpaca.markets/sdks/python/api_reference/data/models.html?utm_source=chatgpt.com)
  Alpaca options historical endpoints exist (bars/trades).  [oai_citation:2‡Alpaca](https://alpaca.markets/sdks/python/api_reference/data/option/historical.html?utm_source=chatgpt.com)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, Sequence, Dict, Tuple, Union, Iterable

import numpy as np
import pandas as pd

# Alpaca
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockQuotesRequest,
    StockLatestQuoteRequest,
    OptionBarsRequest,
    OptionSnapshotRequest,
)
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from scipy.stats import skew as _skew, kurtosis as _kurtosis
from scipy.stats import norm


# ============================================================
# 0) Data fetch helpers (alpaca-py)
# ============================================================

def get_stock_bars_df(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    timeframe: TimeFrame = TimeFrame.Day,
    adjustment: str = "all",
) -> pd.DataFrame:
    """OHLCV+ (vwap, trade_count) DataFrame indexed by timestamp."""
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        adjustment=adjustment,
    )
    bars = stock_client.get_stock_bars(req).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level=0)
    return bars.sort_index()


def get_close(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    return get_stock_bars_df(stock_client, symbol, start, end, timeframe)["close"]


def get_stock_quotes_df(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
) -> pd.DataFrame:
    """
    Historical quotes (bid/ask + sizes). Useful for microstructure metrics.
    """
    req = StockQuotesRequest(
        symbol_or_symbols=symbol,
        start=start,
        end=end,
    )
    q = stock_client.get_stock_quotes(req).df
    if isinstance(q.index, pd.MultiIndex):
        q = q.xs(symbol, level=0)
    return q.sort_index()


def get_latest_quote(
    stock_client: StockHistoricalDataClient,
    symbol: str,
) -> Dict[str, float]:
    """
    Returns latest quote fields: bid_price, ask_price, bid_size, ask_size.
    """
    req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
    out = stock_client.get_stock_latest_quote(req)
    qq = out[symbol]
    return {
        "bid_price": float(qq.bid_price),
        "ask_price": float(qq.ask_price),
        "bid_size": float(qq.bid_size),
        "ask_size": float(qq.ask_size),
    }


def get_option_bars_df(
    opt_client: OptionHistoricalDataClient,
    contract_symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.DataFrame:
    """
    Historical option bars for a contract symbol.
    """
    req = OptionBarsRequest(
        symbol_or_symbols=contract_symbol,
        timeframe=timeframe,
        start=start,
        end=end,
    )
    bars = opt_client.get_option_bars(req).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(contract_symbol, level=0)
    return bars.sort_index()


def get_option_snapshot(
    opt_client: OptionHistoricalDataClient,
    contract_symbol: str,
) -> Dict[str, float]:
    """
    Latest option snapshot (includes greeks & implied vol per Alpaca docs).
     [oai_citation:3‡Alpaca](https://alpaca.markets/sdks/python/api_reference/data/models.html?utm_source=chatgpt.com)
    """
    req = OptionSnapshotRequest(symbol_or_symbols=[contract_symbol])
    snaps = opt_client.get_option_snapshot(req)
    s = snaps[contract_symbol]
    iv = getattr(getattr(s, "greeks", None), "implied_volatility", None)
    if iv is None and hasattr(s, "implied_volatility"):
        iv = s.implied_volatility
    return {"implied_volatility": float(iv) if iv is not None else float("nan")}


# ============================================================
# 1) Returns metrics 
# ============================================================

def log_returns(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """r_t = ln(P_t / P_{t-1})"""
    p = get_close(stock_client, symbol, start, end, timeframe)
    return np.log(p / p.shift(1)).dropna()


def simple_returns(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """r_t = P_t / P_{t-1} - 1"""
    p = get_close(stock_client, symbol, start, end, timeframe)
    return p.pct_change().dropna()


def cumulative_return(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    use_log: bool = False,
    timeframe: TimeFrame = TimeFrame.Day,
) -> float:
    """R = Π(1+r_t)-1  OR exp(Σ log r_t)-1"""
    if use_log:
        r = log_returns(stock_client, symbol, start, end, timeframe)
        return float(np.exp(r.sum()) - 1.0)
    r = simple_returns(stock_client, symbol, start, end, timeframe)
    return float((1.0 + r).prod() - 1.0)


def excess_returns(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """r_t - r_f (rf annual converted to per-period)"""
    r = simple_returns(stock_client, symbol, start, end, timeframe)
    rf_period = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    return r - rf_period


# ============================================================
# 2) Volatility metrics 
# ============================================================

def rolling_volatility(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """σ_t = std(r over window)"""
    r = simple_returns(stock_client, symbol, start, end, timeframe)
    vol = r.rolling(window).std(ddof=1)
    return vol * np.sqrt(periods_per_year) if annualize else vol


def ewma_volatility(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    lam: float = 0.94,
    annualize: bool = True,
    periods_per_year: int = 252,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """σ_t^2 = λ σ_{t-1}^2 + (1-λ) r_{t-1}^2"""
    r = simple_returns(stock_client, symbol, start, end, timeframe).dropna()
    var = np.zeros(len(r))
    var[0] = float(r.iloc[0] ** 2)
    for i in range(1, len(r)):
        var[i] = lam * var[i - 1] + (1.0 - lam) * float(r.iloc[i - 1] ** 2)
    vol = pd.Series(np.sqrt(var), index=r.index)
    return vol * np.sqrt(periods_per_year) if annualize else vol


def garch11_volatility(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    omega: float,
    alpha: float,
    beta: float,
    annualize: bool = True,
    periods_per_year: int = 252,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """
    GARCH(1,1): σ_t^2 = ω + α r_{t-1}^2 + β σ_{t-1}^2
    (You provide params; estimation is separate.)
    """
    r = simple_returns(stock_client, symbol, start, end, timeframe).dropna()
    var = np.zeros(len(r))
    var[0] = float(np.var(r.values))
    for i in range(1, len(r)):
        var[i] = omega + alpha * float(r.iloc[i - 1] ** 2) + beta * var[i - 1]
    vol = pd.Series(np.sqrt(np.maximum(var, 0.0)), index=r.index)
    return vol * np.sqrt(periods_per_year) if annualize else vol


# ============================================================
# 3) Risk-adjusted metrics 
# ============================================================

def sharpe_ratio(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
    timeframe: TimeFrame = TimeFrame.Day,
) -> float:
    """Sharpe = E[r-rf] / std(r-rf)"""
    ex = excess_returns(stock_client, symbol, start, end, rf_annual, periods_per_year, timeframe)
    mu, sd = float(ex.mean()), float(ex.std(ddof=1))
    if sd == 0:
        return float("nan")
    return float((mu / sd) * np.sqrt(periods_per_year))


def sortino_ratio(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
    timeframe: TimeFrame = TimeFrame.Day,
) -> float:
    """Sortino = E[r-rf] / std_downside(r-rf)"""
    ex = excess_returns(stock_client, symbol, start, end, rf_annual, periods_per_year, timeframe)
    downside = ex.where(ex < 0.0, 0.0)
    dd = float(np.sqrt((downside**2).mean()))
    mu = float(ex.mean())
    if dd == 0:
        return float("nan")
    return float((mu / dd) * np.sqrt(periods_per_year))


def information_ratio(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    benchmark: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    periods_per_year: int = 252,
    timeframe: TimeFrame = TimeFrame.Day,
) -> float:
    """IR = E[r-rb] / std(r-rb)"""
    r = simple_returns(stock_client, symbol, start, end, timeframe)
    rb = simple_returns(stock_client, benchmark, start, end, timeframe)
    df = pd.concat([r.rename("r"), rb.rename("rb")], axis=1, join="inner").dropna()
    active = df["r"] - df["rb"]
    mu, sd = float(active.mean()), float(active.std(ddof=1))
    if sd == 0:
        return float("nan")
    return float((mu / sd) * np.sqrt(periods_per_year))


def max_drawdown(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    timeframe: TimeFrame = TimeFrame.Day,
) -> float:
    """MDD = max((peak - equity)/peak) on equity curve Π(1+r)"""
    r = simple_returns(stock_client, symbol, start, end, timeframe)
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = (peak - equity) / peak
    return float(dd.max())


# ============================================================
# 4) Regression metrics 
# ============================================================

def capm_alpha_beta(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    market: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
    timeframe: TimeFrame = TimeFrame.Day,
) -> Tuple[float, float]:
    """
    r_i - rf = alpha + beta*(r_m - rf) + eps
    returns: (alpha_annualized, beta)
    """
    ri = excess_returns(stock_client, symbol, start, end, rf_annual, periods_per_year, timeframe)
    rm = excess_returns(stock_client, market, start, end, rf_annual, periods_per_year, timeframe)
    df = pd.concat([ri.rename("ri"), rm.rename("rm")], axis=1, join="inner").dropna()
    y = df["ri"].values
    X = sm.add_constant(df["rm"].values)
    res = sm.OLS(y, X).fit()
    alpha = float(res.params[0])
    beta = float(res.params[1])
    alpha_ann = (1.0 + alpha) ** periods_per_year - 1.0
    return float(alpha_ann), float(beta)


def multifactor_alpha_betas(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    factors_returns: pd.DataFrame,
    start: Union[str, datetime],
    end: Union[str, datetime],
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
    timeframe: TimeFrame = TimeFrame.Day,
) -> Dict[str, float]:
    """
    r - rf = alpha + Σ beta_k * factor_k + eps
    factors_returns: DataFrame indexed by timestamp; columns are factor returns per-period.
    """
    y = excess_returns(stock_client, symbol, start, end, rf_annual, periods_per_year, timeframe).rename("y")
    df = pd.concat([y, factors_returns], axis=1, join="inner").dropna()
    Y = df["y"].values
    X = sm.add_constant(df.drop(columns=["y"]).values)
    res = sm.OLS(Y, X).fit()
    out = {}
    alpha = float(res.params[0])
    out["alpha_annualized"] = float((1.0 + alpha) ** periods_per_year - 1.0)
    for i, name in enumerate(df.drop(columns=["y"]).columns, start=1):
        out[f"beta_{name}"] = float(res.params[i])
    return out


# ============================================================
# 5) Time series econometrics 
# ============================================================

def ar1_phi(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    timeframe: TimeFrame = TimeFrame.Day,
) -> float:
    """r_t = phi*r_{t-1} + eps (OLS slope)"""
    r = simple_returns(stock_client, symbol, start, end, timeframe).dropna()
    y = r.iloc[1:].values
    x = r.iloc[:-1].values
    res = sm.OLS(y, sm.add_constant(x)).fit()
    return float(res.params[1])


def adf_pvalue(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    use_returns: bool = True,
    timeframe: TimeFrame = TimeFrame.Day,
) -> float:
    """ADF p-value for stationarity (returns or price)."""
    s = simple_returns(stock_client, symbol, start, end, timeframe) if use_returns else get_close(stock_client, symbol, start, end, timeframe)
    s = s.dropna().values
    return float(adfuller(s, autolag="AIC")[1])


def cointegration_pvalue_beta(
    stock_client: StockHistoricalDataClient,
    y_symbol: str,
    x_symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    timeframe: TimeFrame = TimeFrame.Day,
) -> Tuple[float, float]:
    """
    Engle-Granger: y - beta*x stationary
    returns: (pvalue, beta_hat from OLS y ~ beta*x)
    """
    y = get_close(stock_client, y_symbol, start, end, timeframe).rename("y")
    x = get_close(stock_client, x_symbol, start, end, timeframe).rename("x")
    df = pd.concat([y, x], axis=1, join="inner").dropna()
    res = sm.OLS(df["y"].values, sm.add_constant(df["x"].values)).fit()
    beta = float(res.params[1])
    pval = float(coint(df["y"].values, df["x"].values)[1])
    return pval, beta


def spread_series(
    stock_client: StockHistoricalDataClient,
    y_symbol: str,
    x_symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    beta: Optional[float] = None,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """spread_t = y_t - beta*x_t (beta estimated if None)"""
    y = get_close(stock_client, y_symbol, start, end, timeframe).rename("y")
    x = get_close(stock_client, x_symbol, start, end, timeframe).rename("x")
    df = pd.concat([y, x], axis=1, join="inner").dropna()
    if beta is None:
        res = sm.OLS(df["y"].values, sm.add_constant(df["x"].values)).fit()
        beta = float(res.params[1])
    return (df["y"] - beta * df["x"]).rename("spread")


def zscore_price(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    window: int = 60,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """Z_t = (P_t - mu)/sigma (rolling)"""
    p = get_close(stock_client, symbol, start, end, timeframe)
    mu = p.rolling(window).mean()
    sd = p.rolling(window).std(ddof=1)
    return ((p - mu) / sd).dropna()


def zscore_spread(
    stock_client: StockHistoricalDataClient,
    y_symbol: str,
    x_symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    window: int = 60,
    beta: Optional[float] = None,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """Z-score of cointegration spread."""
    sp = spread_series(stock_client, y_symbol, x_symbol, start, end, beta, timeframe)
    mu = sp.rolling(window).mean()
    sd = sp.rolling(window).std(ddof=1)
    return ((sp - mu) / sd).dropna()


def half_life_mean_reversion(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    timeframe: TimeFrame = TimeFrame.Day,
) -> float:
    """
    HL = ln(2)/theta, theta from Δx_t = a + b*x_{t-1} + eps, theta=-b
    """
    x = get_close(stock_client, symbol, start, end, timeframe).dropna()
    dx = x.diff().dropna()
    lag = x.shift(1).dropna()
    df = pd.concat([dx.rename("dx"), lag.rename("lag")], axis=1).dropna()
    res = sm.OLS(df["dx"].values, sm.add_constant(df["lag"].values)).fit()
    b = float(res.params[1])
    theta = -b
    if theta <= 0:
        return float("inf")
    return float(np.log(2.0) / theta)


# ============================================================
# 6) Momentum metrics 
# ============================================================

def sma(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    window: int = 20,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """SMA = mean(P over window)"""
    p = get_close(stock_client, symbol, start, end, timeframe)
    return p.rolling(window).mean().dropna()


def ema(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    span: int = 20,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """EMA(span)"""
    p = get_close(stock_client, symbol, start, end, timeframe)
    return p.ewm(span=span, adjust=False).mean().dropna()


def macd(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.DataFrame:
    """MACD line, signal line, histogram."""
    p = get_close(stock_client, symbol, start, end, timeframe).dropna()
    fast_ema = p.ewm(span=fast, adjust=False).mean()
    slow_ema = p.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    sig = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig
    return pd.DataFrame({"macd": macd_line, "signal": sig, "hist": hist}).dropna()


def roc(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    n: int = 10,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """ROC = (P_t - P_{t-n}) / P_{t-n}"""
    p = get_close(stock_client, symbol, start, end, timeframe)
    return ((p - p.shift(n)) / p.shift(n)).dropna()


def rsi(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    window: int = 14,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """RSI (Wilder smoothing)."""
    p = get_close(stock_client, symbol, start, end, timeframe).dropna()
    d = p.diff()
    up = d.clip(lower=0.0)
    down = (-d).clip(lower=0.0)
    alpha = 1.0 / window
    roll_up = up.ewm(alpha=alpha, adjust=False).mean()
    roll_down = down.ewm(alpha=alpha, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).dropna()


def momentum_12_1(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    periods_per_year: int = 252,
    timeframe: TimeFrame = TimeFrame.Day,
) -> float:
    """
    Classic 12-1 momentum: past 12 months return excluding most recent month.
    Using daily data: lookback ~252, skip ~21.
    """
    p = get_close(stock_client, symbol, start, end, timeframe).dropna()
    if len(p) < 252 + 21 + 2:
        return float("nan")
    p_end = p.iloc[-21]          # end of formation period (skip last month)
    p_start = p.iloc[-(252+21)]  # start 12 months earlier
    return float(p_end / p_start - 1.0)


# ============================================================
# 7) Liquidity & microstructure metrics 
# ============================================================

def bid_ask_spread_latest(
    stock_client: StockHistoricalDataClient,
    symbol: str,
) -> float:
    """(ask - bid) / mid using latest quote."""
    q = get_latest_quote(stock_client, symbol)
    bid, ask = q["bid_price"], q["ask_price"]
    mid = (bid + ask) / 2.0
    if mid == 0:
        return float("nan")
    return float((ask - bid) / mid)


def order_book_imbalance_latest(
    stock_client: StockHistoricalDataClient,
    symbol: str,
) -> float:
    """
    Imbalance = (bid_size - ask_size) / (bid_size + ask_size) using latest quote.
    """
    q = get_latest_quote(stock_client, symbol)
    b, a = q["bid_size"], q["ask_size"]
    denom = b + a
    if denom == 0:
        return float("nan")
    return float((b - a) / denom)


def bid_ask_spread_series(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
) -> pd.Series:
    """(ask - bid)/mid over historical quotes."""
    q = get_stock_quotes_df(stock_client, symbol, start, end)
    bid = q["bid_price"].astype(float)
    ask = q["ask_price"].astype(float)
    mid = (bid + ask) / 2.0
    return ((ask - bid) / mid.replace(0.0, np.nan)).dropna()


def order_book_imbalance_series(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
) -> pd.Series:
    """(bid_size - ask_size)/(bid_size + ask_size) over historical quotes."""
    q = get_stock_quotes_df(stock_client, symbol, start, end)
    b = q["bid_size"].astype(float)
    a = q["ask_size"].astype(float)
    denom = (b + a).replace(0.0, np.nan)
    return ((b - a) / denom).dropna()


def bar_spread_proxy(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """Proxy spread = (high - low)/close from bars."""
    df = get_stock_bars_df(stock_client, symbol, start, end, timeframe)
    return ((df["high"] - df["low"]) / df["close"]).dropna()


def amihud_illiquidity(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """Amihud = |r_t| / DollarVolume_t, DollarVolume ~ close*volume."""
    df = get_stock_bars_df(stock_client, symbol, start, end, timeframe)
    r_abs = df["close"].pct_change().abs()
    dollar_vol = (df["close"] * df["volume"]).replace(0.0, np.nan)
    return (r_abs / dollar_vol).dropna()


def vwap_distance(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """(close - vwap) / vwap"""
    df = get_stock_bars_df(stock_client, symbol, start, end, timeframe)
    vwap = df["vwap"].replace(0.0, np.nan)
    return ((df["close"] - vwap) / vwap).dropna()


def volume_spike_ratio(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    window: int = 20,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """volume / SMA(volume, window)"""
    df = get_stock_bars_df(stock_client, symbol, start, end, timeframe)
    vol = df["volume"]
    return (vol / vol.rolling(window).mean()).dropna()


# ============================================================
# 8) Tail & distribution metrics 
# ============================================================

def historical_var(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    alpha: float = 0.05,
    timeframe: TimeFrame = TimeFrame.Day,
) -> float:
    """VaR_alpha = quantile(r, alpha)"""
    r = simple_returns(stock_client, symbol, start, end, timeframe)
    return float(np.quantile(r.values, alpha))


def historical_cvar(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    alpha: float = 0.05,
    timeframe: TimeFrame = TimeFrame.Day,
) -> float:
    """CVaR_alpha = E[r | r <= VaR_alpha]"""
    r = simple_returns(stock_client, symbol, start, end, timeframe)
    var = float(np.quantile(r.values, alpha))
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) else float("nan")


def return_skewness(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    timeframe: TimeFrame = TimeFrame.Day,
) -> float:
    """Skewness of returns."""
    r = simple_returns(stock_client, symbol, start, end, timeframe)
    return float(_skew(r.values, bias=False))


def return_kurtosis(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    fisher: bool = True,
    timeframe: TimeFrame = TimeFrame.Day,
) -> float:
    """Kurtosis of returns (fisher=True -> excess kurtosis)."""
    r = simple_returns(stock_client, symbol, start, end, timeframe)
    return float(_kurtosis(r.values, fisher=fisher, bias=False))


# ============================================================
# 9) Regime / long-memory metric (Hurst)
# ============================================================

def hurst_exponent(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    max_lag: int = 100,
    timeframe: TimeFrame = TimeFrame.Day,
) -> float:
    """
    Simple Hurst via variance of lagged differences:
    H ~ 2*slope of log(tau) vs log(lag)
    """
    p = get_close(stock_client, symbol, start, end, timeframe).dropna().values
    if len(p) < max_lag + 5:
        return float("nan")
    lags = np.arange(2, max_lag)
    tau = np.array([np.sqrt(np.std(p[lag:] - p[:-lag])) for lag in lags])
    x = np.log(lags)
    y = np.log(tau + 1e-12)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope * 2.0)


# ============================================================
# 10) Volatility products / VIX proxies (term structure + IV spread)
# ============================================================

def vix_term_structure_proxy_ratio(
    stock_client: StockHistoricalDataClient,
    short_vol_etp: str,
    mid_vol_etp: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """
    Proxy term structure using two VIX futures ETPs (both are equities, tradable).
    Example pairs you might try: VXX (short-term) vs a mid-term VIX ETP (if available).
    True VIX index is not provided as equity data by Alpaca.  [oai_citation:4‡Alpaca Community Forum](https://forum.alpaca.markets/t/how-to-get-volatility-data-vix-vxtlt-etc/2907?utm_source=chatgpt.com)
    """
    a = get_close(stock_client, short_vol_etp, start, end, timeframe).rename("short")
    b = get_close(stock_client, mid_vol_etp, start, end, timeframe).rename("mid")
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    return (df["mid"] / df["short"]).rename("term_structure_proxy")


def realized_volatility_annualized(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    window: int = 20,
    periods_per_year: int = 252,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """Realized vol ≈ std(returns, window)*sqrt(ppy)"""
    r = simple_returns(stock_client, symbol, start, end, timeframe)
    return (r.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)).dropna()


def option_implied_vol_latest_snapshot(
    opt_client: OptionHistoricalDataClient,
    contract_symbol: str,
) -> float:
    """
    Latest implied vol from Alpaca option snapshot (when available).  [oai_citation:5‡Alpaca](https://alpaca.markets/sdks/python/api_reference/data/models.html?utm_source=chatgpt.com)
    """
    return float(get_option_snapshot(opt_client, contract_symbol).get("implied_volatility", float("nan")))


# ---- Black-Scholes IV computation from historical option prices ----

def _bs_price(
    S: float, K: float, T: float, r: float, sigma: float, is_call: bool, q: float = 0.0
) -> float:
    """
    Black-Scholes with continuous dividend yield q.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))


def implied_vol_black_scholes(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    is_call: bool,
    q: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Solve for sigma such that BS_price(sigma) ~= price using bisection.
    """
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return float("nan")
    lo, hi = 1e-6, 5.0  # 500% vol upper bound
    # Ensure bracket
    p_lo = _bs_price(S, K, T, r, lo, is_call, q)
    p_hi = _bs_price(S, K, T, r, hi, is_call, q)
    if not np.isfinite(p_lo) or not np.isfinite(p_hi):
        return float("nan")
    if price < min(p_lo, p_hi) or price > max(p_lo, p_hi):
        # Can't bracket (deep ITM/OTM + stale price). Return nan.
        return float("nan")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = _bs_price(S, K, T, r, mid, is_call, q)
        if not np.isfinite(p_mid):
            return float("nan")
        if abs(p_mid - price) < tol:
            return float(mid)
        if (p_mid - price) * (p_lo - price) < 0:
            hi = mid
            p_hi = p_mid
        else:
            lo = mid
            p_lo = p_mid
    return float(0.5 * (lo + hi))


def option_implied_vol_series_from_bars(
    stock_client: StockHistoricalDataClient,
    opt_client: OptionHistoricalDataClient,
    underlying_symbol: str,
    option_contract_symbol: str,
    strike: float,
    expiry: Union[str, date],
    is_call: bool,
    start: Union[str, datetime],
    end: Union[str, datetime],
    rf_annual: float = 0.0,
    q_annual: float = 0.0,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """
    Historical IV series computed by inverting Black-Scholes using:
    - underlying close (alpaca stock bars)
    - option close (alpaca option bars)
    Alpaca historical option bars exist, but they don't directly include IV.  [oai_citation:6‡Alpaca](https://alpaca.markets/sdks/python/api_reference/data/option/historical.html?utm_source=chatgpt.com)
    """
    if isinstance(expiry, str):
        expiry_dt = pd.to_datetime(expiry).date()
    else:
        expiry_dt = expiry

    und = get_close(stock_client, underlying_symbol, start, end, timeframe).rename("S")
    opt = get_option_bars_df(opt_client, option_contract_symbol, start, end, timeframe)["close"].rename("opt_close")
    df = pd.concat([und, opt], axis=1, join="inner").dropna()

    ivs = []
    for ts, row in df.iterrows():
        d = ts.date()
        T_days = (expiry_dt - d).days
        if T_days <= 0:
            ivs.append(np.nan)
            continue
        T = T_days / 365.0
        ivs.append(
            implied_vol_black_scholes(
                price=float(row["opt_close"]),
                S=float(row["S"]),
                K=float(strike),
                T=float(T),
                r=float(rf_annual),
                is_call=bool(is_call),
                q=float(q_annual),
            )
        )
    return pd.Series(ivs, index=df.index, name="implied_vol").dropna()


def realized_minus_implied_vol_spread(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    realized_start: Union[str, datetime],
    realized_end: Union[str, datetime],
    implied_vol_series: pd.Series,
    window: int = 20,
    periods_per_year: int = 252,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """
    (Realized vol) - (Implied vol)
    implied_vol_series: you can pass either snapshot-based IV (constant) or a computed series.
    """
    rv = realized_volatility_annualized(stock_client, symbol, realized_start, realized_end, window, periods_per_year, timeframe)
    df = pd.concat([rv.rename("rv"), implied_vol_series.rename("iv")], axis=1, join="inner").dropna()
    return (df["rv"] - df["iv"]).rename("rv_minus_iv")


# ============================================================
# 11) Portfolio construction metrics 
# ============================================================

def kelly_fraction(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    timeframe: TimeFrame = TimeFrame.Day,
) -> float:
    """
    Simple Kelly for returns: f* = mean(r)/var(r)
    (For log-utility with small returns; clamp in your strategy.)
    """
    r = simple_returns(stock_client, symbol, start, end, timeframe).dropna()
    mu = float(r.mean())
    var = float(r.var(ddof=1))
    if var == 0:
        return float("nan")
    return float(mu / var)


def vol_target_weight(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    target_vol_annual: float = 0.20,
    window: int = 20,
    periods_per_year: int = 252,
    timeframe: TimeFrame = TimeFrame.Day,
) -> float:
    """w = target_vol / realized_vol"""
    r = simple_returns(stock_client, symbol, start, end, timeframe).dropna()
    vol = float(r.tail(window).std(ddof=1) * np.sqrt(periods_per_year))
    if vol == 0:
        return float("nan")
    return float(target_vol_annual / vol)


def correlation_matrix(
    stock_client: StockHistoricalDataClient,
    symbols: Sequence[str],
    start: Union[str, datetime],
    end: Union[str, datetime],
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.DataFrame:
    """Correlation matrix of returns."""
    rets = []
    for s in symbols:
        rets.append(simple_returns(stock_client, s, start, end, timeframe).rename(s))
    df = pd.concat(rets, axis=1, join="inner").dropna()
    return df.corr()


def eigenvalue_spread(
    stock_client: StockHistoricalDataClient,
    symbols: Sequence[str],
    start: Union[str, datetime],
    end: Union[str, datetime],
    timeframe: TimeFrame = TimeFrame.Day,
) -> float:
    """
    Eigenvalue spread of correlation matrix = (largest eigenvalue) / (sum eigenvalues).
    Often used as a "single factor dominance" indicator.
    """
    C = correlation_matrix(stock_client, symbols, start, end, timeframe)
    vals = np.linalg.eigvalsh(C.values)
    vals = np.sort(np.real(vals))
    if vals.sum() == 0:
        return float("nan")
    return float(vals[-1] / vals.sum())


def pca_factor_exposures(
    stock_client: StockHistoricalDataClient,
    symbols: Sequence[str],
    start: Union[str, datetime],
    end: Union[str, datetime],
    n_components: int = 3,
    timeframe: TimeFrame = TimeFrame.Day,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    PCA on standardized returns.
    Returns: (loadings_df, explained_variance_ratio)
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception:
        raise ImportError("Install scikit-learn for PCA: pip install scikit-learn")

    rets = []
    for s in symbols:
        rets.append(simple_returns(stock_client, s, start, end, timeframe).rename(s))
    X = pd.concat(rets, axis=1, join="inner").dropna()
    Z = StandardScaler().fit_transform(X.values)
    pca = PCA(n_components=n_components).fit(Z)
    loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=[f"PC{i+1}" for i in range(n_components)])
    return loadings, pca.explained_variance_ratio_


def risk_parity_weights_inverse_vol(
    stock_client: StockHistoricalDataClient,
    symbols: Sequence[str],
    start: Union[str, datetime],
    end: Union[str, datetime],
    window: int = 60,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.Series:
    """Approx risk parity: w_i ∝ 1/σ_i (σ from rolling std of returns)."""
    vols = {}
    for s in symbols:
        r = simple_returns(stock_client, s, start, end, timeframe).dropna()
        vols[s] = float(r.tail(window).std(ddof=1))
    inv = pd.Series({k: (1.0 / v if v and np.isfinite(v) else np.nan) for k, v in vols.items()}).dropna()
    return inv / inv.sum()


# ============================================================
# 12) Convenience regime flags (using earlier metrics)
# ============================================================

def trending_regime(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    hurst_threshold: float = 0.5,
    timeframe: TimeFrame = TimeFrame.Day,
) -> bool:
    """Trending if H > threshold."""
    return hurst_exponent(stock_client, symbol, start, end, timeframe=timeframe) > hurst_threshold


def high_vol_regime(
    stock_client: StockHistoricalDataClient,
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    window: int = 20,
    vol_threshold_annual: float = 0.30,
    periods_per_year: int = 252,
    timeframe: TimeFrame = TimeFrame.Day,
) -> bool:
    """High vol if realized vol > threshold."""
    rv = realized_volatility_annualized(stock_client, symbol, start, end, window, periods_per_year, timeframe)
    return bool(rv.iloc[-1] > vol_threshold_annual) if len(rv) else False
