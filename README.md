![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Alpaca](https://img.shields.io/badge/Broker-Alpaca-yellow)
![License](https://img.shields.io/badge/license-MIT-red)
![Status](https://img.shields.io/badge/status-Experimental-green)

# 🐶📈Watchdog --- Market Data + Econometrics Metrics + Trading Toolkit

Watchdog is a lightweight **market data + metrics** toolkit built on
**Alpaca-Py**.\
It **does not** ship a "secret strategy." Instead, it gives you a clean
library of **econometrics + quant trading metrics** that are intended to
be **combined by you** into **your own buy/sell logic**, risk rules, and
portfolio constraints.

> ✅ **Design goal:** Measure the market, compute signals, and let
> **you** decide when to buy/sell.\
> ⚠️ **You are responsible** for strategy design, testing, and
> execution.

------------------------------------------------------------------------

## 📌 Table of Contents

-   [What Watchdog Is](#what-watchdog-is)
-   [What Watchdog Is Not](#what-watchdog-is-not)
-   [How to Build Buy/Sell Logic With
    Watchdog](#how-to-build-buysell-logic-with-watchdog)
-   [Metrics Included](#metrics-included)
-   [Example Strategy Patterns](#example-strategy-patterns)
-   [Installation](#installation)
-   [Configuration](#configuration)
-   [Disclaimer](#disclaimer-no-financial-advice)

------------------------------------------------------------------------

## What Watchdog Is

-   ✅ A toolkit to **fetch market data** via Alpaca-Py (bars;
    optionally quotes/options where supported)
-   ✅ A set of **feature/metric functions** you can plug into:
    -   buy/sell rules
    -   ranking + selection logic (cross-sectional)
    -   regime switching
    -   risk controls (drawdown, vol, tail)
    -   sizing + allocation logic

------------------------------------------------------------------------

## What Watchdog Is Not

-   ❌ Not financial advice
-   ❌ Not a turnkey "profitable bot"
-   ❌ No guarantee of performance
-   ❌ No single "right" signal --- **you decide how to combine
    metrics**

------------------------------------------------------------------------

## How to Build Buy/Sell Logic With Watchdog

Watchdog is structured so you can layer your strategy in a
**professional** and testable way:

### 1) 🧭 Universe & Filters

Pick what you trade and apply basic tradability checks: - liquidity
filters (volume, Amihud) - volatility filters (avoid extreme regimes) -
spread filters (avoid wide bid/ask)

**Typical rules:** - Ignore symbols with **low volume** or **high Amihud
illiquidity** - Skip assets where **spread** is above a threshold

------------------------------------------------------------------------

### 2) 🧠 Signal Construction (Your Buy/Sell Triggers)

You create signals from metrics: - **Momentum** (trend-following) -
**Mean reversion** (revert-to-mean) - **Stat arb / pairs**
(cointegration + spread z-score) - **Factor / beta exposure**
constraints

A signal is usually something like:

-   **Buy condition:** `signal_score > buy_threshold`
-   **Sell condition:** `signal_score < sell_threshold`\
    (or explicit stop/exit rules)

------------------------------------------------------------------------

### 3) 🧯 Risk Controls (When NOT to Trade)

Before placing orders, you typically run "risk gates": - max drawdown
limit - high-volatility halt - tail-risk limit (VaR/CVaR) - beta
exposure clamp

Example: - "Don't open new trades when **realized vol \> 35%**" - "Exit
to cash when **max drawdown \> 10%**"

------------------------------------------------------------------------

### 4) 📏 Position Sizing & Portfolio Allocation

Watchdog gives you sizing helpers so your strategy isn't "all-in": - vol
targeting weight - Kelly fraction (simplified) - inverse-vol risk parity
(approx.) - correlation concentration checks (eigenvalue spread)

Example sizing logic: - "Size positions so each name targets **20%
annual vol**" - "Reduce weights when correlation concentration spikes"

------------------------------------------------------------------------

### 5) ✅ Execution (Your Order Logic)

Watchdog **does not** place orders for you by default.\
You feed your final decisions into your execution layer (Alpaca trading
endpoints).

A typical loop looks like:

1.  Compute metrics
2.  Produce signals + apply filters
3.  Compute sizes
4.  Place orders (your code)
5.  Log everything for replay/testing

------------------------------------------------------------------------

## Metrics Included

Most metrics are computed from stock bars
(`open/high/low/close/volume/vwap`). Some require quotes/options if
available. Full list found in `econometrics.py` \### Returns &
Performance - Log returns: `ln(P_t / P_{t-1})` - Simple returns:
`P_t/P_{t-1} - 1` - Cumulative return: `∏(1+r) - 1` - Excess returns:
`r - r_f`

### Volatility

-   Rolling volatility (std over window)
-   EWMA volatility
-   GARCH(1,1) volatility (parameter-driven)

### Risk-Adjusted Metrics

-   Sharpe ratio
-   Sortino ratio
-   Information ratio
-   Max drawdown

### Regression / Factor Exposure

-   CAPM alpha/beta via OLS: `r_i - r_f = α + β(r_m - r_f) + ε`
-   Multi-factor regression (you provide factor series):
    `r - r_f = α + Σ β_k f_k + ε`

### Time Series Econometrics

-   AR(1) φ
-   ADF p-value (stationarity)
-   Cointegration p-value + hedge beta
-   Spread series: `y - βx`
-   Z-score (price / spread)
-   Half-life of mean reversion

### Momentum / Oscillators

-   SMA / EMA
-   MACD (line/signal/hist)
-   ROC
-   RSI (Wilder)
-   12--1 momentum

### Liquidity & Microstructure (As Data Allows)

-   Bid--ask spread (latest quote or quote series)
-   Order book imbalance proxy (bid vs ask size)
-   Amihud illiquidity: `|r| / dollar_volume`
-   VWAP distance
-   Volume spike ratio
-   Bar spread proxy: `(high - low)/close`

### Tail Risk / Distribution Shape

-   Historical VaR
-   Historical CVaR
-   Skewness / Kurtosis

### Regime / Long Memory

-   Hurst exponent

### Volatility Products / "VIX-ish" Proxies

-   Term structure proxy via ratios of tradable VIX ETPs
-   Implied volatility (options snapshots, when available)
-   Historical IV by Black--Scholes inversion (option bars + underlying)
-   Realized--implied vol spread: `RV - IV`

### Portfolio Construction Helpers

-   Kelly fraction (simplified)
-   Vol targeting weight
-   Correlation matrix
-   Eigenvalue spread (concentration)
-   PCA factor exposures (optional)
-   Inverse-vol risk parity weights (approx.)

------------------------------------------------------------------------

## Example Strategy Patterns

Below are common "patterns" showing how these metrics can map to
buy/sell logic.

### 🟦 Pattern A --- Trend-Following Momentum

**Regime filter** - Trade only when `hurst_exponent > 0.5` - Avoid
extreme volatility (`realized_vol < vol_ceiling`)

**Signal** - Buy when `EMA_fast > EMA_slow` (or MACD crosses up) - Sell
when cross reverses OR trailing stop triggered

**Sizing** - Use `vol_target_weight` so high-vol names get smaller size

------------------------------------------------------------------------

### 🟩 Pattern B --- Mean Reversion

**Signal** - Compute rolling z-score of price - Buy when `z < -2` - Sell
when `z > 0` (mean reversion exit), or when z crosses back above -0.5

**Risk** - Skip trades if ADF suggests non-stationarity (or if
volatility spikes)

**Holding horizon** - Use half-life estimate to set expected hold
duration

------------------------------------------------------------------------

### 🟨 Pattern C --- Pairs / Stat-Arb

**Setup** - Test cointegration and compute hedge beta

**Signal** - Compute spread z-score - Short spread when `z > +2` - Long
spread when `z < -2` - Exit when `|z| < 0.5`

**Risk** - Stop out if cointegration breaks (p-value rises) or if spread
volatility explodes

------------------------------------------------------------------------

### 🟥 Pattern D --- Risk-Off / Risk-On Gate

**Risk-off trigger** - If realized vol rises above threshold OR tail
risk worsens (CVaR \< limit) - Reduce exposure / stop opening new trades

**Risk-on** - When vol normalizes and trend regime returns, allow
signals again

------------------------------------------------------------------------

## Installation

``` bash
pip install alpaca-py pandas numpy statsmodels scipy
```

## Disclaimer (No Financial Advice)

This software is for **educational and informational purposes only** and
is not financial, investment, or trading advice.\
You are solely responsible for your trading decisions, backtests, risk
management, and compliance with all applicable laws and broker rules.

### Limitation of Liability (Investment Losses)

**By using this repository and code, you acknowledge and agree that the
author(s) are not responsible for any investment losses, damages, or
other liabilities arising from the use of this tool.** Use at your own
risk.

------------------------------------------------------------------------

# 🔐 Configuration (Alpaca API Setup)

Watchdog connects to the **Alpaca brokerage API** to retrieve market
data and optionally execute trades.

To authenticate, place your **Alpaca API credentials** in two separate
text files located in the root of your project directory.

## Step 1 --- Get Your Alpaca API Keys

Create an account at https://alpaca.markets and generate API credentials
from the dashboard.

You will receive:

API KEY\
API SECRET

## Step 2 --- Create Credential Files

Create the following two files in the same directory as your Watchdog
script.

### key.txt

YOUR_ALPACA_API_KEY

### secret.txt

YOUR_ALPACA_API_SECRET

Important rules:

-   The files must contain **only the key string**
-   No quotes
-   No extra spaces
-   No additional lines

Example:

AKIAXXXXXXXXXXXXX

## Step 3 --- How Watchdog Loads the Credentials

Watchdog reads the credentials automatically:

``` python
key = open('key.txt', 'r').read().replace('\n', '')
secret = open('secret.txt', 'r').read().replace('\n', '')
```

These are passed into Alpaca:

``` python
self.trade_client = TradingClient(key, secret, paper=is_paper)
self.historical_client = StockHistoricalDataClient(key, secret)
```

This allows Watchdog to:

-   retrieve historical market data
-   check if assets are tradable
-   query account positions
-   submit buy/sell orders
------------------------------------------------------------------------

# ⏰ Cron Automation (Running Watchdog Automatically)

Watchdog can be automated using **cron jobs** on Linux or macOS. This
allows your strategy to run at scheduled times without manual execution.

Example: run a trading script every weekday before market open.

Edit your crontab:

``` bash
crontab -e
```

Example cron entry:

``` bash
0 6 * * 1-5 python3 /path/to/watchdog/trade_script.py
```

Meaning:

  Field   Meaning
  ------- ----------------
  0       minute
  6       hour (6 AM)
  \*      every day
  \*      every month
  1-5     Monday--Friday

Example: run every 15 minutes during market hours

``` bash
*/15 9-16 * * 1-5 python3 /path/to/watchdog/trade_script.py
```

### Logging Execution

Cron jobs should always log output for debugging:

``` bash
*/15 9-16 * * 1-5 python3 trade_script.py >> watchdog.log 2>&1
```

This stores stdout and errors in a log file.

------------------------------------------------------------------------

# 🔒 Security Best Practices

Never commit API keys to GitHub.

Add this to `.gitignore`:

    key.txt
    secret.txt

Anyone with these credentials could execute trades on your brokerage
account.

------------------------------------------------------------------------
