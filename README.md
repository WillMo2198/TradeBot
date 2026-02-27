# Watchdog — Market Metrics Toolkit (Alpaca-Py)

Watchdog is a lightweight **market data + metrics** toolkit built on **Alpaca-Py**.  
It **does not** provide “magic” buy/sell signals. Instead, it computes commonly used **econometrics + quant trading metrics** so **you** can implement your own buy/sell logic, risk rules, and portfolio constraints.

> **You are responsible for your strategy decisions.** Watchdog is designed to help you **measure** the market—not tell you what to trade.

---

## What Watchdog Does

- Fetches historical market data (bars, and optionally quotes/options where supported) using **Alpaca-Py**
- Computes a library of **econometrics / quant metrics**
- Provides outputs you can feed into:
  - strategy rules (momentum, mean reversion, stat arb, etc.)
  - regime filters (trend vs mean-reverting, high-vol vs low-vol)
  - position sizing (vol targeting, Kelly fraction, risk parity)
  - portfolio checks (correlation concentration, drawdown limits)

---

## What Watchdog Does *Not* Do

- ❌ It does not provide financial advice
- ❌ It does not guarantee profits or performance
- ❌ It does not claim any metric is “the right” trading signal
- ❌ It does not force a single buy/sell model  
  (You decide how to combine metrics into your own logic.)

---

## Metrics Included

Below is the menu of metrics Watchdog exposes. Most are computed from Alpaca stock bars (`open/high/low/close/volume/vwap`) and some use quotes or options where supported.

### 1) Returns & Performance
- **Log returns**: `ln(P_t / P_{t-1})`
- **Simple returns**: `P_t/P_{t-1} - 1`
- **Cumulative return**: `∏(1+r) - 1`
- **Excess return**: `r - r_f`

### 2) Volatility
- **Rolling volatility** (std of returns over window)
- **EWMA volatility** (exponentially weighted)
- **GARCH(1,1) volatility** (parameter-driven variant)

### 3) Risk-Adjusted Metrics
- **Sharpe ratio**
- **Sortino ratio**
- **Information ratio**
- **Max drawdown**

### 4) Regression / Factor Exposure
- **CAPM alpha & beta** via OLS regression:
  - `r_i - r_f = α + β(r_m - r_f) + ε`
- **Multi-factor regression** (you provide factor return series):
  - `r - r_f = α + Σ β_k f_k + ε`

### 5) Time Series Econometrics
- **AR(1) φ** (autocorrelation / momentum vs mean-reversion hint)
- **ADF test p-value** (stationarity check)
- **Cointegration test p-value + hedge beta** (pairs/stat-arb)
- **Spread series**: `y - βx`
- **Z-score of price** (rolling)
- **Z-score of spread** (rolling)
- **Half-life of mean reversion** (from AR-style regression on spread/price)

### 6) Momentum / Trend / Oscillators
- **SMA / EMA**
- **MACD** (line, signal, histogram)
- **Rate of Change (ROC)**
- **RSI** (Wilder smoothing)
- **12–1 momentum** (classic academic momentum measure)

### 7) Liquidity & Microstructure (as data allows)
- **Bid–ask spread** (latest quote) and/or time series spread (quotes)
- **Order book imbalance** (bid size vs ask size proxy from quotes)
- **Amihud illiquidity**: `|r| / dollar_volume`
- **VWAP distance**: `(close - vwap)/vwap`
- **Volume spike ratio**: `volume / SMA(volume)`
- **Bar spread proxy**: `(high - low)/close`

### 8) Tail Risk & Distribution Shape
- **Historical VaR**
- **Historical CVaR**
- **Skewness** of returns
- **Kurtosis** of returns

### 9) Regime / Long Memory
- **Hurst exponent** (trend vs mean reversion heuristic)

### 10) Volatility Products / “VIX-ish” Proxies (Important!)
Alpaca stock data does not typically include the **true VIX index** as an equity.  
So Watchdog supports:
- **Term structure proxy** using ratios of **tradable VIX ETPs** (e.g., “short-term” vs “mid-term” vol products)
- **Implied volatility** (where available) from **options snapshots**
- **Historical implied volatility** by **inverting Black–Scholes** from historical option bars + underlying prices
- **Realized–implied vol spread**: `RV - IV`

> Practical note: “VIX term structure” is best done using VIX futures data. If your broker/data vendor doesn’t provide that, ETP ratios are a proxy, not a substitute.

### 11) Portfolio Construction Helpers
- **Kelly fraction**: `mean(r) / var(r)` (simplified)
- **Vol targeting weight**: `target_vol / realized_vol`
- **Correlation matrix**
- **Eigenvalue spread** (concentration / single-factor dominance)
- **PCA factor exposures** (optional; requires scikit-learn)
- **Inverse-vol “risk parity” weights** (approx.)

---

## How You’re Expected to Use These Metrics

Watchdog is intentionally modular. A typical workflow is:

1. **Regime filter**
   - e.g., use Hurst + realized volatility to choose momentum vs mean reversion

2. **Signal**
   - momentum: moving average cross, MACD, 12–1 momentum
   - mean reversion: z-score thresholds, half-life
   - stat arb: cointegration + spread z-score

3. **Risk & sizing**
   - vol targeting, max drawdown caps, VaR/CVaR checks
   - beta constraints / factor neutrality (CAPM or multi-factor)

4. **Execution (your code)**
   - place orders with Alpaca trading endpoints based on your rules

---

## Installation

```bash
pip install alpaca-py pandas numpy statsmodels scipy
```

Optional:
```bash
pip install scikit-learn
```

---

## API Keys / Environment

You’ll need Alpaca API credentials (paper trading recommended while testing).

Common approach:
- store keys in environment variables
- or keep `key.txt` / `secret.txt` locally (do not commit to GitHub)

---

## Example (Conceptual)

Pseudo-logic (not financial advice):

- If **Hurst > 0.5** and **vol is moderate** → enable momentum rules
- Else if **spread cointegrated** and **|z| > 2** → stat-arb mean reversion
- Use **vol targeting** to size positions
- Halt trading if **max drawdown** > threshold

---

## Disclaimer (No Financial Advice)

This software is for **educational and informational purposes only** and is not financial, investment, or trading advice. You are solely responsible for any trading decisions you make.

### Limitation of Liability (Investment Losses)
**By using this repository and code, you acknowledge and agree that the author(s) are not responsible for any investment losses, damages, or other liabilities arising from the use of this tool.** Use at your own risk.

---

## License (MIT)

This project can be released under the MIT License.

1) Create a file named `LICENSE` in the repo root  
2) Paste the standard MIT License text  
3) Replace the placeholders with:
- **Year**
- **Your name**

You can find the canonical MIT text at choosealicense.com.

---

## Repo Tips

### Rename GitHub Repo
- On GitHub: **Repo → Settings → General → Repository name**
- Update any local remotes:
```bash
git remote set-url origin <NEW_REPO_URL>
```

---

## Contributing
PRs welcome. Keep changes small and include:
- metric name
- formula (or citation)
- unit tests or a reproducible example

---

## Notes on Data Availability
- Some metrics depend on **quotes** (bid/ask, sizes) or **options** data.
- Availability can vary by asset, subscription level, and market hours.

---

**Author:** (your name here)
