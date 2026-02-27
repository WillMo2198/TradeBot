# Watch Dog 🐶📈

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Alpaca](https://img.shields.io/badge/Broker-Alpaca-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Status](https://img.shields.io/badge/status-Experimental-orange)

Automated stock monitoring + conditional trade execution built on
Alpaca.

Watch Dog is a Python trading bot framework that:

-   Watches a configurable list of stock tickers
-   Pulls real-time prices and 1-minute historical bars
-   Computes 7-day baselines and deviation averages
-   Lets you inject custom buy/sell logic
-   Supports paper trading and live execution

⚠️ **Disclaimer:** Educational use only. Trading involves risk. Use
paper trading before live capital.

⚠️ **Financial Risk Warning:** Trading stocks, options, or other financial instruments involves substantial risk.
You may lose all or more than your invested capital. This project does not provide financial advice. Use at your own risk.

------------------------------------------------------------------------

## Core Features

-   Alpaca Trading + Market Data integration
-   Paper/live trading toggle
-   Tradeability + fractionable checks
-   Market open + holiday detection
-   Baseline normalization engine
-   Positive/negative deviation modeling
-   Market order execution wrapper
-   Extensible rule injection in `watch()` loop

------------------------------------------------------------------------

## How It Works

For each ticker:

1.  Determine baseline:
    -   Open position entry price OR
    -   7-day average close
2.  Normalize historical prices vs baseline
3.  Compute:
    -   `pos_avg` (average positive deviation)
    -   `neg_avg` (average negative deviation)
4.  Compare current deviation to averages
5.  Execute your defined buy/sell logic

You control strategy logic.\
Watch Dog handles market plumbing.

------------------------------------------------------------------------

## Installation

    python -m venv .venv
    source .venv/bin/activate
    pip install alpaca-py

------------------------------------------------------------------------

## Setup

Create:

-   `key.txt`
-   `secret.txt`

Add to `.gitignore`:

    key.txt
    secret.txt
    avgs.json
    .venv/
    __pycache__/

Initialize:

    dog = Watchdog(stocks=["AAPL","MSFT","TSLA","NVDA"], is_paper=True)

------------------------------------------------------------------------

## Running

    python main.py

Behavior:

-   Exits on market holiday
-   Waits for open
-   Loops while open
-   Evaluates your buy/sell rules

------------------------------------------------------------------------

## Injecting Strategy Logic

Inside `watch()`:

    if SELL_CONDITION:
        self.sell(ticker)
    elif BUY_CONDITION:
        self.buy(ticker)

Available variables:

-   position_is_open
-   current_price
-   current_normal_price
-   pos_avg
-   neg_avg
-   tradeable

Example (mean reversion demo):

    if position_is_open and self.can_sell(ticker) and current_normal_price > pos_avg:
        self.sell(ticker)
    elif (not position_is_open) and tradeable and current_normal_price < neg_avg:
        self.buy(ticker)

------------------------------------------------------------------------

## Future Improvements

-   Backtesting engine
-   Risk management module
-   Trade journaling (SQLite/CSV)
-   Dashboard UI
-   Strategy class abstraction
-   Alert integrations

------------------------------------------------------------------------

## License

MIT (recommended)

------------------------------------------------------------------------

## Philosophy

Build your rules.\
Let Watch Dog execute.
