from typing import List, Dict, Any, Optional, Literal
from datetime import datetime, timedelta, date
from econometrics import *


class Watchdog:
    def __init__(self, stocks, is_paper=False):
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.common.exceptions import APIError
        from sp500 import sp500
        key = open('key.txt', 'r').read().replace('\n', '')
        secret = open('secret.txt', 'r').read().replace('\n', '')
        self.trade_client = TradingClient(key, secret, paper=is_paper)
        self.historical_client = StockHistoricalDataClient(key, secret)
        self.daily_stocks = stocks
        self.api_error = APIError
        self.avgs = self.avgs(load_json=False)

    def is_day_trader(self):
        account = self.trade_client.get_account()
        day_trades = int(account.daytrade_count)
        if day_trades < 3:
            return False
        return True

    def buying_power(self) -> float:
        """
        Returns available buying power
        """
        account = self.trade_client.get_account()
        return float(account.buying_power)

    def market_is_open(self) -> bool:
        """
        Returns True if the market is currently open, otherwise False
        """
        clock = self.trade_client.get_clock()
        return bool(clock.is_open)

    def holiday(self) -> bool:
        """
        Returns True if current day is a holiday, otherwise False
        """
        from alpaca.trading.requests import GetCalendarRequest
        today = date.today()
        req = GetCalendarRequest(start=today, end=today)
        calendar = self.trade_client.get_calendar(req)
        return len(calendar) == 0

    def current_price(self, symbol: str) -> float:
        """
        Returns the latest trade price for a stock symbol
        """
        from alpaca.data.historical.stock import StockLatestTradeRequest
        req = StockLatestTradeRequest(symbol_or_symbols=symbol)
        trade = self.historical_client.get_stock_latest_trade(req)[symbol]
        return float(trade.price)

    def hist(self, symbol: str, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """
        Get bars of stock 1 min interval, from start/end time
        """
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(1, TimeFrameUnit.Minute), start=start, end=end)
        bars = self.historical_client.get_stock_bars(req)[symbol]
        history = []
        for b in bars:
            volume = float(b.volume)
            open_ = float(b.open)
            close = float(b.close)
            high = float(b.high)
            low = float(b.low)
            history.append({"t": b.timestamp, "o": open_, "h": high, "l": low, "c": close, "v": volume})
        return history

    def open_positions(self) -> dict:
        open_position_symbols = {}
        positions = self.trade_client.get_all_positions()
        for i in positions:
            entry_price = float(i.avg_entry_price)
            qty = float(i.qty)
            open_position_symbols[i.symbol] = {'qty': qty, 'entry_price': entry_price}
        return open_position_symbols

    def position_qty(self, symbol: str, price: float, dollars=1: int) -> float:
        """
        return current position qty or amount that can be purchased given dollar amount
        """
        open_positions = self.open_positions()
        buy_power = self.buying_power()
        if symbol in open_positions.keys():
            pos = symbol
            return float(open_positions[pos]['qty'])
        else:
            return dollars / price

    def order(self, symbol: str, side, tif: Literal["day", "gtc"] = "day", qty: Optional[float] = 1):
        """
        Submit a market BUY/SELL.
        """
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import TimeInForce
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY if tif.lower() == "day" else TimeInForce.GTC,
        )
        self.trade_client.submit_order(order_data=order)

    def buy(self, ticker: str):
        from alpaca.trading import OrderSide
        try:
            price = self.current_price(ticker)
            self.order(ticker, OrderSide.BUY, qty=self.position_qty(ticker, price=price))
            self.avgs[ticker]['baseline'] = price
        except (self.api_error, KeyError) as e:
            print('Error {} buying {}...'.format(e, ticker))
            self.daily_stocks.remove(ticker)

    def sell(self, ticker: str):
        from alpaca.trading import OrderSide
        try:
            return self.order(ticker, OrderSide.SELL, qty=self.position_qty(ticker, price=None))
        except self.api_error:
            print('Error {} selling {}...'.format(e, ticker))
            pass

    def watch(self):
        while self.market_is_open():
            print('Watching...')
            for ticker in self.daily_stocks:
                position_is_open = ticker in self.open_positions().keys()
                current_price = self.current_price(ticker)
                """
                if position_is_open and [PUT YOUR SELL CONDITIONS HERE]:
                    print("sell {}".format(ticker))
                    self.sell(ticker)
                elif not position_is_open and [PUT YOUR BUY CONDITIONS HERE]:
                    print("buy {}".format(ticker))
                    self.buy(ticker)
                """


def main():
    from time import sleep
    dog = Watchdog()
    if not dog.holiday():
        while not dog.market_is_open():
            sleep(5)
        dog.watch(['AAPL', 'MSFT', 'TSLA', 'NVDA'])
    else:
        print("Today is a market holiday.")
    quit()


if __name__ == "__main__":
    main()
