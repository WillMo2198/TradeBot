from typing import List, Dict, Any, Optional, Literal
from datetime import datetime, timedelta, date


class Watchdog:
    def __init__(self, is_paper=False):
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.common.exceptions import APIError
        from sp500 import sp500
        key = open('key.txt', 'r').read().replace('\n', '')
        secret = open('secret.txt', 'r').read().replace('\n', '')
        self.trade_client = TradingClient(key, secret, paper=is_paper)
        self.historical_client = StockHistoricalDataClient(key, secret)
        self.daily_stocks = sp500.copy()
        self.error = APIError
        self.avgs = self.avgs()
        self.delta = 0.40

    def top_penny_stocks(self, top_n: int = 15, max_universe: int = 3000, max_price: float = 5.0) -> list:
        """
        Returns tickers of the top `top_n` most-traded penny stocks (price < max_price),
        ranked by snapshot daily volume
        """
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass, AssetStatus
        from alpaca.data.requests import StockSnapshotRequest
        assets = self.trade_client.get_all_assets(GetAssetsRequest(
            status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY
        ))
        symbols = [a.symbol for a in assets if a.tradable][:max_universe]
        scored = []
        for i in range(0, len(symbols), 200):
            batch = symbols[i:i + 200]
            snaps = self.historical_client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=batch))
            for sym, s in snaps.items():
                lt = getattr(s, "latest_trade", None)
                db = getattr(s, "daily_bar", None)
                if not lt or not db:
                    continue
                price = float(getattr(lt, "price", 0.0) or 0.0)
                vol = int(getattr(db, "volume", 0) or 0)
                if 0 < price < max_price and vol > 0:
                    scored.append((vol, sym))
        scored.sort(reverse=True)
        return list(sym for _, sym in scored[:top_n])

    def is_tradeable(self, symbol):
        try:
            asset = self.trade_client.get_asset(symbol)
        except (self.error, KeyError):
            return False
        return bool(asset.fractionable) and bool(asset.tradable)

    @staticmethod
    def option_ticker(ticker: str, cp: str, strike: float) -> str:
        d = (datetime.now() + timedelta(days=10)).strftime("%Y%m%d")
        yy, mm, dd = d[2:4], d[4:6], d[6:8]
        strike_int = int(round(float(strike) * 1000))
        strike_str = f"{strike_int:08d}"
        return f"{ticker}{yy}{mm}{dd}{cp}{strike_str}"

    def submit_option(self, option_symbol: str, qty: int, side: str):
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
        order = MarketOrderRequest(
            symbol=option_symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        return self.trade_client.submit_order(order)

    def daily_stocks(self) -> List[str]:
        from sp500 import sp500
        symbols = sp500.copy()
        daily_stocks = []
        for sym in symbols:
            if self.is_tradeable(sym):
                daily_stocks.append(sym)
        return daily_stocks

    def is_day_trader(self):
        account = self.trade_client.get_account()
        day_trades = int(account.daytrade_count)
        if day_trades < 3:
            return False
        return True

    def avgs(self, json_file='avgs.json'):
        from json import dump, load
        try:
            with open(json_file, 'r') as f:
                return load(f)
        except FileNotFoundError:
            print('Generating daily averages...')
            today = datetime.today() - timedelta(days=1)
            one_week_ago = today - timedelta(days=7)
            averages = {}
            open_pos = self.open_positions()
            for ticker in self.daily_stocks:
                try:
                    hist = self.hist(ticker, start=one_week_ago, end=today)
                    if ticker not in open_pos.keys():
                        avg_price = self.seven_day_average_price(ticker)
                    else:
                        avg_price = open_pos[ticker]['entry_price']
                    prices = self.normal_hist(hist, baseline=avg_price)
                    positive_avg, negative_avg = self.positive_negative_avg(prices)
                    averages[ticker] = {'pos_avg': positive_avg, 'neg_avg': negative_avg, 'baseline': avg_price}
                except (self.error, KeyError):
                    self.daily_stocks.remove(ticker)
                    continue
            with open(json_file, 'w') as f:
                dump(averages, f)
                print('Stock data dumped to {}'.format(json_file))
                return averages

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

    def seven_day_average_price(self, symbol: str) -> float:
        from statistics import mean
        end = datetime.today() - timedelta(days=1)
        start = end - timedelta(weeks=1)
        bars = self.hist(symbol, start, end)
        prices = []
        for b in bars:
            prices.append(b["c"])
        return mean(prices)

    def can_sell(self, symbol: str) -> float:
        orders = self.trade_client.get_orders()
        for o in orders:
            if (o.side == "buy" and o.filled_at and o.symbol == symbol) or self.is_day_trader():
                return False
        return True

    @staticmethod
    def normal_hist(bars: List[Dict[str, Any]], baseline: Optional[float]) -> List:
        ys = []
        for p in bars:
            open_ = p.get("h")
            close = p.get("l")
            if open_ is None or close is None:
                continue
            ys.append(float(open_) - float(baseline))
            ys.append(float(close) - float(baseline))
        return ys

    @staticmethod
    def strike(option_symbol: str) -> float:
        """
        Extract strike price from an OCC option symbol.
        Example: AAPL240621C00180000 -> 180.0
        """
        # Last 8 characters encode strike * 1000
        strike_part = option_symbol[-8:]
        return int(strike_part) / 1000.0

    def open_positions(self) -> dict:
        open_position_symbols = {}
        positions = self.trade_client.get_all_positions()
        for i in positions:
            entry_price = float(i.avg_entry_price)
            qty = float(i.qty)
            open_position_symbols[i.symbol] = {'qty': qty, 'entry_price': entry_price}
        return open_position_symbols

    def position_qty(self, symbol: str, price) -> float:
        """
        return current position qty or amount that can be purchased
        """
        open_positions = self.open_positions()
        if symbol in open_positions.keys():
            pos = symbol
            return float(open_positions[pos]['qty'])
        else:
            return (self.buying_power() / len(self.daily_stocks)) / price

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

    def buy(self, ticker: str, price: float, options=False):
        from alpaca.trading import OrderSide
        if not options:
            try:
                self.order(ticker, OrderSide.BUY, qty=self.position_qty(ticker, price=price))
            except (self.error, KeyError) as e:
                print('Error {} buying {}...'.format(e, ticker))
                self.daily_stocks.remove(ticker)
        else:
            from json import load, dump
            strike = str(round(price+self.avgs[ticker]['pos_avg'], 0))
            opt_ticker = ''
            for contract in self.options_contracts(ticker):
                print(contract)

            self.submit_option(opt_ticker, 1, 'buy')
            try:
                with open('options.json', 'r') as f:
                    data = load(f)
            except FileNotFoundError:
                data = {}
            with open('options.json') as f:
                data[ticker] = opt_ticker
                dump(data, f)

    def sell(self, ticker: str, price: Optional[float], options=False):
        from alpaca.trading import OrderSide
        if not options:
            self.order(ticker, OrderSide.SELL, qty=self.position_qty(ticker, price=None))
        else:
            if len(self.options_contract(ticker)) > 0:
                from json import load
                with open('options.json', 'r') as f:
                    opt_tickers = load(f)
                opt_ticker = opt_tickers[ticker]
                strike = self.strike(opt_ticker)
                if price > strike:
                    self.submit_option(self.avgs[ticker]['opt_ticker'], 1, 'sell')

    def options_contracts(self, ticker: str) -> List[str]:
        """
        Returns a list of OCC option symbols for all unexpired contracts of ticker
        """
        from alpaca.trading.requests import GetOptionContractsRequest
        from alpaca.trading.enums import AssetStatus
        req = GetOptionContractsRequest(
            underlying_symbols=[ticker],
            status=AssetStatus.ACTIVE,
            expiration_date_gte=date.today(),
        )
        contracts = self.trade_client.get_option_contracts(req)
        return [sym for sym, _ in contracts]

    @staticmethod
    def positive_negative_avg(normalized_prices: List):
        from statistics import mean, StatisticsError
        pos = []
        neg = []
        for price in normalized_prices:
            if price >= 0:
                pos.append(price)
            else:
                neg.append(price)
        try:
            pos_avg = mean(pos)
        except StatisticsError:
            pos_avg = 0
        try:
            neg_avg = mean(neg)
        except StatisticsError:
            neg_avg = 0
        if pos_avg == 0:
            pos_avg = -neg_avg
        elif neg_avg == 0:
            neg_avg = -pos_avg
        return pos_avg, neg_avg

    def find_option_contract_symbol(self, ticker: str, expiry_yyyymmdd: str, cp: str, strike: float):
        from alpaca.trading.enums import ContractType
        from alpaca.trading import GetOptionContractsRequest
        from alpaca.trading import AssetStatus

        exp = expiry_yyyymmdd.replace("-", "")
        exp_date = date(int(exp[:4]), int(exp[4:6]), int(exp[6:8]))

        req = GetOptionContractsRequest(
            underlying_symbols=[ticker],
            status=AssetStatus.ACTIVE,
            type=ContractType.CALL if cp.upper() == "C" else ContractType.PUT,
            expiration_date_gte=exp_date,
            expiration_date_lte=exp_date,
            strike_price_gte=str(strike-0.0001),
            strike_price_lte=str(strike+0.0001),
        )
        contracts = self.trade_client.get_option_contracts(req)
        if not contracts:
            raise ValueError("Contract not found on Alpaca (expiry/strike/type may not exist).")
        return contracts



    def debug_account(self):
        c = self.trade_client
        acct = c.get_account()
        return {
            "status": str(acct.status),
            "trading_blocked": bool(getattr(acct, "trading_blocked", False)),
            "options_level": str(getattr(acct, "options_trading_level", None)),
        }

    def unexpired_option_contracts(self, ticker: str, days_out: int = 120):
        from alpaca.trading.requests import GetOptionContractsRequest
        from alpaca.trading.enums import AssetStatus
        c = self.trade_client
        token = None
        out = []

        while True:
            req = GetOptionContractsRequest(
                underlying_symbols=[ticker],
                status=AssetStatus.ACTIVE,
                expiration_date_gte=date.today(),
                expiration_date_lte=date.today() + timedelta(days=days_out),
                page_token=token
            )
            resp = c.get_option_contracts(req)
            contracts = resp.option_contracts
            out.extend([oc.symbol for oc in contracts])

            token = resp.next_page_token
            if not token:
                break
        return out

    def watch(self, options=False):
        while self.market_is_open():
            for ticker in self.daily_stocks:
                position_is_open = ticker in self.open_positions().keys()
                current_price = self.current_price(ticker)
                print("{}'s current price: {}".format(ticker, current_price))
                try:
                    ticker_data = self.avgs[ticker]
                except (KeyError, self.error):
                    self.daily_stocks.remove(ticker)
                    continue
                current_normal_price = current_price - ticker_data['baseline']
                if current_normal_price >= ticker_data['pos_avg'] and position_is_open:
                    print("sell {}".format(ticker))
                    self.sell(ticker, current_price, options)
                elif not position_is_open and current_normal_price <= ticker_data['neg_avg'] and self.is_tradeable(ticker):
                    print("buy {}".format(ticker))
                    self.buy(ticker, current_price, options)


def main():
    from time import sleep
    dog = Watchdog()
    ticker = 'AAPL'
    print(dog.unexpired_option_contracts(ticker))
    if not dog.holiday():
        while not dog.market_is_open():
            sleep(5)
        print('Watching...')
        #dog.watch()
    else:
        print("Today is a market holiday!")


if __name__ == "__main__":
    main()
