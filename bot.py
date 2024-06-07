from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import GetOrdersRequest
from alpaca.common import exceptions
from datetime import datetime
from lxml.html import fromstring
from requests import get
from time import sleep
from yfinance import Ticker
from logging import basicConfig as loggingConfig
from logging import info as log
from logging import INFO
from holidays import financial_holidays
from main import out


class Bot:
    def __init__(self, paper=True):
        key = open('key.txt', 'r').read()
        secret = open('secret.txt', 'r').read()
        self.trading_client = TradingClient(key, secret, paper=paper)

    def balance(self):
        account = dict(self.trading_client.get_account())
        balance = float(account['buying_power'])
        return float(balance)

    def get_stock_data(self, stock, key):
        open_pos = list(self.trading_client.get_all_positions())
        for pos in range(len(open_pos)):
            open_pos[pos] = dict(open_pos[pos])
            if open_pos[pos]['symbol'] == stock:
                try:
                    return float(open_pos[pos][key])
                except KeyError:
                    raise KeyError('Invalid data type')
                except TypeError:
                    return open_pos[pos][key]
        raise KeyError('No such stock')

    def open_pos(self, stock=''):
        stocks = []
        open_pos = list(self.trading_client.get_all_positions())
        for pos in range(len(open_pos)):
            stocks.append(open_pos[pos].symbol)
        if stock in stocks:
            return True
        elif stock == '':
            return stocks
        elif stock not in stocks:
            return False

    def sell_price(self, stock):
        request_params = GetOrdersRequest(
            status='closed',
            symbol=stock,
            direction='desc'
        )
        orders = list(self.trading_client.get_orders(request_params))
        for i in range(len(orders)):
            order = dict(orders[i])
            order_type = str(order['side'])
            status = str(order['status'])
            if order_type == 'OrderSide.SELL' and status == 'OrderStatus.FILLED':
                return float(order['filled_avg_price'])
            else:
                pass
        return False

    def order(self, stock, order_type, buying_power):
        loggingConfig(filename='output.log', encoding='utf-8', level=INFO)
        if order_type == 'BUY':
            out('Buying...' + self.date_and_time())
            order_data = MarketOrderRequest(
                symbol=stock,
                qty=int(buying_power / self.current_price(stock)) - 1,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
            )
            sold = False
            while not sold:
                try:
                    self.trading_client.submit_order(order_data)
                    sold = True
                except exceptions:
                    sleep(.5)
                    self.trading_client.submit_order(order_data)
                    sold = True
        elif order_type == 'SELL':
            out('Selling...' + self.date_and_time())
            order_data = MarketOrderRequest(
                symbol=stock,
                qty=self.get_stock_data(stock, 'qty_available'),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            self.trading_client.submit_order(order_data)
        else:
            raise Exception("Invalid order type, BUY or SELL")
        out('{0} Order complete!'.format(order_type) + self.date_and_time())

    def reset(self, stocks):
        loggingConfig(filename='output.log', encoding='utf-8', level=INFO)
        out('Resetting...' + self.date_and_time())
        while True:
            if self.market_open():
                out('Closing all positions')
                self.trading_client.close_all_positions(True)
                balance = self.balance() / len(stocks)
                for stock in stocks:
                    if self.current_price(stock) < self.get_avgs(stock, diff=False):
                        self.order(stock, 'BUY', buying_power=balance)
                        out('Quitting...')
                        quit()
                sleep(5)
            elif not self.market_open():
                out('Market not open. ' + self.date_and_time())
                out('Quitting...' + self.date_and_time())
                quit()

    @staticmethod
    def get_avgs(stock, mult=1, diff=True):
        hist = list(Ticker(stock).history(interval='1m', period='2d')['Close'])
        hist_diff = []
        pos_hist_diff = []
        neg_hist_diff = []
        if diff:
            for i in range(len(hist) + 1):
                if i + 1 >= len(hist):
                    break
                hist_diff.append(hist[i] - hist[i + 1])
            temp_hist_diff = []
            for i in hist_diff:
                if i != 0.:
                    temp_hist_diff.append(i)
            for i in temp_hist_diff:
                if i > 0.:
                    pos_hist_diff.append(i)
                elif i < 0.:
                    neg_hist_diff.append(-i)
            return (sum(pos_hist_diff) / len(pos_hist_diff)) * mult, (sum(neg_hist_diff) / len(neg_hist_diff)) * -mult
        elif not diff:
            return sum(hist) / len(hist)

    @staticmethod
    def market_open():
        day = datetime.now().weekday()
        date = datetime.today()
        time = int(datetime.now().strftime("%H%M"))
        open_time = 930
        close_time = 1600
        weekdays = [0, 1, 2, 3, 4]
        if open_time <= time <= close_time and day in weekdays and date not in financial_holidays('NYSE'):
            return True
        else:
            return False

    @staticmethod
    def current_price(stock):
        http_headers = {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,/;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "en-GB,en;q=0.9,en-US;q=0.8,ml;q=0.7",
            "cache-control": "max-age=0",
            "dnt": "1",
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "none",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36"
        }
        summary_url = "https://finance.yahoo.com/quote/%s?p=%s" % (stock, stock)
        while True:
            summary_response = get(summary_url, verify=True, headers=http_headers, timeout=30)
            summary_parser = fromstring(summary_response.text)
            try:
                price = \
                    summary_parser.xpath('//*[@id="quote-header-info"]/div[3]/div[1]/div[1]/fin-streamer[1]/text()')[0]
                return float(price)
            except IndexError:
                sleep(5)
                continue

    @staticmethod
    def date_and_time():
        return datetime.now().strftime(' %Y-%m-%d %H:%M:%S')
