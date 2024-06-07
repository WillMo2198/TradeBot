from logging import basicConfig as loggingConfig
from logging import info as log
from logging import INFO
from os import remove
from os.path import isfile
from time import sleep
from bot import Bot

bot = Bot(paper=False)
ticker = 'T'

if isfile('output.log'):
    remove('output.log')

loggingConfig(filename='output.log', encoding='utf-8', level=INFO)


def out(prompt):
    print(prompt)
    log(prompt)


def main():
    pos_avg, neg_avg = bot.get_avgs(ticker, 2.67)
    out('Starting bot...' + bot.date_and_time())
    out('Current price: ' + str(bot.current_price(ticker)) + bot.date_and_time())
    out('Avg price: ' + str(bot.get_avgs(ticker, diff=False)))
    out('Pos Avg: ' + str(pos_avg) + bot.date_and_time())
    out('Neg Avg: ' + str(neg_avg) + bot.date_and_time())
    if len(bot.open_pos()) == 0:
        bot.reset([ticker])
    purchase_price = bot.get_stock_data(ticker, 'avg_entry_price')
    price = bot.get_stock_data(ticker, 'avg_entry_price')
    while True:
        if bot.market_open() and bot.current_price(ticker) != price:
            price = bot.current_price(ticker)
            out('Current difference: ' + str(price - purchase_price) + bot.date_and_time())
            if price-purchase_price > pos_avg and bot.open_pos(ticker):
                bot.order(ticker, 'SELL', bot.balance())
                sleep(5)
                out('Sell price: ' + str(bot.sell_price(ticker)) + bot.date_and_time())
                out('Quitting...' + bot.date_and_time())
                quit()
            elif bot.sell_price(ticker)-price < neg_avg and not bot.open_pos(ticker):
                bot.order(ticker, 'BUY', bot.balance())
                sleep(5)
                out('Buy price: '.format() + str(bot.get_stock_data(ticker, 'avg_entry_price')) + bot.date_and_time())
                out('Quitting...' + bot.date_and_time())
                quit()
            else:
                continue
        elif not bot.market_open():
            out('Market not open' + bot.date_and_time())
            out('Quitting...' + bot.date_and_time())
            quit()
        else:
            continue


if __name__ == '__main__':
    main()
