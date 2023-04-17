"""
TODO: 
====
1. start_cash_balance with respect to security_age
2. Take care of timezones or run the script in EST zone only

3. Robinhood does not have the transaction date. we need to find a way to save and use date on which he security is bought
4. stocks removed from s&p500() that we still hold

5. portfolio.stock.date reflects the first time the security is bought even after multiple purchases?
"""
# yahooFinance provides last traded price as closing price, so we are good for trading. The resolution is pretty good. Nothing much to do here

import datetime
import json
from json import JSONEncoder
import os
import pprint

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

import yfinance as yf
import yahoo_fin.stock_info as si
import robin_stocks.robinhood as r
from dotenv import load_dotenv

pp = pprint.PrettyPrinter(indent=4)

tzname = 'America/New_York'

model_parameters = {
    "history": "7y",                        # history to consider std and mean of stock daily price change
    "buy_trigger": 2,                       # times standard deviation
    "sell_trigger": 1,                      # times the avg cost of the security to grow before we sell
    "security_age": 20,                     # number of days to hold the security before we cut the losses
    "lockin_gains_factor": 1000,            # times the orignal amount to grow before we lockin the gains.
    "mean_type": "+ve",                     # only consider stocks with +ve mean of ND. These stocks have been growing over the period of time
    "max_stocks_to_buy": 5,                 # number of stocks to buy at buy trigger. We can change this value to be more adaptive based on market cap of the security and other parameters.
    "prefer_beta": True,                    # favors stocks that has larger beta
    "above_beta_mean": True,                # only biy stocks above mean beta value of s&p

    # Display test results. Debugging Tools
    "print_final_portfolio": False,         # Prints the portfolio list at the end of each backtest iteration
    "plot_every_test_graph": True,          # Prints the model performance during the back end against s&P500
    "plot_summary_graph": True,             # prints the summary graph
    "dump_all_trades": False,               # dumps all sells at the end of the trade. Use it sparingly, with iteration set to 1
    "start_cash_balance": 10000,            # Model started with this cash balance
}

# subclass JSONEncoder
class DateTimeEncoder(JSONEncoder):
    #Override the default method
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()


class brokerage:
    def __init__(self):
        pass

    def cashbalance(self):
        pass

    def get_stocks(self):
        pass

    def buy_a_stock(self, ticker, quantity):
        pass

    def sell_a_stock(self, ticker):
        pass

    def get_current_stock_price(self, ticker):
        pass

    def get_open_price_of_stock(self, ticker):
        pass


class NextDay:
    """Class to implement an iterator
    of 1 increment"""

    def __init__(self, backtest_day):
        assert backtest_day < 0
        self.backtest_day = backtest_day

    def __iter__(self):
        return self

    def __next__(self):
        if self.backtest_day < 0:
            ret = self.backtest_day
            self.backtest_day += 1
            return ret
        else:
            raise StopIteration

    def current_day(self):
        return self.backtest_day


class LocalBrokerage(brokerage):
    def __init__(self, cash=10000, backtest_days=-1):
        self.backtest_days = backtest_days
        self.backtest = backtest_days != -1
        if os.path.exists("localbrokerage.json"):
            with open("localbrokerage.json", "r") as f:
                self.account = json.load(f)
        else:
            self.account = {'cash_balance': cash, 'portfolio': {}}

        self.stocks = si.tickers_sp500()
        self.stocks_ts = pd.DataFrame()
        for ticker in self.stocks:
            if False and os.path.exists(os.path.join("history", ticker)):
                data = pd.read_csv(os.path.join("history", ticker))
                data['Date'] = pd.to_datetime(data['Date']).tz_localize(None)
                data = data.set_index('Date')
            else:
                data = yf.Ticker(ticker).history(period=model_parameters["history"])
                data.to_csv(os.path.join("history", ticker))
            d = data.copy()[['Open', 'Close']]
            self.stocks_ts[ticker+"_Open"] = d['Open']
            self.stocks_ts[ticker+"_Close"] = d['Close']

            self.stocks_ts = self.stocks_ts.copy()
        self.stocks_ts.index = self.stocks_ts.index.tz_convert(tzname)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with open("localbrokerage.json", "w") as f:
             json.dump(self.account, f, indent=4, cls=DateTimeEncoder)

    def day_iter(self):
        if not self.backtest:
            raise Exception("LocalBrokerage is not in backtest mode")

        self.iter = NextDay(-self.backtest_days)
        return self.iter

    def backtest_day(self):
        if not self.backtest:
            raise Exception("LocalBrokerage is not in backtest mode")

        return self.backtest_days

    def cashbalance(self):
        return self.account['cash_balance']

    def get_stocks(self):
        return self.account['portfolio'].keys()

    def get_stock_holdings(self, ticker):
        if not self.has_stock(ticker):
            raise Exception("Stock %s not found" % ticker)

        return self.account.portfolio[ticker]

    def has_stock(self, ticker):
        return ticker in self.account['portfolio']

    def buy_a_stock(self, ticker, quantity, backtest_day=-1):
        close = self.get_current_stock_price(ticker, backtest_day=backtest_day)
   
        portfolio = self.account['portfolio']
        if close * quantity > self.account['cash_balance']:
            raise Exception("Not enough cash available")

        if ticker not in portfolio:
            portfolio[ticker] = []
        t = pd.Timestamp.now(tz=tzname)
        t = t.tz_convert(tzname)
        portfolio[ticker].append({"count": quantity,
                                  "cost": close,
                                  "date": str(t)
                                 })
        self.account['cash_balance'] -= quantity * close

        return

    def sell_a_stock(self, ticker, backtest_day=-1):
        close = self.get_current_stock_price(ticker, backtest_day=backtest_day)
   
        if ticker not in self.account['portfolio']:
            raise Exception("We don't own any of these stocks")

        portfolio = self.account['portfolio']
        # Find number of shares
        quantity = 0
        for s in portfolio[ticker]:
            quantity += s['count']

        self.account['cash_balance'] += quantity * close
        portfolio.pop(ticker)

        return

    def get_current_stock_price(self, ticker, backtest_day=-1):
        if self.backtest:
            close = self.stocks_ts[ticker+"_Close"][len(self.stocks_ts[ticker+"_Close"])+backtest_day]
        else:
            close = yf.Ticker(ticker).history("1d")['Close'][0]
        return close

    def get_open_price_of_stock(self, ticker, backtest_day=-1):
        if self.backtest:
            open = self.stocks_ts[ticker+"_Open"][len(self.stocks_ts[ticker+"_Open"])+backtest_day]
        else:
            open = yf.Ticker(ticker).history("1d")['Open'][0]

        return open

    def avg_cost_of_stock(self, ticker):
        if ticker not in self.account["portfolio"]:
            raise Exception("Stock %s not found in portfolio" % ticker)

        avg = 0
        for s in self.account["portfolio"][ticker]:
            avg += s['cost']
        return avg/len(self.account["portfolio"][ticker])
 
    def oldest_stock(self, ticker):
        if ticker not in self.account["portfolio"]:
            raise Exception("Stock %s not found in portfolio" % ticker)

        oldest = pd.Timestamp(self.account["portfolio"][ticker][0]['date']).tz_convert(tzname)
        for e in self.account["portfolio"][ticker]:
            if oldest > pd.Timestamp(e['date']).tz_convert(tzname):
                oldest = pd.Timestamp(e['date']).tz_convert(tzname)

        return oldest

    def netgain(self, ticker, backtest_day=-1):
        if ticker not in self.account["portfolio"]:
            raise Exception("Stock %s not found in portfolio" % ticker)
        netgain = 0
        for e in self.account["portfolio"][ticker]:
            netgain += e['count'] * (self.get_current_stock_price(ticker, backtest_day=backtest_day) -
                                     e['cost'])

        return netgain

    def calculate_networth(self, backtest_day=-1):
       networth = self.cashbalance()
       for ticker in self.get_stocks():
           for e in self.account["portfolio"][ticker]:
               networth += e['count'] * self.get_current_stock_price(ticker, backtest_day=backtest_day)
       return networth


class RobinhoodBrokerage(brokerage):
    def __init__(self):
        load_dotenv()
        r.login(username=os.environ['robin_username'],
                password=os.environ['robin_password'],
                expiresIn=86400,
                by_sms=True)
        positions = r.get_open_stock_positions()
        my_stocks = r.build_holdings()
        self.portfolio = {}
        for p in positions:
            ticker = r.get_symbol_by_url(p["instrument"])
            if float(my_stocks[ticker]['quantity']) > 0:
                self.portfolio[ticker] = {'count': float(my_stocks[ticker]['quantity']),
                                          'cost': float(my_stocks[ticker]['average_buy_price']),
                                          'date': pd.Timestamp(p['updated_at']).tz_convert(tzname)}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        r.logout()

    def cashbalance(self):
        return float(r.load_phoenix_account()['account_buying_power']['amount'])

    def get_stocks(self):
        return list(portfolio.keys())

    def has_stock(self, ticker):
        return ticker in self.portfolio

    def get_stock_holdings(self, ticker):
        if not self.has_stock(ticker):
            raise Exception("Stock %s not found" % ticker)
 
        return porfolio[ticker]

    def buy_a_stock(self, ticker, quantity, backtest_day=-1):
        cost = self.get_current_stock_price(ticker)
        if self.cashbalance() > quantity * cost:
            t = pd.Timestamp.now(tz=tzname)
            t = t.tz_convert(tzname)
            self.portfolio[ticker] = {'count': quantity,
                                      'cost': cost,
                                      'date': t
                                     }
          
        print(self.portfolio)
        #raise Exception("Not Implemented")

    def sell_a_stock(self, ticker, backtest_day=-1):
        assert self.has_stock(ticker)
 
        self.portfolio.pop(ticker) 
        #raise Exception("Not Implemented")

    def get_current_stock_price(self, ticker, backtest_day=-1):
        close = yf.Ticker(ticker).history("1d")['Close'][0]
   
        return close

    def get_open_price_of_stock(self, ticker, backtest_day=-1):
        assert backtest_day == -1

        open = yf.Ticker(ticker).history("1d")['Open'][0]

        return open

    def avg_cost_of_stock(self, ticker):
        if not self.has_stock(ticker):
            raise Exception("Stock %s not found" % ticker)

        return self.portfolio[ticker]['cost']

    def oldest_stock(self, ticker):
        if not self.has_stock(ticker):
            raise Exception("Stock %s not found" % ticker)

        return self.portfolio[ticker]['date']

    def netgain(self, ticker, backtest_day=-1):
        if not self.has_stock(ticker):
            raise Exception("Stock %s not found" % ticker)

        holdings = r.build_holdings()[ticker]
        return float(holdings['equity_change'])

    def calculate_networth(self, backtest_day=-1):
       networth = self.cashbalance()

       for ticker, value in r.build_holdings().items():
           networth += float(value['equity'])

       return networth


class Model:
    # Read the last few years of stocks and indices to calculate betas and standard deviations
    def __init__(self, brokerage):
        self.brokerage = brokerage
        self.stocks = si.tickers_sp500()
        self.indices = ['^IXIC', '^GSPC', '^DJI']
        self.stocks_ts = pd.DataFrame()
        self.stocks_spread = pd.DataFrame()
        self.indices_ts = pd.DataFrame()

        self.security_profit = pd.DataFrame(columns=['beta', 'days', 'profit'])
        self.security_loss = pd.DataFrame(columns=['beta', 'loss'])

        self.price_movement = []
        self.cash_inhand = []

        for idx in self.indices:
            data = yf.Ticker(idx).history(period=model_parameters["history"])
            d = data.copy()[['Open', 'Close']]
            self.indices_ts[idx+"_Open"] = d['Open']
            self.indices_ts[idx+"_Close"] = d['Close']

        self.betas = pd.read_csv("betas-sp500.csv")
        self.betas = self.betas.drop(['Unnamed: 0'], axis=1)
        self.betas = self.betas.rename(columns={"Recent": "Beta"})
        self.betas = self.betas.set_index('Ticker')

        for ticker in self.stocks:
            if False and os.path.exists(os.path.join("history", ticker)):
                data = pd.read_csv(os.path.join("history", ticker))
                data['Date'] = pd.to_datetime(data['Date']).tz_convert(tzname)
                data = data.set_index('Date')
            else:
                data = yf.Ticker(ticker).history(period=model_parameters["history"])
                data.to_csv(os.path.join("history", ticker))
            d = data.copy()[['Open', 'Close']]
            self.stocks_ts[ticker+"_Open"] = d['Open']
            self.stocks_ts[ticker+"_Close"] = d['Close']
            self.stocks_spread[ticker] = (d['Close'] - d['Open']) * 100/d['Open']

            self.stocks_ts = self.stocks_ts.copy()
            self.stocks_spread = self.stocks_spread.copy()

        self.indices_ts.index = self.indices_ts.index.tz_convert(tzname)
        self.stocks_ts.index = self.stocks_ts.index.tz_convert(tzname)
        self.std = self.stocks_spread.describe().loc['std']
        self.mean = self.stocks_spread.describe().loc['mean']
        self.std = pd.DataFrame.from_dict(self.std)
        self.std.columns = ['std',]
        self.betas_mean = self.betas.describe()['Beta']['mean']

        self.mean = pd.DataFrame.from_dict(self.mean)
        self.mean.columns = ['mean',]


    def analyze_a_stock(self, ticker):
        data = yf.Ticker(ticker).history(period=model_parameters["history"])
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'])])
        fig.update_layout(title=ticker,
                          yaxis_title=ticker +' Stock',
                          shapes = [dict(x0='2022-12-09', x1='2022-12-09', y0=0, y1=1, xref='x', yref='paper',
                          line_width=2)],
                          annotations=[dict( x='2022-12-09', y=0.05, xref='x', yref='paper',
                          showarrow=False, xanchor='left', text='Increase Period Begins')])

        fig.show()

        self.stocks_spread['Date'] = self.stocks_spread.index
        fig = px.line(self.stocks_spread, x="Date", y=ticker, title="Stocks Daily Price change", markers=True) 
        fig.show()

        x = pd.Series(self.stocks_spread[ticker])
        ax = x.plot.kde(figsize=(30,8))

    def analyze_std_mean(self):
        fig = px.line(self.std, title="Daily Stock Price change", markers=True)
        fig.show()
        fig = px.line(self.betas, title="Stocks Betas", markers=True)
        fig.show()

    def _getsellbuy(self, backtest_day=-1):
        todays_delta = {}
        for ticker in self.stocks:
            open = self.brokerage.get_open_price_of_stock(ticker, backtest_day=backtest_day)
            close = self.brokerage.get_current_stock_price(ticker, backtest_day=backtest_day)
            todays_delta[ticker] = (close - open) * 100/open


        todays_delta = pd.Series(todays_delta)
        latest_diff = pd.DataFrame.from_dict(todays_delta)
        latest_diff.columns = ['diff',]
        latest_diff['std'] = self.std['std']

        # Find stocks that are in buy range
        latest_diff['buy'] = np.where(((model_parameters["mean_type"] != "+ve" or self.mean['mean'] > 0) & (latest_diff['diff'] < 0) &
                                  (model_parameters["above_beta_mean"] == False or self.betas_mean >= self.betas['Beta']) &
                                  (latest_diff['diff'] < -model_parameters["buy_trigger"] * self.std['std'])), True, False)

        latest_diff['beta'] = self.betas['Beta']

        # Find stocks that are in sell range
        latest_diff['sell'] = False
        index = self.stocks_ts.index[backtest_day]
        for ticker in self.stocks:
            if not self.brokerage.has_stock(ticker):
                continue

            oldest = self.brokerage.oldest_stock(ticker)
            t = index - oldest

            avg = self.brokerage.avg_cost_of_stock(ticker)

            netgain = self.brokerage.netgain(ticker, backtest_day=backtest_day)
            if avg + model_parameters["sell_trigger"] * self.std['std'][ticker] * avg / 100 <= self.brokerage.get_current_stock_price(ticker, backtest_day=backtest_day):
                # If the avg cost of the security has grown more than std
                # print(s, avg, portfolio[s]['costs'], std['std'][s+"_Diff"], stocks_ts.loc[index, s+"_Close"])

                if model_parameters["dump_all_trades"]:
                    print("Selling %s (beta %f) at profit. closing price %f. netgain %f days %d" %
                          (s, betas['Beta'][s], stocks_ts.loc[index, s+"_Close"], netgain, t.days),
                          self.get_stock_holdings(ticker))
                self.security_profit.loc[index] = {'beta':self.betas['Beta'][ticker],
                                              'days': t.days, 'profit': netgain}
                latest_diff.at[ticker, 'sell'] = True
            elif t.days > model_parameters["security_age"]:
                # if the security has aged for certain days, cut the losses
                if model_parameters["dump_all_trades"]:
                    print("Dumping %s(beta %f) because of age. closing price %f. netgain %f" %
                          (ticker, self.betas['Beta'][ticker], netgain),
                          self.get_stock_holdings(ticker))
                if netgain > 0:
                        self.security_profit.loc[len(self.security_profit.index)] = {'beta':self.betas['Beta'][ticker],
                                                                           'days':t.days, 'profit':netgain}
                else:
                        self.security_loss.loc[len(self.security_loss.index)] = {'beta':self.betas['Beta'][ticker],
                                                                       'loss':netgain}
                latest_diff.at[ticker, 'sell'] = True
  
        sell_stocks = latest_diff[latest_diff['sell']]
        buy_stocks = latest_diff[latest_diff['buy']]
        if model_parameters["prefer_beta"]:
            buy_stocks = buy_stocks.sort_values('beta',ascending=False)
        return sell_stocks, buy_stocks

    def do_todays_trade(self, backtest_day=-1):

        sell, buy = self._getsellbuy(backtest_day=backtest_day)

        # process the stocks that are marked sell
        for st in sell.iterrows():
            ticker = st[0].split('_')[0]
            self.brokerage.sell_a_stock(ticker, backtest_day=backtest_day)

        # buy stocks that are marked by. We are buying max_stocks_to_buy number of stocks
        # TODO: The number of stocks to be must be adaptive. Will come up with some
        # algorithm based on:
        # 1. Market capitalization
        # 2. Beta
        # and other criteria
        # The goal is to put the money to work
        for st in buy.iterrows():
            ticker = st[0].split('_')[0]
            if self.brokerage.cashbalance() > self.brokerage.get_current_stock_price(ticker, backtest_day=backtest_day) * model_parameters["max_stocks_to_buy"]:
                self.brokerage.buy_a_stock(ticker, model_parameters["max_stocks_to_buy"], backtest_day=backtest_day)

        # lock in the gains after 10% increase of networth
        nw = self.brokerage.calculate_networth(backtest_day=backtest_day)
        self.price_movement.append(nw)
        self.cash_inhand.append(self.brokerage.cashbalance())
        if nw > model_parameters["start_cash_balance"] * model_parameters["lockin_gains_factor"]:
            #print(backtest_start_date, self.brokerage.calculate_networth(backtest_start_date), current_account, portfolio)
            for ticker in self.brokerage.get_stocks():
                self.brokerage.sell_a_stock(ticker, backtest_day=backtest_day)
        return 

    def dump_profit_loss_distribution(self):
        if model_parameters["dump_all_trades"]:
            print("Profit Distribution")
            print("===================")
            print(self.stocks_profit.describe())
            print()
            print("Loss Distribution")
            print("=================")
            print(self.stocks_loss.describe())
        return

    def do_backtest(self):
        for i in self.brokerage.day_iter():
            self.do_todays_trade(backtest_day=i)

        networth = self.brokerage.calculate_networth()

        # normalize s&p500 for starting balance
        idx_list = list(self.indices_ts.loc[self.indices_ts.index[-self.brokerage.backtest_day():], "^GSPC_Close"] *
                        model_parameters["start_cash_balance"]/self.indices_ts.loc[self.indices_ts.index[-self.brokerage.backtest_day()], "^GSPC_Close"])
        pm_pct = (self.price_movement[-1]-model_parameters["start_cash_balance"]) * 100/model_parameters["start_cash_balance"]

        d = pd.DataFrame({'Portfolio_Performance':self.price_movement,'SP_Performance':idx_list, "Cash_In_Hand": self.cash_inhand})
        d.index = self.indices_ts.index[-self.brokerage.backtest_day():]
        idx_pct = (idx_list[-1] - idx_list[0])*100/idx_list[0]
        print("Total networth: %d (Cash %d) after going back %d days (%s)" % 
              (networth, self.brokerage.cashbalance(),
               self.brokerage.backtest_day(),
               self.stocks_ts.index[-self.brokerage.backtest_day()]))
        print("Model (%f)%% vs S&P Performance (%f)%%" % (pm_pct, idx_pct))
        if model_parameters["print_final_portfolio"]:
            pp.pprint(portfolio)
        if model_parameters["plot_every_test_graph"]:
            fig = px.line(d, title="Model (%f)%% vs S&P Performance (%f)%% starting at %s" % 
                          (pm_pct, idx_pct, self.stocks_ts.index[-self.brokerage.backtest_day()]),
                           markers=True)
            fig.show()


"""
with LocalBrokerage(cash=10000) as lb:
    if lb.cashbalance() > lb.get_current_stock_price('aapl') * 5:
        lb.buy_a_stock('aapl', 5)
    else:
        print("Don't have enough cash to buy AAPL")
    if lb.cashbalance() > lb.get_current_stock_price('msft') * 5:
        lb.buy_a_stock('msft', 5)
    else:
        print("Don't have enough cash to buy MSFT")
    if lb.cashbalance() > lb.get_current_stock_price('cost') * 5:
        lb.buy_a_stock('cost', 5)
    else:
        print("Don't have enough cash to buy COST")

    if 'cost' in lb.get_stocks():
        lb.sell_a_stock('cost')

with LocalBrokerage(cash=10000) as lb:
    pp.pprint(lb.account)

with LocalBrokerage(cash=10000, backtest_days=150) as lb:
    m = Model(lb)
    #m.analyze_a_stock('AAPL')
    #m.analyze_std_mean()
    m.do_backtest()

"""
with LocalBrokerage(cash=10000) as lb:
    m = Model(lb)
    #m.analyze_a_stock('AAPL')
    #m.analyze_std_mean()
    m.do_todays_trade()
"""
with RobinhoodBrokerage() as lb:
    m = Model(lb)
    #m.analyze_a_stock('AAPL')
    #m.analyze_std_mean()
    m.do_todays_trade()
"""
