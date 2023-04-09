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

pp = pprint.PrettyPrinter(indent=4)


model_parameters = {
    "history": "7y",                        # history to consider std and mean of stock daily price change
    "buy_trigger": 2,                       # times standard deviation
    "sell_trigger": 1,                      # times the avg cost of the security to grow before we sell
    "security_age": 90,                     # number of days to hold the security before we cut the losses
    "lockin_gains_factor": 1000,            # times the orignal amount to grow before we lockin the gains.
    "mean_type": "+ve",                     # only consider stocks with +ve mean of ND. These stocks have been growing over the period of time
    "max_stocks_to_buy": 5,                 # number of stocks to buy at buy trigger. We can change this value to be more adaptive based on market cap of the security and other parameters.
    "prefer_beta": True,                    # favors stocks that has larger beta
    "above_beta_mean": False,               # only biy stocks above mean beta value of s&p

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


class LocalBrokerage(brokerage):
    def __init__(self, cash=10000):
        if os.path.exists("localbrokerage.json"):
            with open("localbrokerage.json", "r") as f:
                self.account = json.load(f)
        else:
            self.account = {'cash_balance': cash, 'portfolio': {}}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with open("localbrokerage.json", "w") as f:
             json.dump(self.account, f, indent=4, cls=DateTimeEncoder)

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

    def buy_a_stock(self, ticker, quantity):
        close = yf.Ticker(ticker).history(period="1d")['Close'][0]
   
        portfolio = self.account['portfolio']
        if close * quantity > self.account['cash_balance']:
            raise Exception("Not enough cash available")

        if ticker not in portfolio:
            portfolio[ticker] = []
        portfolio[ticker].append({"count": quantity,
                                  "cost": close,
                                  "date": datetime.datetime.now()})
        self.account['cash_balance'] -= quantity * close

        return

    def sell_a_stock(self, ticker):
        close = yf.Ticker(ticker).history(period="1d")['Close'][0]
   
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

    def get_current_stock_price(self, ticker):
        close = yf.Ticker(ticker).history(period="1d")['Close'][0]
        return close

    def get_open_price_of_stock(self, ticker):
        open = yf.Ticker(ticker).history(period="1d")['Open'][0]
        return open

    def avg_cost_of_stock(self, ticker):
        if s not in self.account.portfolio:
            raise Exception("Stock %s not found in portfolio" % ticker)

        avg = 0
        for s in self.account.portfolio[ticker]:
            avg += sum(self.account.portfolio[ticker]['cost'])
        return avg/len(portfolio[ticker])
 
    def oldest_stock(self, ticker):
        if s not in self.account.portfolio:
            raise Exception("Stock %s not found in portfolio" % ticker)

        oldest = pd.to_datetime(self.account.portfolio[ticker][0]['date'])
        for e in self.account.portfolio[ticker]:
            if oldest > pd.to_datetime(e['date']):
                oldest = pd.to_datetime(e['date'])

        return oldest

    def netgain(self, ticker):
        if s not in self.account.portfolio:
            raise Exception("Stock %s not found in portfolio" % ticker)
        netgain = 0
        for e in self.account.portfolio[ticker]:
            netgain += e['shares'] * (self.get_current_stock_price(ticker) - portfolio[ticker]['costs'])

        return netgain


class Model:
    # Read the last few years of stocks and indices to calculate betas and standard deviations
    def __init__(self, brokerage):
        self.brokerage = brokerage
        self.stocks = si.tickers_sp500()
        self.indices = ['^IXIC', '^GSPC', '^DJI']
        self.stocks_ts = pd.DataFrame()
        self.stocks_spread = pd.DataFrame()
        self.indices_ts = pd.DataFrame()

        for idx in self.indices:
            data = yf.Ticker(idx).history(period=model_parameters["history"])
            d = data.copy()[['Open', 'Close']]
            self.indices_ts[idx+"_Open"] = d['Open']
            self.indices_ts[idx+"_Close"] = d['Close']

        self.betas = pd.read_csv("betas.csv")
        self.betas = self.betas.drop(['index', 'Attribute', 'Unnamed: 0.1', 'Unnamed: 0'], axis=1)
        self.betas = self.betas.rename(columns={"Recent": "Beta"})
        self.betas = self.betas.set_index('Ticker')

        for ticker in self.stocks:
            if os.path.exists(os.path.join("history", ticker)):
                data = pd.read_csv(os.path.join("history", ticker))
            else:
                data = yf.Ticker(ticker).history(period=model_parameters["history"])
                data.to_csv(os.path.join("history", ticker))
            d = data.copy()[['Open', 'Close']]
            self.stocks_ts[ticker+"_Open"] = d['Open']
            self.stocks_ts[ticker+"_Close"] = d['Close']
            self.stocks_spread[ticker] = (d['Close'] - d['Open']) * 100/d['Open']

            self.stocks_ts = self.stocks_ts.copy()
            self.stocks_spread = self.stocks_spread.copy()

        self.std = self.stocks_spread.describe().loc['std']
        self.mean = self.stocks_spread.describe().loc['mean']
        self.std = pd.DataFrame.from_dict(self.std)
        self.std.columns = ['std',]
        self.betas_mean = self.betas.describe()['Beta']['mean']

        self.mean = pd.DataFrame.from_dict(self.mean)
        self.mean.columns = ['mean',]


    def analyze_a_stock(self, ticker):
        data = yf.Ticker(ticker).history(period=history)
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

    def _getsellbuy(self, security_profit, security_loss):
        todays_delta = {}
        for ticker in self.stocks:
            open = self.brokerage.get_open_price_of_stock(ticker)
            close = self.brokerage.get_current_stock_price(ticker)
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
        today = pd.to_datetime(datetime.datetime.now())
        for ticker in self.stocks:
            if not self.brokerage.has_stock(ticker):
                continue

            oldest = self.oldest_stock(self, ticker)
            t = today - pd.to_datetime(oldest)

            avg = self.avg_cost_of_stock(ticker)

            netgain = self.netgain(ticker)
            if avg + model_parameters["sell_trigger"] * self.std['std'][ticker] * avg / 100 <= self.get_current_stock_price(ticker):
                # If the avg cost of the security has grown more than std
                # print(s, avg, portfolio[s]['costs'], std['std'][s+"_Diff"], stocks_ts.loc[index, s+"_Close"])

                if model_parameters["dump_all_trades"]:
                    print("Selling %s (beta %f) at profit. closing price %f. netgain %f days %d" %
                          (s, betas['Beta'][s], stocks_ts.loc[index, s+"_Close"], netgain, t.days),
                          self.get_stock_holdings(ticker))
                security_profit.loc[index] = {'beta':self.betas['Beta'][ticker],
                                              'days': t.days, 'profit': netgain}
                latest_diff.at[ticker, 'sell'] = True
            elif t.days > model_parameters["security_age"]:
                # if the security has aged for certain days, cut the losses
                if model_parameters["dump_all_trades"]:
                    print("Dumping %s(beta %f) because of age. closing price %f. netgain %f" %
                          (ticker, self.betas['Beta'][ticker], netgain),
                          self.get_stock_holdings(ticker))
                if loss > 0:
                        security_profit.loc[len(security_profit.index)] = {'beta':self.betas['Beta'][ticker],
                                                                           'days':t.days, 'profit':netgain}
                else:
                        security_loss.loc[len(security_loss.index)] = {'beta':self.betas['Beta'][ticker],
                                                                       'loss':netgain}
                latest_diff.at[ticker, 'sell'] = True
  
        sell_stocks = latest_diff[latest_diff['sell']]
        buy_stocks = latest_diff[latest_diff['buy']]
        if model_parameters["prefer_beta"]:
            buy_stocks = buy_stocks.sort_values('beta',ascending=False)
        return sell_stocks, buy_stocks

    def calculate_networth(self):
       networth = self.brokerage.cashbalance()
       for ticker in self.brokerage.get_stocks():
           networth += self.brokerage.netgain(ticker)
       return networth

    def do_todays_trade(self):
        stocks_profit = pd.DataFrame(columns=['beta', 'days', 'profit'])
        stocks_loss = pd.DataFrame(columns=['beta', 'loss'])

        price_movement = []
        cash_inhand = []

        sell, buy = self._getsellbuy(stocks_profit, stocks_loss)

        # process the stocks that are marked sell
        for st in sell.iterrows():
            ticker = st[0].split('_')[0]
            self.sell_a_stock(ticker)
            if self.has_stock(ticker):
               current_account += portfolio[stock]['shares'] * stocks_ts.loc[backtest_start_date][stock+"_Close"]
      
        # buy stocks that are marked by. We are buying max_stocks_to_buy number of stocks
        # TODO: The number of stocks to be must be adaptive. Will come up with some
        # algorithm based on:
        # 1. Market capitalization
        # 2. Beta
        # and other criteria
        # The goal is to put the money to work
        for st in buy.iterrows():
            ticker = st[0].split('_')[0]
            if self.cashbalance() > self.get_current_stock_price(ticker) * model_parameters["max_stocks_to_buy"]:
                self.buy_a_stock('msft', model_parameters["max_stocks_to_buy"])

        # lock in the gains after 10% increase of networth
        nw = self.calculate_networth()
        price_movement.append(nw)
        cash_inhand.append(self.brokerage.cashbalance())
        if nw > model_parameters["start_cash_balance"] * model_parameters["lockin_gains_factor"]:
            #print(backtest_start_date, calculate_networth(backtest_start_date), current_account, portfolio)
            for ticker in self.brokerage.get_stocks():
                self.brokerage.sell_a_stock(ticker)

        if model_parameters["dump_all_trades"]:
            print("Profit Distribution")
            print("===================")
            print(stocks_profit.describe())
            print()
            print("Loss Distribution")
            print("=================")
            print(stocks_loss.describe())
        return


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
"""

with LocalBrokerage(cash=10000) as lb:
    m = Model(lb)
    #m.analyze_a_stock('AAPL')
    #m.analyze_std_mean()
    m.do_todays_trade()
