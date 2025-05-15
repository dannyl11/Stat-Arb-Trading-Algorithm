import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller

num_sims = 1000

ticker_1 = 'NVDA'
ticker_2 = 'AAPL'

data_1 = yf.download(ticker_1, period='1y', progress=False)
data_2 = yf.download(ticker_2, period='1y', progress=False)

data_1 = data_1['Close']
data_2 = data_2['Close']

data_1_shifted = data_1.shift(1).dropna()
log_return_1 = np.log(data_1/data_1_shifted) #df

#calculate sigma (volatility)
sigma_1 = log_return_1.std(ddof=1).iloc[0] 

#calcualte mu (drift)
daily_mu_1 = log_return_1.mean().iloc[0] 
mu_1 = daily_mu_1 + (sigma_1**2)/2

#intial stock prices
stock_1_s0 = data_1.iloc[0, 0] #float

t = 1
n = 252 #252 trading days/steps in calendar year

#find alpha, beta, phi, and sigma_u historically
ind = sm.add_constant(data_1)
ols = sm.OLS(data_2, ind).fit()
residuals = ols.resid

alpha = ols.params.iloc[0]
beta = ols.params.iloc[1]

auto_reg = AutoReg(residuals.reset_index(drop=True), lags=1).fit()

phi = auto_reg.params.iloc[1]  # lag-1 coefficient
sigma_u = np.std(auto_reg.resid)

def createCointegratedSeries(alpha, beta, phi, sigma_u, price_path, n):
    #create mean-reverting residuals
    u_t = np.zeros(n)
    epsilon = np.random.normal(0, sigma_u, n)
    for t in range(n):
        u_t[t] = phi*u_t[t-1] + epsilon[t]

    #create the cointegrated series
    coint_path = alpha + beta * price_path + u_t
    return coint_path

def geomBrownianMotion(s0, mu, sigma, t, n):
    dt = t/n
    dW = np.random.normal(0, np.sqrt(dt), n) #weiner process has mean 0 and st dev equal to square root of time step
    weiner_process = np.cumsum(dW)
    gbm = np.exp((mu - sigma**2 / 2) * dt + sigma * weiner_process)
    price_path = s0 * gbm
    return price_path

def simulateStrategy(num_sims):
    all_returns = []
    for _ in range(num_sims):
        #Simulate stock price paths
        stock_1_price_path = geomBrownianMotion(stock_1_s0, mu_1, sigma_1, t, n)
        stock_2_price_path = createCointegratedSeries(alpha, beta, phi, sigma_u, 
                                                stock_1_price_path, n)
        future_data = pd.DataFrame({ticker_1: stock_1_price_path,
                                    ticker_2: stock_2_price_path})
        # rebase the data
        rebase_df = future_data.copy()
        for i in range(0, len(rebase_df), 30):
            rebase_value_1 = rebase_df.loc[i, ticker_1]
            rebase_df.loc[i:i+29, ticker_1] = (rebase_df.loc[i:i+29, ticker_1] / 
                                                rebase_value_1)
            rebase_value_2 = rebase_df.loc[i, ticker_2]
            rebase_df.loc[i:i+29, ticker_2] = (rebase_df.loc[i:i+29, ticker_2] / 
                                                rebase_value_2)
        # calculate the ratio, mean, std, entry, and exit
        rebase_df['Ratio'] = rebase_df[ticker_1]/rebase_df[ticker_2]
        mean = rebase_df['Ratio'].mean()
        std = rebase_df['Ratio'].std()
        entry_threshold = 1.15 #75% CI
        exit_threshold = 0.67 #50% CI

        #calculating PnL
        future_data['Ratio'] = rebase_df['Ratio']

        starting_balance = 10000
        balance = starting_balance
        position = 'neutral'

        stock_1 = {'long': [], 'short': []} #dict of all long and short positions with prices
        stock_2 = {'long': [], 'short': []}

        pnl = []

        for i in range(len(future_data)):
            ratio = future_data.loc[i, 'Ratio']
            entry_short = mean + entry_threshold * std
            entry_long = mean - entry_threshold * std
            exit_short = mean + exit_threshold * std
            exit_long = mean - exit_threshold * std
            curr_price_1 = future_data.loc[i, ticker_1]
            curr_price_2 = future_data.loc[i, ticker_2]
            allocation = balance / 2
            shares_1 = allocation / curr_price_1
            shares_2 = allocation / curr_price_2
            if position == 'neutral':
                if ratio < entry_long:
                    position = 'long'
                    stock_1['long'].append((curr_price_1, shares_1))
                    stock_2['short'].append((curr_price_2, shares_2))
                elif ratio > entry_short:
                    position = 'short'
                    stock_1['short'].append((curr_price_1, shares_1))
                    stock_2['long'].append((curr_price_2, shares_2))
            elif position == 'long':
                if ratio > exit_long:
                    entry_price_1, shares_1 = stock_1['long'].pop(0) #FIFO
                    entry_price_2, shares_2 = stock_2['short'].pop(0)
                    position = 'neutral'
                    balance += (curr_price_1 - entry_price_1) * shares_1
                    balance += (entry_price_2 - curr_price_2) * shares_2
            elif position == 'short':
                if ratio < exit_short:
                    entry_price_1, shares_1 = stock_1['short'].pop(0) #FIFO
                    entry_price_2, shares_2 = stock_2['long'].pop(0)
                    position = 'neutral'
                    balance += (entry_price_1 - curr_price_1) * shares_1
                    balance += (curr_price_2 - entry_price_2) * shares_2
            total_long = 0
            total_short = 0
            for val in stock_1['long']:
                price, shares = val
                total_long += (curr_price_1 - price) * shares
            for val in stock_1['short']:
                price, shares = val
                total_short += (price - curr_price_1) * shares
            for val in stock_2['long']:
                price, shares = val
                total_long += (curr_price_2 - price) * shares
            for val in stock_2['short']:
                price, shares = val
                total_short += (price - curr_price_2) * shares
            pnl.append(balance + total_long + total_short)
        pnl = pd.Series(pnl)
        final_balance = pnl.iloc[-1]
        returns = ((final_balance - starting_balance)/starting_balance)*100
        all_returns.append(returns)
        plt.plot(pnl)
        print(f'Simulation {_}')
    return (sum(all_returns)/len(all_returns)) #average returns

avg_return = simulateStrategy(1000)
print(f'Average return over {num_sims} simulations: {round(avg_return, 2)}%')
plt.title(f'{num_sims} Simulations of {ticker_1}/{ticker_2} Pairs Trading Strategy')
plt.xlabel('Days')
plt.ylabel('Balance')
plt.axhline(y=1000, color='black')
plt.show()
