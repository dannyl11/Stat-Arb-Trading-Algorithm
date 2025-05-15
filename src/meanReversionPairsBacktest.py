import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import date
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

ticker_1 = 'NVDA'
ticker_2 = 'AAPL'

data_1 = yf.download(ticker_1, period='1y', progress=False)
data_2 = yf.download(ticker_2, period='1y', progress=False)

data_1 = data_1['Close'] #df
data_2 = data_2['Close'] #df

#Test correlation
data_1_shifted = data_1.shift(1).dropna()
data_2_shifted = data_2.shift(1).dropna()

log_return_1 = np.log(data_1/data_1_shifted)
log_return_2 = np.log(data_2/data_2_shifted)

returns_df = pd.DataFrame(({ticker_1: log_return_1[ticker_1].dropna(), 
                            ticker_2: log_return_2[ticker_2].dropna()}))
correlation = returns_df.corr().iloc[0, 1]

#Test cointegration
ind = sm.add_constant(data_1)
ols = sm.OLS(data_2, ind).fit()
residuals = ols.resid

adf_test = adfuller(residuals)
p_value = adf_test[1]
print(f'Cointegration p-value = {round(p_value, 4)}')


###############################################################################
#Rebasing Data
rebase_df = pd.DataFrame(({ticker_1: data_1[ticker_1], 
                           ticker_2: data_2[ticker_2]}))
rebase_df.reset_index(drop=True, inplace=True)

for i in range(0, len(rebase_df), 30):
    rebase_value_1 = rebase_df.loc[i, ticker_1]
    rebase_df.loc[i:i+29, ticker_1] = (rebase_df.loc[i:i+29, ticker_1] / 
                                        rebase_value_1)
    rebase_value_2 = rebase_df.loc[i, ticker_2]
    rebase_df.loc[i:i+29, ticker_2] = (rebase_df.loc[i:i+29, ticker_2] / 
                                        rebase_value_2)

rebase_df['Ratio'] = rebase_df[ticker_1]/rebase_df[ticker_2]
# print(rebase_df.to_string())

###############################################################################
#Generating signals
mean = rebase_df['Ratio'].mean()
std = rebase_df['Ratio'].std()
entry_threshold = 1.5 #87% CI
exit_threshold = 0.67 #50% CI

sell_signals = rebase_df[rebase_df['Ratio'] > mean + entry_threshold * std]
#selling pair means short stock1 and long stock 2
buy_signals = rebase_df[rebase_df['Ratio'] < mean - entry_threshold * std]
#buying pair means long stock1 and short stock 2
print(f'+{entry_threshold}sigma:\n{sell_signals}')
print(f'-{entry_threshold}sigma:\n{buy_signals}')

###############################################################################
#Backtesting
backtest_df = pd.DataFrame({ticker_1: data_1[ticker_1], 
                            ticker_2: data_2[ticker_2]})
backtest_df.reset_index(inplace=True, drop=True)
backtest_df['Ratio'] = rebase_df['Ratio']
# print(backtest_df.to_string())

starting_balance = 10000
balance = starting_balance
position = 'neutral'

stock_1 = {'long': [], 'short': []} #dict of all long and short positions with prices
stock_2 = {'long': [], 'short': []}

pnl = []

for i in range(len(backtest_df)):
    ratio = backtest_df.loc[i, 'Ratio']
    entry_short = mean + entry_threshold * std
    entry_long = mean - entry_threshold * std
    exit_short = mean + exit_threshold * std
    exit_long = mean - exit_threshold * std
    curr_price_1 = backtest_df.loc[i, ticker_1]
    curr_price_2 = backtest_df.loc[i, ticker_2]
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

# print(pnl.to_string())

print(f'Final balance = {final_balance:.2f}, Return = {returns:.2f}%')      

###############################################################################
#Visualize  Correlation
plt.figure()
plt.plot(log_return_1, label=f'{ticker_1}')
plt.plot(log_return_2, label=f'{ticker_2}')
plt.title(f'Correlation {ticker_1}/{ticker_2} {date.today()}\n' \
          f'P-value = {round(p_value, 4)}')
plt.legend()

#Visualize Ratio Movement
plt.figure()
plt.plot(rebase_df['Ratio'])
# plt.plot(buy_signals, linestyle=None, marker='^', color='green')
# plt.plot(sell_signals, linestyle=None, marker='^', color='red')
plt.title(f'Mean Reversion Pair {ticker_1}/{ticker_2} {date.today()}\n' \
          f'Mean = {mean:.4f} Sigma = {std:.4f}')
plt.xlabel('Trading Days')
plt.grid(axis='y')
plt.ylim(mean-4*std, mean+4*std)
plt.axhline(mean + entry_threshold * std, color='green', 
            linestyle='dashed', label=f'+{entry_threshold}σ')
plt.axhline(mean + exit_threshold * std, color='green', 
            linestyle='dashed', label=f'+{exit_threshold}σ')
plt.axhline(mean - exit_threshold * std, color='red', 
            linestyle='dashed', label=f'-{exit_threshold}σ')
plt.axhline(mean - entry_threshold * std, color='red', 
            linestyle='dashed', label=f'-{entry_threshold}σ')
plt.axhline(mean, color='black', linestyle='dashed')
plt.legend()

#Visualize Trading Strategy PnL
plt.figure(figsize=(10, 5))
plt.plot(pnl)
plt.title(f'PnL {ticker_1}/{ticker_2}\n' \
          f'Return = {returns:.2f}%')
plt.xlabel('Trading Days')
plt.ylabel('Balance')
plt.grid(True)

plt.show()
