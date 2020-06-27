"""
Helper file for trading patterns

Bid columns - selling price (shorting price)
Ask columns - buying price (longing price)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import talib
import os
from plotly.offline import plot
import plotly.graph_objs as go

sns.set()

# data_file = '..\\data\\5Yrs Daily + 1hr + 5m.xlsx'
# sheet_names = ['Daily', '1 HR', '5m', 'Brief']
# data = pd.read_excel(data_file, sheet_name=sheet_names[1], parse_dates=True, index_col=0)
data = pd.read_csv('..\\data\\DataFile.csv', index_col=0, parse_dates=True)
data = data.sort_index()
data['RSI'] = talib.RSI(data['(Close, Ask)']).fillna(0)
data['Hour of Day'] = data.index.hour
data['Direction'] = np.where(data['(Close, Ask)'].pct_change() > 0, 1, np.where(data['(Close, Ask)'].pct_change() <0, -1, 0))
data['Real Volume Change'] = data['Real Volume'].pct_change().abs()
columns = data.columns
bid_columns = [col for col in columns if 'Bid' in col]
ask_columns = [col for col in columns if 'Ask' in col]

ask_data = data.loc[:, ask_columns]
bid_data = data.loc[:, bid_columns]

ask_data['Spread H-L'] = ask_data['(High, Ask)'] - ask_data['(Low, Ask)']
ask_data['Spread O-C'] = ask_data['(Open, Ask)'] - ask_data['(Close, Ask)']
bid_data['Spread H-L'] = bid_data['(High, Bid)*'] - bid_data['(Low, Bid)*']
bid_data['Spread O-C'] = bid_data['(Open, Bid)*'] - bid_data['(Close, Bid)*']

# Pattern recognition
# patterns = talib.get_function_groups()['Pattern Recognition']

ao = ask_data[ask_columns[0]]
ah = ask_data[ask_columns[1]]
al = ask_data[ask_columns[2]]
ac = ask_data[ask_columns[3]]

# bo = bid_data[bid_columns[0]]
# bh = bid_data[bid_columns[1]]
# bl = bid_data[bid_columns[2]]
# bc = bid_data[bid_columns[3]]

# print('Started pattern recognition .. ')
# for pattern in patterns:
#     data[pattern] = getattr(talib, pattern)(ao, ah, al, ac)
    # bid_data[pattern] = getattr(talib, pattern)(bo, bh, bl, bc)

# ask_data.to_csv('..\\data\\AskPatterns.csv')
# bid_data.to_csv('..\\data\\BidPatterns.csv')
# temp = data[patterns].abs().sum()
# working_patterns = list(temp[temp != 0].sort_values().index)
# print('Completed Pattern recognition!')


def moving_average(asset_data, col, params):
    short_window, long_window = params
    signals = pd.DataFrame(index=asset_data.index)
    short_ = 'SMA' + str(short_window)
    long_ = 'SMA' + str(long_window)
    signals[long_] = talib.SMA(asset_data[col], timeperiod=long_window).fillna(0)
    signals[short_] = talib.SMA(asset_data[col], timeperiod=short_window).fillna(0)
    mask = np.where(signals[short_].iloc[short_window:] > signals[long_].iloc[short_window:], 1, 0)

    return mask


def rsi(asset_data, col):
    timeperiod = 20
    signals = pd.DataFrame(index=asset_data.index)
    rsi_ = '-'.join(['rsi', str(timeperiod)])
    signals[rsi_] = talib.RSI(asset_data[col], timeperiod=timeperiod)
    mask = []
    isOpen = False
    isClose = True
    for ind in range(len(signals)):
        if ind < timeperiod:
            mask.append(0)
        elif signals[rsi_].iloc[ind] < 30 and isClose is True and isOpen is False:
            mask.append(1)
            isOpen = True
            isClose = False
        elif signals[rsi_].iloc[ind] > 70 and isOpen is True and isClose is False:
            mask.append(-1)
            isClose = True
            isOpen = False
        else:
            mask.append(0)
    mask = np.array(mask)
    return mask


def pattern_mask(asset_data, pattern_used):
    if pattern_used is None:
        pattern_used = patterns[-1]
    mask = []
    isOpen = False
    isClose = True
    signal_data = asset_data[pattern_used]
    for ind in range(len(signal_data.index)):
        if ind == 0:
            mask.append(0)
            continue
        signal = signal_data.iloc[ind]
        if signal == 100 and isOpen is False and isClose is True:
            mask.append(1)
            isOpen = True
            isClose = False
        elif signal == 0 and isOpen is True and isClose is False:
            mask.append(0)
        elif signal == -100 and isOpen is True and isClose is False:
            mask.append(-1)
            isClose = True
            isOpen = False
        else:
            mask.append(0)
    return mask


def plot_candlestick(asset_data, limit=1000, col_type='ask'):
    if col_type == 'ask':
        cols = ask_columns
    else:
        cols = bid_columns
    if limit:
        asset_data = asset_data.iloc[:limit]
    trace = go.Candlestick(x=asset_data.index,
                           open=asset_data[cols[0]],
                           high=asset_data[cols[1]],
                           low=asset_data[cols[2]],
                           close=asset_data[cols[3]])
    charts = [trace]
    plot(charts, filename='candlestickfile.html')
    return


if __name__ == '__main__':
    plot_candlestick(asset_data=data)
