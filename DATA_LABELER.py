# %%
import alpaca_trade_api as tradeapi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from collections import deque

def label_df(df, MA_WINDOW, FILTER_WINDOW, FILTER_POLY, BUY_WINDOW, SELL_WINDOW):

    #MA_WINDOW = 30
    #FILTER_WINDOW = 30
    #FILTER_POLY = 1
    MA = np.array(savgol_filter(np.array(df['close'].rolling(MA_WINDOW).mean()), FILTER_WINDOW, FILTER_POLY))
    ROC = np.gradient(MA, axis=0)
    prices = df['close'].values
    plt.plot(MA)
    plt.plot(prices)
    std = np.nanstd(ROC)
    RBT = std
    RST = -std

    maxima = argrelextrema(MA, np.greater)
    minima = argrelextrema(MA, np.less)
    buy_indices = set()
    sell_indices = set()

    for i in range(len(maxima[0])):
        sell_index = maxima[0][i]
        start_index = max(0, sell_index - SELL_WINDOW)
        end_index = min(sell_index + SELL_WINDOW, len(prices) - 1)
        for j in range(start_index, end_index):
            if j not in sell_indices:
                sell_indices.add(j)

    for i in range(len(minima[0])):
        buy_index = minima[0][i]
        start_index = max(0, buy_index - BUY_WINDOW)
        end_index = min(buy_index + BUY_WINDOW, len(prices) - 1)
        for j in range(start_index, end_index):
            if j not in buy_indices:
                buy_indices.add(j)

    intersection = buy_indices.intersection(sell_indices)\

    for i in intersection:
        sell_indices.remove(i)
        buy_indices.remove(i)

    labels = []
    BUY_LABEL = "BUY"
    SELL_LABEL = "SELL"
    HOLD_LABEL = "HOLD"

    for i in range(MA_WINDOW):
        labels.append("HOLD")

    for i in range(MA_WINDOW, len(prices)):
        if i in sell_indices:
            labels.append(SELL_LABEL)
        elif i in buy_indices:
            labels.append(BUY_LABEL)
        else:
            labels.append(HOLD_LABEL)

    df['label'] = labels
    return df

DATA_FOLDER_PATH = "DATA/AAPL"
LABELED_DATA_FOLDER_PATH = "LABELED_DATA/AAPL"
for file in os.listdir(DATA_FOLDER_PATH):
    df = pd.read_csv(DATA_FOLDER_PATH + "/" + file)
    labeled_df = label_df(df, 30, 30, 1, 10, 10)
    labeled_df = labeled_df.loc[:, ~labeled_df.columns.str.contains('^Unnamed')]
    labeled_df.to_csv(LABELED_DATA_FOLDER_PATH + "/" + file, index = False)
