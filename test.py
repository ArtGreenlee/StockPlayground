import alpaca_trade_api as tradeapi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os

data = []
DATA_FOLDER_PATH = "DATA/AAPL"
for file in os.listdir(DATA_FOLDER_PATH):
    data.append(pd.read_csv(DATA_FOLDER_PATH + "/" + file))

def ROC_SIMULATION(prices, window, ROC_BUY_THRESHOLD, ROC_SELL_THRESHOLD):
    dataframe = pd.DataFrame(prices)
    MA = dataframe.rolling(window).mean()
    ROC = np.gradient(MA)
    buy_price = 0

    
    avg_change = 0
    num_trades = 0

    for i in range(window, len(prices)):
        rate_of_change = ROC[i]
        price = prices[i]

        if rate_of_change > ROC_BUY_THRESHOLD and buy_price == 0:
            buy_price = price
        
        if rate_of_change < ROC_SELL_THRESHOLD and buy_price != 0:
            avg_change += buy_price / price
            num_trades += 1
            buy_price = 0

    if num_trades == 0:
        return 0
    
    return avg_change / num_trades

def find_best_parameters(prices):
    max_change = 0
    best_window = 0
    best_RBT = 0
    best_RST = 0

    ROC_MIN = .001
    ROC_MAX = .02
    ROC_STEP = .002

    for window in range(50, 100):
        RBT = ROC_MIN
        while RBT < ROC_MAX:
            RBT += ROC_MIN
            RST = ROC_STEP
            while RST < ROC_MAX:
                RST += ROC_STEP
                change = ROC_SIMULATION(prices, window, RBT, RST)
                if change > max_change:
                    best_window = window
                    best_RBT = RBT
                    best_RST = RST
                    max_change = change

    return max_change, best_window, best_RBT, best_RST

def train_test(train_window):
    training_prices = []
    for i in range(train_window):
        training_prices.extend(data[i]['close'].values)
    test_prices = data[train_window]['close'].values
    max_change, best_window, best_RBT, best_RST = find_best_parameters(training_prices)
    print("for train window length: ", train_window, " best parameters are: ", best_window, best_RBT, best_RST)
    test_change = ROC_SIMULATION(test_prices, best_window, best_RBT, best_RST)
    print("test change: ", test_change)
    return test_change

max_test_change = 0
for train_window in range(0, 100):
    test_change = train_test(train_window)
    if test_change > max_test_change:
        max_test_change = test_change


