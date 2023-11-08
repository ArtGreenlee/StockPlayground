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

class Trade:
    buy_time = ""
    buy_price = 0
    sell_price = 0  
    sell_time = ""

def ROC_SIMULATION(prices, window, ROC_BUY_THRESHOLD, ROC_SELL_THRESHOLD):
    dataframe = pd.DataFrame(prices)
    MA = np.array(dataframe.rolling(window).mean())
    ROC = np.gradient(MA, axis=0)
    buy_price = 0

    avg_change = 0
    num_trades = 0

    for i in range(window, len(prices)):
        rate_of_change = ROC[i]
        price = prices[i]

        if rate_of_change > ROC_BUY_THRESHOLD and buy_price == 0:
            buy_price = price
        
        if rate_of_change < ROC_SELL_THRESHOLD and buy_price != 0:
            avg_change += price / buy_price
            num_trades += 1
            buy_price = 0

    if num_trades == 0:
        return 1
    
    return avg_change / num_trades

def find_best_parameters(prices):
    max_change = 0
    best_window = 0
    best_RBT = 0
    best_RST = 0

    ROC_MIN = .001
    ROC_MAX = 2
    ROC_STEP = .001

    for window in range(1, 150):
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
    num_tests = 10
    avg_test_change = 0

    RESULTS_FOLDER_PATH = "RESULTS/"
    RESULTS_FILE = open(RESULTS_FOLDER_PATH + "window_" + str(train_window) + ".txt", "a")

    for i in range(num_tests):
        print("Test ", i + 1, " of ", num_tests, "for window length: ", train_window)
        training_prices = []

        for j in range(train_window):
            training_prices.extend(data[i * 2 + j]['close'].values)

        test_prices = data[train_window]['close'].values
        max_change, best_window, best_RBT, best_RST = find_best_parameters(training_prices)
        results_string = "RBT: " + str(best_RBT) + " RST " + str(best_RST) + " best_window " + str(best_window) + " change " + str(max_change) + "\n"
        RESULTS_FILE.write(results_string)
        print(results_string)
        test_change = ROC_SIMULATION(test_prices, best_window, best_RBT, best_RST)
        print("test change: ", test_change)
        avg_test_change += test_change

    print("AVG TEST_CHANGE: ", avg_test_change / num_tests)
    RESULTS_FILE.write("AVG_TEST_CHANGE: " + str(avg_test_change / num_tests) + "\n")
    RESULTS_FILE.close()
    return avg_test_change / num_tests

max_test_change = 0
for train_window in range(1, 100):
    test_change = train_test(train_window)
    if test_change > max_test_change:
        max_test_change = test_change


