# %%
import alpaca_trade_api as tradeapi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import datetime
# API Info for fetching data, portfolio, etc. from Alpaca
BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_API_KEY = "PKZOQ4KFK6JBDSDCZ8QW"
ALPACA_SECRET_KEY = "RnEHRqATzw7ybJ0iZ1dqjTRKonvRTQCshwDnA3jb"

# Instantiate REST API Connection
api = tradeapi.REST(key_id=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, 
                    base_url=BASE_URL, api_version='v2')    

# %%
def ROC_SIMULATION(prices, window, ROC_BUY_THRESHOLD, ROC_SELL_THRESHOLD):
    MA = prices.rolling(window).mean()
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

# %%
def find_best_parameters(prices):
    max_change = 0
    best_window = 0
    best_RBT = 0
    best_RST = 0

    ROC_MIN = .001
    ROC_MAX = .05
    ROC_STEP = .001

    for window in range(10, 150):
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

# %%
def train_test(train_window):
    training_length = train_window
    start_date = "08/01/23"
    start_date = datetime.datetime.strptime(start_date, "%m/%d/%y")
    end_date = start_date + datetime.timedelta(days=training_length)
    training_date = end_date + datetime.timedelta(days=1)
    start = str(start_date).split()[0]
    end = str(end_date).split()[0]
    training = str(training_date).split()[0]
    print("Start date: ", start)
    print("End date: ", end)
    TRAINING_DATA = api.get_bars("AAPL", tradeapi.rest.TimeFrame.Minute, start, end, adjustment='raw').df
    TEST_DATA = api.get_bars("AAPL", tradeapi.rest.TimeFrame.Minute, training, training, adjustment='raw').df   
    training_prices = TRAINING_DATA['close']
    test_prices = TEST_DATA['close']
    max_change, best_window, best_RBT, best_RST = find_best_parameters(training_prices)
    print("for train window length: ", train_window, " best parameters are: ", best_window, best_RBT, best_RST)
    test_change = ROC_SIMULATION(test_prices, best_window, best_RBT, best_RST)
    return test_change

# %%
max_test_change = 0
for train_window in range(0, 100):
    test_change = train_test(train_window)
    if test_change > max_test_change:
        max_test_change = test_change
        print(max_test_change)


