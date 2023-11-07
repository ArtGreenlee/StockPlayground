import alpaca_trade_api as tradeapi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import pandas_market_calendars as mcal

# API Info for fetching data, portfolio, etc. from Alpaca
BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_API_KEY = "PKZOQ4KFK6JBDSDCZ8QW"
ALPACA_SECRET_KEY = "RnEHRqATzw7ybJ0iZ1dqjTRKonvRTQCshwDnA3jb"

# Instantiate REST API Connection
api = tradeapi.REST(key_id=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, 
                    base_url=BASE_URL, api_version='v2')    

start_date = "01/01/23"
end_date = "11/06/23"
start_date = datetime.datetime.strptime(start_date, "%m/%d/%y")
end_date = datetime.datetime.strptime(end_date, "%m/%d/%y")

TICKER = "AAPL"

DATA_FILE_PATH = "DATA/"

DATA_FOLDER = DATA_FILE_PATH + TICKER
if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)

nyse = mcal.get_calendar('NYSE')
market_dates = nyse.valid_days(start_date=start_date, end_date=end_date)
valid_dates = []
for i in range(len(market_dates)):
    valid_dates.append(str(market_dates[i]).split()[0])

for i in range(365):
    start = str(start_date).split()[0]
    if start in valid_dates:
        TRAINING_DATA = api.get_bars(TICKER, tradeapi.rest.TimeFrame.Minute, start, start, adjustment='raw').df
        TRAINING_DATA.to_csv(DATA_FOLDER + "/" + start)
    start_date += datetime.timedelta(days=1)
