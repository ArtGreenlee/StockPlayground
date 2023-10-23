import alpaca_trade_api as tradeapi
import numpy as np

import matplotlib.pyplot as plt

# API Info for fetching data, portfolio, etc. from Alpaca
BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_API_KEY = "PK57KQG0FN3KCTVNJYIH"
ALPACA_SECRET_KEY = "PfKx2nmpUhC7jsZj9d2P2DOzuZJY0nvCC4cg9yrj"

# Instantiate REST API Connection
api = tradeapi.REST(key_id=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, 
                    base_url=BASE_URL, api_version='v2')    

# Fetch Apple data from last 100 daysAPPLE_DATA = api.get_barset('AAPL', 'day', limit=100)# Preview Dataprint(APPLE_DATA.df.head())

# Fetch Apple data from last 100 days
APPLE_DATA = api.get_bars('AAPL', 'day').df


# Reformat data (drop multiindex, rename columns, reset index)
APPLE_DATA.columns = APPLE_DATA.columns.to_flat_index()
APPLE_DATA.columns = [x[1] for x in APPLE_DATA.columns]
APPLE_DATA.reset_index(inplace=True)
print(APPLE_DATA.head())

# Plot stock price data
plot = APPLE_DATA.plot(x="time", y="close", legend=False)
plot.set_xlabel("Date")
plot.set_ylabel("Apple Close Price ($)")
plt.show()