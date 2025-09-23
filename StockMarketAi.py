import datetime as dt
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader as data
import plotly.graph_objects as go


plt.style.use('fivethirtyeight')

# Load the data
stock = 'AMZN'
start = dt.datetime(2010, 1, 1)
df = yf.download(stock, start=start, end=datetime.today())
df = df.reset_index()

data01 = df.to_csv('stock_data.csv')
data01 = pd.read_csv('stock_data.csv')

#Candlesticks
# fig = go.Figure(data=[go.Candlestick(x=data01['Date'],
#                 open=data01['Open'], high=data01['High'],
#                 low=data01['Low'], close=data01['Close'])])
# fig.update_layout(title=f'{stock} Candlestick Chart', xaxis_title='Date', yaxis_title='Price (USD)')
#
# fig.show()

#df = df.drop(['Date'], axis=1)

print(df.columns)

import matplotlib.dates as mdates



#Plot the close price
plt.figure(figsize=(12,6))
plt.plot(df['Close'], label=f'{stock} Close Price', linewidth=2)
plt.title(f'{stock} Close Price History')
plt.xlabel('Date')

plt.ylabel('Close Price USD ($)')
plt.legend()

# Format x-axis with readable dates
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Show year-month-day
plt.xticks(fontsize=8, rotation=0)
plt.tight_layout()
plt.show()


#Plot the Open price
plt.figure(figsize=(12,6))
plt.plot(df['Open'], label=f'{stock} Open Price', linewidth=2)
plt.title(f'{stock} Open Price History')
plt.xlabel('Date')
plt.ylabel('Open Price USD ($)')
plt.legend()

# Format x-axis with readable dates
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Show year-month-day
plt.xticks(fontsize=8, rotation=0)
plt.tight_layout()
plt.show()


#Plot the Day - High price
plt.figure(figsize=(12,6))
plt.plot(df['Low'], label=f'{stock} Low prices', linewidth=2)
plt.title(f'{stock} Low prices')
plt.xlabel('Date')
plt.ylabel('Low prices USD ($)')
plt.legend()

# Format x-axis with readable dates
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Show year-month-day
plt.xticks(fontsize=8, rotation=0)
plt.tight_layout()
plt.show()


#Plot the Day - Low price
plt.figure(figsize=(12,6))
plt.plot(df['Volume'], label=f'{stock}Volume', linewidth=2)
plt.title(f'{stock} Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()

# Format x-axis with readable dates
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Show year-month-day
plt.xticks(fontsize=8, rotation=0)
plt.tight_layout()
plt.show()


ma100 = pd.DataFrame(df)
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

plt.figure(figsize=(12,6))
plt.plot(df.Close, label=f'{stock} Close Price', linewidth=1)
plt.plot(ma100, label=f'{stock} 100-Day MA', linewidth=1)
plt.plot(ma200, label=f'{stock} 200-Day MA', linewidth=1)
plt.title(f'{stock} Moving Averages')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.legend()
plt.show()


