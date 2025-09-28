import datetime as dt
import pandas as pd
import yfinance as yf

def load_stock_data(symbol='SPGI', start=dt.datetime(2010,1,1)):
    df = yf.download(symbol, start=start, end=dt.datetime.today())
    df = df.reset_index()
    df.to_csv('stock_data.csv', index=False)
    return df

def load_csv(path='stock_data.csv'):
    return pd.read_csv(path)