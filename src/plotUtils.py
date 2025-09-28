import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

plt.style.use('fivethirtyeight')

pio.renderers.default = "browser"

def plot_line_chart(dates, values, title, ylabel, label):
    plt.figure(figsize=(12,6))
    plt.plot(dates, values, label=label, linewidth=2)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(fontsize=8, rotation=0)
    plt.tight_layout()
    plt.show()


def plot_candlestick(data01, symbol='STOCK'):
    fig = go.Figure(data=[go.Candlestick(x=data01['Date'],
                    open=data01['Open'], high=data01['High'],
                    low=data01['Low'], close=data01['Close'])])
    fig.update_layout(title=f'{symbol} Candlestick Chart', xaxis_title='Date', yaxis_title='Price (USD)')

    fig.show()


def plot_training_history(history):
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()

def plot_predictions(dates, original, predicted, stock_name='STOCK'):
    plt.figure(figsize=(12,6))
    plt.plot(dates, original, label='Original Price', linewidth=1)
    plt.plot(dates, predicted, label='Predicted Price', linewidth=1)
    plt.title(f'{stock_name} Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price USD ($)')
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(fontsize=8, rotation=0)
    plt.tight_layout()
    plt.show()
