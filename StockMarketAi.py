import datetime as dt
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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
plt.plot(df['Date'], df['Close'], label=f'{stock} Close Price', linewidth=2)
plt.title(f'{stock} Close Price History')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.legend()
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(fontsize=8, rotation=0)
plt.tight_layout()
plt.show()


#Plot the Open price
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Open'], label=f'{stock} Open Price', linewidth=2)
plt.title(f'{stock} Open Price History')
plt.xlabel('Date')
plt.ylabel('Open Price USD ($)')
plt.legend()
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(fontsize=8, rotation=0)
plt.tight_layout()
plt.show()


#Plot the Day - High price
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Low'], label=f'{stock} Low prices', linewidth=2)
plt.title(f'{stock} Low prices')
plt.xlabel('Date')
plt.ylabel('Low prices USD ($)')
plt.legend()
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(fontsize=8, rotation=0)
plt.tight_layout()
plt.show()


#Plot the Day - Low price
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Volume'], label=f'{stock} Volume', linewidth=2)
plt.title(f'{stock} Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(fontsize=8, rotation=0)
plt.tight_layout()
plt.show()


ma100 = pd.DataFrame(df)
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'], label=f'{stock} Close Price', linewidth=1)
plt.plot(df['Date'], ma100, label=f'{stock} 100-Day MA', linewidth=1)
plt.plot(df['Date'], ma200, label=f'{stock} 200-Day MA', linewidth=1)
plt.title(f'{stock} Moving Averages')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.legend()
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(fontsize=8, rotation=0)
plt.tight_layout()
plt.show()


ema100 = df.Close.ewm(span=100, adjust=False).mean()
ema200 = df.Close.ewm(span=200, adjust=False).mean()

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'], label=f'{stock} Close Price', linewidth=1)
plt.plot(df['Date'], ema100, label=f'{stock} Exponential 100-Day MA', linewidth=1)
plt.plot(df['Date'], ema200, label=f'{stock} Exponential 200-Day MA', linewidth=1)
plt.title(f'{stock} Exponential Moving Averages')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.legend()
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(fontsize=8, rotation=0)
plt.tight_layout()
plt.show()


#Trainning and Testing the model
dataTraining = pd.DataFrame(df.Close[0:int(len(df)*0.70)])
dataTesting = pd.DataFrame(df.Close[int(len(df)*0.70) : int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
dataTrainingArray = scaler.fit_transform(dataTraining)

x_train = []
y_train = []

for i in range(100, dataTrainingArray.shape[0]):
    x_train.append(dataTrainingArray[i-100:i])
    y_train.append(dataTrainingArray[i, 0])

x_train, y_train  = np.array(x_train), np.array(y_train)

print(x_train.shape)
print(y_train.shape)


#Model Building
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1))


model.summary()


# Add EarlyStopping callback
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=30, callbacks=[early_stop])


past100days = dataTraining.tail(100)
final_df = pd.concat([past100days, dataTesting], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0]) 


x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)

y_predicted = model.predict(x_test)

scaler.scale_
scaler_factor = 1 / 0.0035166

y_predicted = y_predicted * scaler_factor
y_test = y_test * scaler_factor

scaler.scale_
scaler_factor = 1 / 0.0035166
y_predicted = y_predicted * scaler_factor
y_test = y_test * scaler_factor

#Final Graph
plt.figure(figsize=(12, 6))
plt.plot(y_test, label = 'Original Price', linewidth = 1)
plt.plot(y_predicted, label = 'Predicted Price', linewidth = 1)
plt.legend()
plt.show()








