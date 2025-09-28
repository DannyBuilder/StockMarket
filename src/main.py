from dataLoader import load_stock_data, load_csv
from process import create_sequences, scale_data
from evaluateTest import evaluate
from plotUtils import plot_candlestick, plot_training_history, plot_predictions
from model import build_model, train_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

SEQ_LEN = 100 

def main(stock_symbol='SPGI'):
    # Load data
    df = load_stock_data(stock_symbol)

    data01 = pd.read_csv('stock_data.csv')
    
    # Plot candlestick chart
    plot_candlestick(data01, stock_symbol)
    
    # Prepare data
    dataTraining = pd.DataFrame(df.Close[0:int(len(df)*0.70)])
    dataTesting = pd.DataFrame(df.Close[int(len(df)*0.70) : int(len(df))])
    dates_testing = df['Date'][int(len(df)*0.70):].reset_index(drop=True)
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataTrainingArray = scaler.fit_transform(dataTraining)
    
    # Scale data
    #train_scaled, test_scaled, scaler = scale_data(train_data, test_data)
    
    # Create sequences
    x_train, y_train = create_sequences(dataTrainingArray)
    
    #x_test, y_test = create_sequences(np.concatenate([train_scaled[-SEQ_LEN:], test_scaled], axis=0))
    
    # Reshape for LSTM
    #x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    #x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    
    # ----- Build and train model (fully copied from original) -----

    model = build_model((x_train.shape[1], 1))
    
    history = train_model(model, x_train, y_train, epochs=50, batch_size=32)


    # Plot training vs validation loss
    history = plot_training_history(history)

    past100days = dataTraining.tail(100)
    final_df = pd.concat([past100days, dataTesting], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0]) 


    x_test, y_test = np.array(x_test), np.array(y_test)
    
    # Predict using last 100 days + test set
    y_predicted = model.predict(x_test)
    y_predicted = y_predicted.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    # Use inverse_transform to get actual prices
    y_predicted_actual = scaler.inverse_transform(y_predicted)
    y_test_actual = scaler.inverse_transform(y_test)
    
    #Final Graphs
    plot_predictions(dates_testing, y_test_actual, y_predicted_actual, stock_symbol)
    
    # Evaluate Using R^2 Score, RMSE, MAE
    results = evaluate(y_test_actual, y_predicted_actual)
    print(f"Evaluation results for {stock_symbol}:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    
    # Tomorrow's predicted price
    print(f"Predicted price for tomorrow: {y_predicted_actual[-1][0]:.2f} USD")


if __name__ == "__main__":
    main()
