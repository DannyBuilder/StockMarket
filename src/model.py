from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))

    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, x_train, y_train, epochs=50, batch_size=32):
    """
    Train model with 20% of training data as validation, early stopping.
    """
    x_train_final, x_val, y_train_final, y_val = train_test_split(
        x_train, y_train, test_size=0.2, shuffle=False
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        x_train_final, y_train_final,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
    )
    
    return history
