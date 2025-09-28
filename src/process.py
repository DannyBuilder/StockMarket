import numpy as np
from sklearn.preprocessing import MinMaxScaler

SEQ_LEN = 100 

def scale_data(train_data, test_data):
    scaler = MinMaxScaler(feature_range=(0,1))
    train_scaled = scaler.fit_transform(train_data)
    combined = np.concatenate([train_data[-SEQ_LEN:], test_data])
    test_scaled = scaler.transform(combined)[SEQ_LEN:] 
    return train_scaled, test_scaled, scaler


def create_sequences(data):
    x = []
    y = []
    for i in range(SEQ_LEN, len(data)):
        x.append(data[i-SEQ_LEN:i])
        y.append(data[i, 0])
    return np.array(x), np.array(y)
