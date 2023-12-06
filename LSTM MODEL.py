import numpy as np
import pandas as pd

import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense

from sklearn.preprocessing import MinMaxScaler

# Load your dataset
df = pd.read_csv('H:\Binali Crypto\crypto csv\BTC-USD.csv')

# Assuming 'Close' is the target variable
# You may need to preprocess other columns as per your requirement
close_prices = df['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)


# Create a dataset with time steps
def create_dataset(dataset, time_step=1):
    X_data, y_data = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X_data.append(a)
        y_data.append(dataset[i + time_step, 0])
    return np.array(X_data), np.array(y_data)


time_step = 60
X, y = create_dataset(scaled_data, time_step)

# Reshape input to be [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Splitting dataset into train and test split
training_size = int(len(X) * 0.67)
test_size = len(X) - training_size
X_train, X_test = X[0:training_size], X[training_size:len(X)]
y_train, y_test = y[0:training_size], y[training_size:len(y)]

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Save the model
model.save('crypto_prediction_model.h5')

# Save the scaler
scaler_filename = "scaler.save"
joblib.dump(scaler, scaler_filename)

# Load the model and scaler
model = load_model('crypto_prediction_model.h5')
scaler = joblib.load('scaler.save')


def predict_high_low(crypto_data):
    # 'crypto_data' is the recent price data for the cryptocurrency
    # Make sure to preprocess this data in the same way as you did before training

    # Make predictions
    predicted = model.predict(crypto_data)
    # Reverse the MinMaxScaler transformation
    predicted_prices = scaler.inverse_transform(predicted)

    # Return the predicted high and low
    predicted_high = np.max(predicted_prices)
    predicted_low = np.min(predicted_prices)
    return predicted_high, predicted_low
