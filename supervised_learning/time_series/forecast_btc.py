import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def load_data(file_path):
    data = np.load(file_path)
    return data['X'], data['y'], data['scaler']

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(X, y):
    # Split data into training and validation sets
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Build model
    model = build_model(X_train.shape[1:])
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32
    )
    
    return model, history

if __name__ == "__main__":
    # Load preprocessed data
    X, y, scaler = load_data("preprocessed_data.npz")
    
    # Train the model
    model, history = train_model(X, y)
    
    # Save the model
    model.save("btc_forecast_model.h5")
