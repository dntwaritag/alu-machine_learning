import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(input_file, output_file):
    # Load raw dataset
    df = pd.read_csv(input_file)

    # Drop unnecessary columns and rows with missing/invalid values
    df = df.dropna().reset_index(drop=True)
    df = df[df['Volume_(BTC)'] > 0]

    # Select relevant features
    features = ['Open', 'High', 'Low', 'Close',
                'Volume_(BTC)', 'Volume_(Currency)']
    df = df[features]

    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)

    # Create sliding windows of 24 hours
    window_size = 24 * 60  # 24 hours, assuming 1-minute intervals
    X, y = [], []
    for i in range(len(normalized_data) - window_size):
        X.append(normalized_data[i:i + window_size])
        y.append(normalized_data[i + window_size, 3])  # Predict Close price

    X = np.array(X)
    y = np.array(y)

    # Save processed data
    np.savez(output_file, X=X, y=y, scaler=scaler)


if __name__ == "__main__":
    preprocess_data("coinbase.csv", "preprocessed_data.npz")
