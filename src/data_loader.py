import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess():
    # Load dataset (skip bad rows)
    df = pd.read_csv("../data/stock_data.csv", skiprows=2)

    # Fix column names
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

    # Convert Close to numeric
    df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
    df.dropna(inplace=True)

    print("Cleaned Data:")
    print(df.head())

    # Use only Close column
    data = df[['Close']].values

    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    x, y = [], []

    for i in range(60, len(scaled_data)):
        x.append(scaled_data[i-60:i])
        y.append(scaled_data[i])

    x, y = np.array(x), np.array(y)

    print("Shape of X:", x.shape)
    print("Shape of Y:", y.shape)

    return x, y, scaler

if __name__ == "__main__":
    load_and_preprocess()