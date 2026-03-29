import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

SEQ_LEN = 60

def load_and_preprocess(ticker="TCS.NS", start="2018-01-01"):
    """
    Downloads live NSE stock data and prepares sequences for LSTM.
    Returns X, Y arrays and the fitted scaler.
    """
    df = yf.download(ticker, start=start, progress=False)

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    close_prices = df[['Close']].values

    # Normalize to 0-1 range
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Build 60-day sequences
    X, Y = [], []
    for i in range(SEQ_LEN, len(scaled_data)):
        X.append(scaled_data[i - SEQ_LEN:i])
        Y.append(scaled_data[i])

    X, Y = np.array(X), np.array(Y)

    print(f"Loaded {ticker}: {len(df)} rows → X shape: {X.shape}")
    return X, Y, scaler


if __name__ == "__main__":
    X, Y, scaler = load_and_preprocess("TCS.NS")
    print("X:", X.shape, "  Y:", Y.shape)
