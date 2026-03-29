import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR

st.set_page_config(page_title="AI Stock Predictor", page_icon="📈", layout="wide")

# -------------------- CACHE --------------------
@st.cache_data
def load_data(ticker):
    return yf.download(ticker, start="2018-01-01", progress=False)

# -------------------- SEQUENCE --------------------
def make_sequences(scaled, seq_len=60):
    X, Y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i].ravel())
        Y.append(scaled[i])
    return np.array(X), np.array(Y)

# -------------------- UI --------------------
st.title("📈 AI Stock Price Predictor")
st.caption("Sequence Model • NSE Stocks")

stock_options = {
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS"
}

ticker = st.selectbox("Select Stock", list(stock_options.values()))
custom = st.text_input("Or enter custom ticker (e.g. ZOMATO.NS)")

if custom.strip():
    ticker = custom.strip().upper()

predict_btn = st.button("Run Prediction")

# -------------------- MAIN LOGIC --------------------
if predict_btn:
    with st.spinner("Fetching data..."):

        df = load_data(ticker)

        # ✅ CLEAN DATA (CRITICAL FIX)
        if df.empty:
            st.error("No data fetched. Invalid ticker.")
            st.stop()

        df = df[['Close']].dropna()

        if len(df) < 100:
            st.error("Not enough data to train model.")
            st.stop()

        # ✅ FIXED (1D array)
        close_prices = df['Close'].values
        current_price = float(close_prices[-1])

        # -------------------- SCALING --------------------
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close_prices.reshape(-1, 1))

        # -------------------- SEQUENCES --------------------
        SEQ_LEN = 60
        X, Y = make_sequences(scaled, SEQ_LEN)

        if len(X) == 0:
            st.error("Sequence generation failed.")
            st.stop()

        # -------------------- TRAIN TEST SPLIT --------------------
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]

        # -------------------- MODEL --------------------
        model = SVR(kernel='rbf', C=1000, gamma=0.01, epsilon=0.01)
        model.fit(X_train, Y_train.ravel())

        # -------------------- PREDICTIONS --------------------
        pred_scaled = model.predict(X_test).reshape(-1, 1)
        predictions = scaler.inverse_transform(pred_scaled)
        actual = scaler.inverse_transform(Y_test.reshape(-1, 1))

        # -------------------- METRICS --------------------
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)

        # ✅ BETTER ACCURACY (MAPE)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        accuracy = max(0, 100 - mape)

        # -------------------- NEXT DAY PREDICTION --------------------
        last_seq = scaled[-SEQ_LEN:].ravel().reshape(1, -1)
        next_scaled = model.predict(last_seq).reshape(-1, 1)
        next_price = float(scaler.inverse_transform(next_scaled)[0][0])

        price_change = ((next_price - current_price) / current_price) * 100

        # -------------------- SIGNAL --------------------
        if price_change > 1.5:
            signal = "BUY"
        elif price_change < -1.5:
            signal = "SELL"
        else:
            signal = "HOLD"

    # -------------------- OUTPUT --------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Current Price", f"₹{current_price:.2f}")
    col2.metric("Predicted Price", f"₹{next_price:.2f}")
    col3.metric("Change %", f"{price_change:.2f}%")

    st.write(f"### Signal: {signal}")
    st.write(f"Model Accuracy: {accuracy:.2f}%")

    st.write("---")

    # -------------------- CHART --------------------
    fig, ax = plt.subplots()
    ax.plot(actual, label="Actual")
    ax.plot(predictions, label="Predicted")
    ax.legend()
    st.pyplot(fig)

    st.write("---")

    st.write(f"RMSE: ₹{rmse:.2f}")
    st.write(f"MAE: ₹{mae:.2f}")
