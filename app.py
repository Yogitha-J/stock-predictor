import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="QuantVision — AI Stock Predictor",
    page_icon="⚡",
    layout="wide"
)

# ─── Minimal CSS (safe version) ──────────────────────────
st.markdown("""
<style>
body {background-color:#03050a; color:#e2e8f8;}
.metric {font-size:20px; font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────
st.title("⚡ QuantVision — AI Stock Predictor")

# ─── Helper Functions ───────────────────────────────────
def make_sequences(data, seq_len=60):
    X, Y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i].flatten())
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

def plot_chart(actual, predicted):
    fig, ax = plt.subplots()
    ax.plot(actual, label="Actual")
    ax.plot(predicted, label="Predicted")
    ax.legend()
    st.pyplot(fig)

# ─── Input ──────────────────────────────────────────────
ticker = st.text_input("Enter NSE Ticker", "TCS.NS")
seq_len = st.slider("Lookback Window", 30, 90, 60)

if st.button("Run Prediction"):

    df = yf.download(ticker, start="2018-01-01", progress=False)

    if df.empty:
        st.error("Invalid ticker")
        st.stop()

    data = df[['Close']].values
    current_price = float(data[-1])

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, Y = make_sequences(scaled, seq_len)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    model = SVR(kernel='rbf', C=1000, gamma=0.1, epsilon=0.01)
    model.fit(X_train, Y_train)

    pred_scaled = model.predict(X_test).reshape(-1, 1)
    predictions = scaler.inverse_transform(pred_scaled)
    actual = scaler.inverse_transform(Y_test.reshape(-1, 1))

    # Next prediction
    last_seq = scaled[-seq_len:].flatten().reshape(1, -1)
    next_scaled = model.predict(last_seq)
    next_price = float(scaler.inverse_transform(next_scaled.reshape(-1,1))[0][0])

    change = ((next_price - current_price) / current_price) * 100

    # Metrics
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)

    # ─── Output ─────────────────────────────────────────
    st.subheader("Results")

    st.write(f"Current Price: ₹{current_price:.2f}")
    st.write(f"Predicted Next Price: ₹{next_price:.2f}")
    st.write(f"Change: {change:.2f}%")

    st.subheader("Metrics")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"R²: {r2:.4f}")

    st.subheader("Chart")
    plot_chart(actual[-100:], predictions[-100:])
