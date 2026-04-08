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

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantVision — AI Stock Predictor",
    page_icon="⚡",
    layout="wide"
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg: #03050a;
  --surface: #080d17;
  --card: #0b1220;
  --border: #1a2540;
  --accent: #00f5c3;
  --accent2: #7b61ff;
  --text: #e2e8f8;
  --muted: #4a5568;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00f5c3 0%, #7b61ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
}

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent);
}

.metric-lbl {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.signal-wrap {
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">QuantVision</div>', unsafe_allow_html=True)
st.markdown('<p style="color:#4a5568; letter-spacing: 2px;">SVR-BASED MARKET INFERENCE ENGINE</p>', unsafe_allow_html=True)
st.divider()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    stock_options = {
        "TCS": "TCS.NS",
        "Infosys": "INFY.NS",
        "Wipro": "WIPRO.NS",
        "Reliance": "RELIANCE.NS",
        "HDFC Bank": "HDFCBANK.NS",
    }
    selected_name = st.selectbox("Select Asset", list(stock_options.keys()))
    ticker = stock_options[selected_name]
    
    custom = st.text_input("Custom Ticker", placeholder="e.g. ZOMATO.NS")
    if custom.strip():
        ticker = custom.strip().upper()

    window = st.select_slider("Lookback Window", options=[30, 60, 90], value=60)
    predict_btn = st.button("⚡ EXECUTE", use_container_width=True, type="primary")

# ─── Helper Functions ────────────────────────────────────────────────────────
def make_sequences(data, seq_len):
    X, Y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i].flatten())
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

def set_dark_chart(fig, ax):
    fig.patch.set_facecolor('#03050a')
    ax.set_facecolor('#080d17')
    ax.tick_params(colors='#4a5568', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#1a2540')
    ax.grid(True, color='#1a2540', alpha=0.3)

# ─── Main Logic ──────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner(f"Analyzing {ticker}..."):
        df = yf.download(ticker, start="2019-01-01", progress=False)
        if df.empty:
            st.error("Invalid Ticker Data")
            st.stop()

        # Prep Data
        close_prices = df[['Close']].values
        current_price = float(close_prices[-1][0])
        
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close_prices)
        
        X, Y = make_sequences(scaled, window)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]

        # Train SVR
        model = SVR(kernel='rbf', C=1e3, gamma=0.1)
        model.fit(X_train, Y_train)

        # Predict
        pred_scaled = model.predict(X_test).reshape(-1, 1)
        predictions = scaler.inverse_transform(pred_scaled)
        actual = scaler.inverse_transform(Y_test.reshape(-1, 1))

        # Future
        last_seq = scaled[-window:].flatten().reshape(1, -1)
        next_val = model.predict(last_seq)
        next_price = float(scaler.inverse_transform(next_val.reshape(-1, 1))[0][0])
        
        # UI Metrics
        change = ((next_price - current_price) / current_price) * 100
        color = "#00f5c3" if change > 0 else "#ff4d6d"
        
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-val">₹{current_price:,.2f}</div><div class="metric-lbl">Current</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{color}">₹{next_price:,.2f}</div><div class="metric-lbl">Predicted</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{color}">{change:+.2f}%</div><div class="metric-lbl">Expected</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-val">{r2_score(actual, predictions):.2f}</div><div class="metric-lbl">R² Score</div></div>', unsafe_allow_html=True)

        st.divider()

        # Charting with Matplotlib
        st.markdown("### 📈 Prediction vs Actual")
        fig, ax = plt.subplots(figsize=(12, 4))
        set_dark_chart(fig, ax)
        
        ax.plot(actual[-100:], color='#4fc3f7', label='Actual', linewidth=2)
        ax.plot(predictions[-100:], color='#00f5c3', label='Predicted', linestyle='--', linewidth=2)
        ax.legend(facecolor='#0b1220', edgecolor='#1a2540', labelcolor='white')
        
        st.pyplot(fig)
        
        # Signal Output
        st.divider()
        if change > 0.5:
            st.success("✨ **STRONG BUY SIGNAL** — Model predicts upward momentum.")
        elif change < -0.5:
            st.error("⚠️ **SELL SIGNAL** — Model predicts downward trend.")
        else:
            st.warning("⏸ **HOLD SIGNAL** — Model predicts sideways movement.")
else:
    st.info("Configure the parameters in the sidebar and click Execute to start the analysis.")
