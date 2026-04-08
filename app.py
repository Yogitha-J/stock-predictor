import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

# Page Config
st.set_page_config(page_title="AI Stock Predictor", page_icon="📈", layout="wide")

# CSS Styling - Fixed Syntax [cite: 31, 34]
st.markdown("""
<style>
.metric-card{background:#1e2130;border:1px solid #2e3650;border-radius:12px;padding:20px;text-align:center;margin:6px 0}
.metric-value{font-size:2rem;font-weight:700;color:#00d4aa}
.metric-label{font-size:0.85rem;color:#8892b0;margin-top:4px}
.signal-buy{background:#0d3320;border:2px solid #00d4aa;border-radius:12px;padding:24px;text-align:center}
.signal-sell{background:#3d0d0d;border:2px solid #ff6b6b;border-radius:12px;padding:24px;text-align:center}
.signal-hold{background:#1a1a0d;border:2px solid #ffd700;border-radius:12px;padding:24px;text-align:center}
.signal-text{font-size:2.5rem;font-weight:800}
.explainer-box{background:#1e2130;border-left:4px solid #00d4aa;border-radius:0 8px 8px 0;padding:16px 20px;margin:12px 0}
</style>
""", unsafe_allow_html=True)

st.markdown("## 📈 AI Stock Price Predictor")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔮 Live Prediction", "📊 Model Performance", "🧠 How It Works"])

def make_sequences(scaled_data, seq_len=60):
    X, Y = [], []
    for i in range(seq_len, len(scaled_data)):
        # Ravel ensures the 60-day window is a 1D flat array for the SVR [cite: 66]
        X.append(scaled_data[i-seq_len:i].flatten()) 
        Y.append(scaled_data[i, 0])
    return np.array(X), np.array(Y)

with tab1:
    col_input, col_info = st.columns([1, 2])
    with col_input:
        stock_options = {
            "TCS (Tata Consultancy)":"TCS.NS",
            "Infosys":"INFY.NS",
            "Wipro":"WIPRO.NS",
            "Reliance Industries":"RELIANCE.NS",
            "HDFC Bank":"HDFCBANK.NS",
        }
        selected_name = st.selectbox("Choose a stock", list(stock_options.keys()))
        ticker = stock_options[selected_name]
        
        custom = st.text_input("Or enter any NSE ticker", placeholder="e.g. ZOMATO.NS")
        if custom.strip():
            ticker = custom.strip().upper()
            selected_name = ticker
            
        predict_btn = st.button("🚀 Run Prediction", use_container_width=True, type="primary")

    with col_info:
        st.markdown("""<div class="explainer-box"><b>Model Logic:</b><br>
        Uses 5 years of historical data to train an RBF-Kernel SVR. 
        It analyzes 60-day price trends to predict the next closing price.</div>""", unsafe_allow_html=True)

    if predict_btn:
        with st.spinner(f"Analyzing {selected_name}..."):
            # 1. Fetch Data [cite: 70]
            df = yf.download(ticker, start="2018-01-01", progress=False)
            if df.empty:
                st.error(f"Could not fetch data for `{ticker}`.")
                st.stop()

            # 2. Fix Dimension Error [cite: 18, 43, 64]
            # Use .reshape(-1, 1) to keep it 2D for the scaler, but access values correctly
            close_prices = df[['Close']].values 
            current_price = float(close_prices[-1][0]) # Extract scalar from the last row [cite: 70]

            # 3. Scaling [cite: 71]
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled = scaler.fit_transform(close_prices)

            # 4. Prepare Sequences [cite: 71]
            SEQ_LEN = 60
            X, Y = make_sequences(scaled, SEQ_LEN)
            
            if len(X) < 10:
                st.error("Not enough data to create sequences.")
                st.stop()

            # 5. Train/Test Split [cite: 71, 72]
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            Y_train, Y_test = Y[:split], Y[split:]

            # 6. Model Training (SVR with RBF Kernel) 
            model = SVR(kernel='rbf', C=1e3, gamma=0.1)
            model.fit(X_train, Y_train)

            # 7. Predictions 
            pred_scaled = model.predict(X_test).reshape(-1, 1)
            predictions = scaler.inverse_transform(pred_scaled)
            actual = scaler.inverse_transform(Y_test.reshape(-1, 1))

            # 8. Future Prediction [cite: 73]
            last_seq = scaled[-SEQ_LEN:].flatten().reshape(1, -1)
            next_price_scaled = model.predict(last_seq)
            next_price = float(scaler.inverse_transform(next_price_scaled.reshape(-1, 1))[0][0])
            
            price_change = ((next_price - current_price) / current_price) * 100
            
            # 9. Metrics [cite: 72, 75]
            rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
            accuracy = max(0, 100 - (rmse / current_price * 100))

            # UI Display
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f'<div class="metric-card"><div class="metric-value">₹{current_price:,.2f}</div><div class="metric-label">Current Price</div></div>', unsafe_allow_html=True)
            
            color = "#00d4aa" if next_price > current_price else "#ff6b6b"
            m2.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color}">₹{next_price:,.2f}</div><div class="metric-label">Predicted Price</div></div>', unsafe_allow_html=True)
            
            arrow = "▲" if price_change > 0 else "▼"
            m3.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color}">{arrow} {abs(price_change):.2f}%</div><div class="metric-label">Exp. Change</div></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="metric-card"><div class="metric-value">{accuracy:.1f}%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)

            # Chart [cite: 78, 79]
            st.markdown("#### 📈 Prediction Trend")
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('#0f1117')
            ax.set_facecolor('#1e2130')
            ax.plot(actual[-100:], color='#4fc3f7', label='Actual', linewidth=2)
            ax.plot(predictions[-100:], color='#00d4aa', label='Predicted', linestyle='--', linewidth=2)
            ax.legend()
            st.pyplot(fig)

with tab2:
    st.info("Model performance metrics are calculated using an 80/20 train-test split on historical data[cite: 81, 82].")

with tab3:
    st.write("This model utilizes a Support Vector Regressor (SVR) which is effective for non-linear time series data like stock prices[cite: 87].")
