import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

# Setup
warnings.filterwarnings("ignore")
st.set_page_config(page_title="AlphaPredict AI", page_icon="📈", layout="wide")

# =============================
# ✨ MAGICAL UI CUSTOMIZATION
# =============================
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(to bottom, #0f172a, #1e293b);
        color: #f8fafc;
    }
    /* Glassmorphism Cards */
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        transition: transform 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-5px);
        border-color: #38bdf8;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #0284c7, #38bdf8);
        border: none;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# 🧠 BRAIN: LOGIC & PROCESSING
# =============================
def make_sequences(scaled_data, seq_len=60):
    X, Y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i].flatten()) 
        Y.append(scaled_data[i, 0])
    return np.array(X), np.array(Y)

# =============================
# 🖥️ UI: LAYOUT
# =============================
st.title("📈 AlphaPredict Stock AI")
st.markdown("---")

tab1, tab2 = st.tabs(["🔮 Forecast Terminal", "📚 Model Strategy"])

with tab1:
    col_input, col_chart = st.columns([1, 3], gap="large")
    
    with col_input:
        st.subheader("Configuration")
        stock_options = {
            "Reliance Ind.": "RELIANCE.NS",
            "TCS": "TCS.NS",
            "HDFC Bank": "HDFCBANK.NS",
            "Infosys": "INFY.NS",
            "Zomato": "ZOMATO.NS"
        }
        selected_name = st.selectbox("Select Asset", list(stock_options.keys()))
        ticker = stock_options[selected_name]
        
        custom = st.text_input("Custom NSE Ticker", placeholder="AAPL")
        if custom.strip():
            ticker = custom.strip().upper()
            selected_name = ticker

        predict_btn = st.button("RUN AI ANALYSIS")

    with col_chart:
        if predict_btn:
            with st.spinner("Processing Deep Market Data..."):
                # 1. DATA FETCHING (Fixed for 2024-2026 yfinance versions)
                df = yf.download(ticker, start="2019-01-01", progress=False)
                
                if df.empty:
                    st.error("Ticker not found. Please check the symbol.")
                else:
                    # FIX: Handle Multi-Index columns
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    
                    data = df[['Close']].copy()
                    current_price = float(data.iloc[-1])
                    
                    # 2. PREPROCESSING
                    scaler = MinMaxScaler(feature_range=(0,1))
                    scaled_data = scaler.fit_transform(data)
                    
                    SEQ_LEN = 60
                    X, Y = make_sequences(scaled_data, SEQ_LEN)
                    
                    # 3. TRAINING (SVR)
                    split = int(0.9 * len(X))
                    X_train, Y_train = X[:split], Y[:split]
                    X_test, Y_test = X[split:], Y[split:]
                    
                    model = SVR(kernel='rbf', C=1e3, gamma=0.1)
                    model.fit(X_train, Y_train)
                    
                    # 4. PREDICTIONS
                    pred_scaled = model.predict(X_test).reshape(-1, 1)
                    predictions = scaler.inverse_transform(pred_scaled)
                    actual = scaler.inverse_transform(Y_test.reshape(-1, 1))
                    
                    # 5. FUTURE PRICE
                    last_seq = scaled_data[-SEQ_LEN:].flatten().reshape(1, -1)
                    next_price_scaled = model.predict(last_seq)
                    next_price = float(scaler.inverse_transform(next_price_scaled.reshape(-1, 1))[0][0])
                    
                    # 📊 METRICS DISPLAY
                    m_col1, m_col2, m_col3 = st.columns(3)
                    
                    with m_col1:
                        st.markdown(f'<div class="metric-container"><div class="metric-label">Current Price</div><div class="metric-value">₹{current_price:,.2f}</div></div>', unsafe_allow_html=True)
                    
                    change = ((next_price - current_price) / current_price) * 100
                    color = "#4ade80" if change > 0 else "#f87171"
                    
                    with m_col2:
                        st.markdown(f'<div class="metric-container"><div class="metric-label">AI Forecast</div><div class="metric-value" style="color:{color}">₹{next_price:,.2f}</div></div>', unsafe_allow_html=True)
                    
                    with m_col3:
                        arrow = "▲" if change > 0 else "▼"
                        st.markdown(f'<div class="metric-container"><div class="metric-label">Expected Move</div><div class="metric-value" style="color:{color}">{arrow} {abs(change):.2f}%</div></div>', unsafe_allow_html=True)

                    # 📉 CHARTING
                    st.markdown("### Market Movement vs AI Prediction")
                    fig, ax = plt.subplots(figsize=(12, 5))
                    fig.patch.set_facecolor('#0f172a')
                    ax.set_facecolor('#1e293b')
                    
                    ax.plot(actual, color='#94a3b8', alpha=0.5, label='Actual Price', linewidth=2)
                    ax.plot(predictions, color='#38bdf8', label='AI Prediction', linewidth=2)
                    
                    ax.tick_params(colors='white')
                    ax.spines['bottom'].set_color('white')
                    ax.spines['left'].set_color('white')
                    ax.legend(facecolor='#1e293b', labelcolor='white')
                    plt.grid(color='rgba(255,255,255,0.05)')
                    
                    st.pyplot(fig)
        else:
            st.info("👈 Enter a ticker and click 'Run AI Analysis' to begin.")

with tab2:
    st.markdown("""
    ### How the AlphaPredict Engine Works
    This system uses a **Support Vector Regressor (SVR)** with a Radial Basis Function (RBF) kernel.
    
    1. **Windowing:** The model looks at the last **60 trading days** as a single pattern.
    2. **Pattern Matching:** It maps these 60-day patterns to the 61st day's price.
    3. **Non-Linearity:** Unlike basic trend lines, SVR finds a high-dimensional curve that fits complex market fluctuations.
    """)
