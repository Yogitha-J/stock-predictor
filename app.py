import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@300;400;600&display=swap');

:root {
  --bg:         #03050a;
  --surface:    #080d17;
  --card:       #0b1220;
  --border:     #1a2540;
  --accent:     #00f5c3;
  --accent2:    #7b61ff;
  --danger:     #ff4d6d;
  --warn:       #ffd166;
  --text:       #e2e8f8;
  --muted:      #4a5568;
  --glow:       0 0 30px rgba(0,245,195,0.15);
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }

.hero { text-align: center; padding: 2.5rem 0 1.5rem; position: relative; }
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #00f5c3 0%, #7b61ff 50%, #ff4d6d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}
.hero-sub { font-size: 0.78rem; color: var(--muted); letter-spacing: 0.25em; text-transform: uppercase; }
.hero-line { height: 1px; background: linear-gradient(90deg, transparent, var(--accent), var(--accent2), transparent); margin: 1.5rem 0; opacity: 0.6; }

.stTabs [data-baseweb="tab-list"] { background: var(--surface) !important; border-radius: 12px !important; padding: 4px !important; border: 1px solid var(--border) !important; gap: 4px !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; border-radius: 8px !important; color: var(--muted) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.78rem !important; padding: 8px 20px !important; border: none !important; transition: all 0.2s ease !important; }
.stTabs [aria-selected="true"] { background: var(--card) !important; color: var(--accent) !important; border: 1px solid var(--border) !important; }

.stButton > button { background: linear-gradient(135deg, var(--accent), var(--accent2)) !important; color: #03050a !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; border: none !important; border-radius: 10px !important; padding: 0.65rem 1.5rem !important; transition: all 0.25s ease !important; text-transform: uppercase !important; }
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 30px rgba(0,245,195,0.35) !important; }

.metric-card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 1.4rem 1.2rem; text-align: center; position: relative; transition: all 0.3s ease; }
.metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--accent), var(--accent2)); }
.metric-val { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 800; color: var(--accent); line-height: 1; margin-bottom: 0.4rem; }
.metric-lbl { font-size: 0.68rem; color: var(--muted); letter-spacing: 0.15em; text-transform: uppercase; }

.signal-wrap { border-radius: 16px; padding: 2rem; text-align: center; position: relative; }
.signal-buy  { background: #001a12; border: 1.5px solid var(--accent); }
.signal-sell { background: #1a0008; border: 1.5px solid var(--danger); }
.signal-hold { background: #1a1400; border: 1.5px solid var(--warn);   }
.signal-text { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; }

.sec-header { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; color: var(--text); text-transform: uppercase; margin: 1.5rem 0 0.8rem; display: flex; align-items: center; gap: 10px; }
.sec-header::after { content: ''; flex: 1; height: 1px; background: var(--border); }

.info-box { background: var(--card); border-left: 3px solid var(--accent); border-radius: 0 10px 10px 0; padding: 1rem 1.2rem; font-size: 0.8rem; color: #a0b0c8; line-height: 1.7; margin: 0.8rem 0; }
.stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 1rem 0; }
.stat-item { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; text-align: center; }
.stat-val { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: var(--accent2); }

.pipeline { display: flex; flex-direction: column; gap: 0; margin: 1rem 0; }
.pipe-step { display: flex; align-items: flex-start; gap: 16px; padding: 1rem 0; border-bottom: 1px solid var(--border); }
.pipe-num { width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #03050a; font-family: 'Syne', sans-serif; font-size: 0.75rem; font-weight: 800; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.fancy-divider { height: 1px; background: linear-gradient(90deg, transparent, var(--border), transparent); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─── Helper Functions ─────────────────────────────────────────────────────────
def make_sequences(scaled_data, seq_len=60):
    X, Y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len:i].flatten())
        Y.append(scaled_data[i, 0])
    return np.array(X), np.array(Y)

def set_dark_chart(fig, axes):
    fig.patch.set_facecolor('#03050a')
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor('#080d17')
        ax.tick_params(colors='#4a5568', labelsize=8)
        ax.spines['bottom'].set_color('#1a2540')
        ax.spines['left'].set_color('#1a2540')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, color='#1a2540', linewidth=0.5, linestyle='--', alpha=0.6)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">QuantVision</div>
  <div class="hero-sub">AI · SVR Engine · NSE Markets · Real-time Inference</div>
  <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

if 'results' not in st.session_state:
    st.session_state.results = None

tab1, tab2, tab3 = st.tabs(["⚡ Live Prediction", "📊 Model Performance", "🧠 How It Works"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_input, col_info = st.columns([1, 2], gap="large")

    with col_input:
        st.markdown('<div class="sec-header">⚙ Configuration</div>', unsafe_allow_html=True)
        stock_options = {
            "TCS · Tata Consultancy": "TCS.NS",
            "Infosys": "INFY.NS",
            "Wipro": "WIPRO.NS",
            "Reliance Industries": "RELIANCE.NS",
            "HDFC Bank": "HDFCBANK.NS",
            "ICICI Bank": "ICICIBANK.NS",
            "HCL Technologies": "HCLTECH.NS",
        }
        selected_name = st.selectbox("Select stock", list(stock_options.keys()))
        ticker = stock_options[selected_name]

        custom = st.text_input("Custom NSE ticker", placeholder="e.g. ZOMATO.NS")
        if custom.strip():
            ticker = custom.strip().upper()
            selected_name = ticker

        seq_len_opt = st.select_slider("Lookback window (days)", options=[30, 45, 60, 90], value=60)
        predict_btn = st.button("⚡ Run Prediction", use_container_width=True, type="primary")

    with col_info:
        st.markdown('<div class="sec-header">📡 About this engine</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box"><b>SVR · RBF Kernel</b><br>Maps non-linear price moves into a higher-dimensional space to find the optimal hyperplane.</div>
        <div class="info-box"><b>Feature Window</b><br>Uses a sliding window of past closing prices as features for the regression model.</div>
        """, unsafe_allow_html=True)

    if predict_btn:
        with st.spinner(f"Fetching data for {ticker}..."):
            df = yf.download(ticker, start="2019-01-01", progress=False)
            if df.empty:
                st.error("Invalid Ticker.")
                st.stop()

            close_prices = df[['Close']].values
            current_price = float(close_prices[-1][0])
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(close_prices)

            X, Y = make_sequences(scaled, seq_len_opt)
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            Y_train, Y_test = Y[:split], Y[split:]

            model = SVR(kernel='rbf', C=1e3, gamma=0.1)
            model.fit(X_train, Y_train)

            pred_scaled = model.predict(X_test).reshape(-1, 1)
            predictions = scaler.inverse_transform(pred_scaled)
            actual = scaler.inverse_transform(Y_test.reshape(-1, 1))

            last_seq = scaled[-seq_len_opt:].flatten().reshape(1, -1)
            next_price = float(scaler.inverse_transform(model.predict(last_seq).reshape(-1, 1))[0][0])
            change = ((next_price - current_price) / current_price) * 100

            st.session_state.results = {
                "ticker": ticker, "name": selected_name, "actual": actual, "predictions": predictions,
                "current_price": current_price, "next_price": next_price, "price_change": change,
                "rmse": np.sqrt(mean_squared_error(actual, predictions)), "mae": mean_absolute_error(actual, predictions),
                "r2": r2_score(actual, predictions), "acc": max(0, 100 - (np.sqrt(mean_squared_error(actual, predictions))/current_price*100)),
                "X_train_len": len(X_train), "X_test_len": len(X_test), "seq_len": seq_len_opt
            }

    if st.session_state.results:
        r = st.session_state.results
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-val">₹{r["current_price"]:,.2f}</div><div class="metric-lbl">Current</div></div>', unsafe_allow_html=True)
        color = "#00f5c3" if r["price_change"] > 0 else "#ff4d6d"
        m2.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{color}">₹{r["next_price"]:,.2f}</div><div class="metric-lbl">Predicted</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{color}">{r["price_change"]:+.2f}%</div><div class="metric-lbl">Expected</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-val">{r["acc"]:.1f}%</div><div class="metric-lbl">Accuracy</div></div>', unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns([1, 1, 1])
        with sc2:
            if r["price_change"] > 0.5:
                st.markdown('<div class="signal-wrap signal-buy"><div class="signal-text" style="color:#00f5c3">BUY SIGNAL</div></div>', unsafe_allow_html=True)
            elif r["price_change"] < -0.5:
                st.markdown('<div class="signal-wrap signal-sell"><div class="signal-text" style="color:#ff4d6d">SELL SIGNAL</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="signal-wrap signal-hold"><div class="signal-text" style="color:#ffd166">HOLD SIGNAL</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-header">📈 Prediction vs Actual</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 4.5))
        set_dark_chart(fig, ax)
        ax.plot(r["actual"][-100:], color='#4fc3f7', label='Actual', lw=2)
        ax.plot(r["predictions"][-100:], color='#00f5c3', label='Predicted', lw=1.5, ls='--')
        ax.legend(facecolor='#0b1220', edgecolor='#1a2540', labelcolor='white')
        st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if st.session_state.results:
        r = st.session_state.results
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="sec-header">Error Metrics</div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="stat-grid">
                <div class="stat-item"><div class="stat-val">₹{r['rmse']:,.1f}</div><div class="stat-lbl">RMSE</div></div>
                <div class="stat-item"><div class="stat-val">₹{r['mae']:,.1f}</div><div class="stat-lbl">MAE</div></div>
                <div class="stat-item"><div class="stat-val">{r['r2']:.4f}</div><div class="stat-lbl">R² Score</div></div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown('<div class="sec-header">Residuals</div>', unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            set_dark_chart(fig2, ax2)
            ax2.hist((r['actual'] - r['predictions']).flatten(), bins=30, color='#7b61ff', alpha=0.7)
            st.pyplot(fig2)
    else:
        st.info("Run a prediction first.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — How It Works
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-header">🔬 Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline">
        <div class="pipe-step"><div class="pipe-num">01</div><div><div class="pipe-title">Data Ingestion</div><div class="pipe-desc">Fetch 5Y OHLCV data.</div></div></div>
        <div class="pipe-step"><div class="pipe-num">02</div><div><div class="pipe-title">Scaling</div><div class="pipe-desc">Normalize prices to [0,1].</div></div></div>
        <div class="pipe-step"><div class="pipe-num">03</div><div><div class="pipe-title">SVR Model</div><div class="pipe-desc">Apply RBF Kernel with C=1000.</div></div></div>
    </div>
    """, unsafe_allow_html=True)
