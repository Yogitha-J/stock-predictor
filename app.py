import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
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

/* ── Root Variables ── */
:root {
  --bg:       #03050a;
  --surface:  #080d17;
  --card:     #0b1220;
  --border:   #1a2540;
  --accent:   #00f5c3;
  --accent2:  #7b61ff;
  --danger:   #ff4d6d;
  --warn:     #ffd166;
  --text:     #e2e8f8;
  --muted:    #4a5568;
  --glow:     0 0 30px rgba(0,245,195,0.15);
}

/* ── Base ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }

/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    position: relative;
}
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
.hero-sub {
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.25em;
    text-transform: uppercase;
}
.hero-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), var(--accent2), transparent);
    margin: 1.5rem 0;
    opacity: 0.6;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    padding: 8px 20px !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: var(--card) !important;
    color: var(--accent) !important;
    border: 1px solid var(--border) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #03050a !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,245,195,0.35) !important;
}

/* ── Inputs / Selects ── */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,245,195,0.15) !important;
}

/* ── Metric Cards ── */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-card:hover {
    border-color: rgba(0,245,195,0.3);
    transform: translateY(-3px);
    box-shadow: var(--glow);
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
    margin-bottom: 0.4rem;
}
.metric-lbl {
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* ── Signal Cards ── */
.signal-wrap {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.signal-buy  { background: #001a12; border: 1.5px solid var(--accent); }
.signal-sell { background: #1a0008; border: 1.5px solid var(--danger); }
.signal-hold { background: #1a1400; border: 1.5px solid var(--warn);   }
.signal-icon { font-size: 2.8rem; display: block; margin-bottom: 0.3rem; }
.signal-text {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: 0.05em;
}
.signal-desc { font-size: 0.75rem; color: var(--muted); margin-top: 0.5rem; letter-spacing: 0.1em; }

/* ── Section Headers ── */
.sec-header {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Info Box ── */
.info-box {
    background: var(--card);
    border-left: 3px solid var(--accent);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    font-size: 0.8rem;
    color: #a0b0c8;
    line-height: 1.7;
    margin: 0.8rem 0;
}
.info-box b { color: var(--accent); }

/* ── Stat Row ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 1rem 0;
}
.stat-item {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent2);
}
.stat-lbl {
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Pipeline Steps ── */
.pipeline {
    display: flex;
    flex-direction: column;
    gap: 0;
    margin: 1rem 0;
}
.pipe-step {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    padding: 1rem 0;
    border-bottom: 1px solid var(--border);
}
.pipe-step:last-child { border-bottom: none; }
.pipe-num {
    width: 32px; height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #03050a;
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 800;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
}
.pipe-title { font-size: 0.85rem; font-weight: 600; color: var(--text); margin-bottom: 3px; }
.pipe-desc  { font-size: 0.75rem; color: var(--muted); line-height: 1.6; }

/* ── Divider ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1.5rem 0;
}

/* ── Stagger animation ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-up { animation: fadeUp 0.5s ease forwards; }
.delay-1 { animation-delay: 0.08s; opacity: 0; }
.delay-2 { animation-delay: 0.16s; opacity: 0; }
.delay-3 { animation-delay: 0.24s; opacity: 0; }
.delay-4 { animation-delay: 0.32s; opacity: 0; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">QuantVision</div>
  <div class="hero-sub">AI · SVR Engine · NSE Markets · Real-time Inference</div>
  <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["⚡  Live Prediction", "📊  Model Performance", "🧠  How It Works"])

# ─── Helper ───────────────────────────────────────────────────────────────────
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

# ─── Shared session state so tab2 can use results ─────────────────────────────
if 'results' not in st.session_state:
    st.session_state.results = None

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_input, col_info = st.columns([1, 2], gap="large")

    with col_input:
        st.markdown('<div class="sec-header">⚙ Configuration</div>', unsafe_allow_html=True)
        stock_options = {
            "TCS · Tata Consultancy": "TCS.NS",
            "Infosys":                 "INFY.NS",
            "Wipro":                   "WIPRO.NS",
            "Reliance Industries":     "RELIANCE.NS",
            "HDFC Bank":               "HDFCBANK.NS",
            "ICICI Bank":              "ICICIBANK.NS",
            "HCL Technologies":        "HCLTECH.NS",
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
<div class="info-box fade-up">
<b>SVR · RBF Kernel</b><br>
Support Vector Regression with a Radial Basis Function kernel maps complex 
non-linear price movements into a higher-dimensional feature space, finding 
the optimal regression hyperplane with maximum margin.
</div>
<div class="info-box fade-up delay-1">
<b>Feature Window</b><br>
Each prediction uses a sliding window of the last <i>N</i> closing prices 
(default 60 days) as a flat feature vector. The model learns temporal 
dependencies across this window.
</div>
<div class="info-box fade-up delay-2">
<b>Data Source</b><br>
5 years of daily OHLCV data via Yahoo Finance. Prices are MinMax scaled 
to [0,1] before training and inverse-transformed for output.
</div>
        """, unsafe_allow_html=True)

    # ── Run ──
    if predict_btn:
        with st.spinner(f"Fetching & training on {selected_name} ..."):
            df = yf.download(ticker, start="2018-01-01", progress=False)
            if df.empty:
                st.error(f"Could not fetch data for `{ticker}`. Check the ticker symbol.")
                st.stop()

            close_prices = df[['Close']].values
            current_price = float(close_prices[-1][0])

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(close_prices)

            SEQ_LEN = seq_len_opt
            X, Y = make_sequences(scaled, SEQ_LEN)

            if len(X) < 20:
                st.error("Not enough historical data. Try a different ticker.")
                st.stop()

            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            Y_train, Y_test = Y[:split], Y[split:]

            model = SVR(kernel='rbf', C=1e3, gamma=0.1, epsilon=0.01)
            model.fit(X_train, Y_train)

            pred_scaled  = model.predict(X_test).reshape(-1, 1)
            predictions  = scaler.inverse_transform(pred_scaled)
            actual        = scaler.inverse_transform(Y_test.reshape(-1, 1))

            last_seq       = scaled[-SEQ_LEN:].flatten().reshape(1, -1)
            next_scaled    = model.predict(last_seq)
            next_price     = float(scaler.inverse_transform(next_scaled.reshape(-1, 1))[0][0])
            price_change   = ((next_price - current_price) / current_price) * 100

            rmse  = float(np.sqrt(mean_squared_error(actual, predictions)))
            mae   = float(mean_absolute_error(actual, predictions))
            r2    = float(r2_score(actual, predictions))
            acc   = max(0, 100 - (rmse / current_price * 100))

            # Store for tab2
            st.session_state.results = dict(
                ticker=ticker, name=selected_name,
                actual=actual, predictions=predictions,
                current_price=current_price, next_price=next_price,
                price_change=price_change,
                rmse=rmse, mae=mae, r2=r2, acc=acc,
                df=df, seq_len=SEQ_LEN,
                X_train_len=len(X_train), X_test_len=len(X_test)
            )

        r = st.session_state.results
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

        # ── Metrics row ──
        color    = "#00f5c3" if r['price_change'] > 0 else "#ff4d6d"
        arrow    = "▲" if r['price_change'] > 0 else "▼"
        m1, m2, m3, m4 = st.columns(4)

        m1.markdown(f"""<div class="metric-card fade-up">
            <div class="metric-val">₹{r['current_price']:,.2f}</div>
            <div class="metric-lbl">Current Price</div></div>""", unsafe_allow_html=True)
        m2.markdown(f"""<div class="metric-card fade-up delay-1">
            <div class="metric-val" style="color:{color}">₹{r['next_price']:,.2f}</div>
            <div class="metric-lbl">Predicted Next</div></div>""", unsafe_allow_html=True)
        m3.markdown(f"""<div class="metric-card fade-up delay-2">
            <div class="metric-val" style="color:{color}">{arrow} {abs(r['price_change']):.2f}%</div>
            <div class="metric-lbl">Expected Change</div></div>""", unsafe_allow_html=True)
        m4.markdown(f"""<div class="metric-card fade-up delay-3">
            <div class="metric-val">{r['acc']:.1f}%</div>
            <div class="metric-lbl">Model Accuracy</div></div>""", unsafe_allow_html=True)

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

        # ── Signal ──
        sc1, sc2, sc3 = st.columns([1, 1, 1])
        if r['price_change'] > 0.5:
            sc2.markdown(f"""<div class="signal-wrap signal-buy fade-up">
                <span class="signal-icon">📈</span>
                <div class="signal-text" style="color:#00f5c3">BUY SIGNAL</div>
                <div class="signal-desc">MODEL FORECASTS UPWARD MOVEMENT</div>
            </div>""", unsafe_allow_html=True)
        elif r['price_change'] < -0.5:
            sc2.markdown(f"""<div class="signal-wrap signal-sell fade-up">
                <span class="signal-icon">📉</span>
                <div class="signal-text" style="color:#ff4d6d">SELL SIGNAL</div>
                <div class="signal-desc">MODEL FORECASTS DOWNWARD MOVEMENT</div>
            </div>""", unsafe_allow_html=True)
        else:
            sc2.markdown(f"""<div class="signal-wrap signal-hold fade-up">
                <span class="signal-icon">⏸</span>
                <div class="signal-text" style="color:#ffd166">HOLD SIGNAL</div>
                <div class="signal-desc">MODEL FORECASTS SIDEWAYS MOVEMENT</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

        # ── Chart ──

st.markdown("""
    <style>
    .chart-container {
        border-radius: 10px;
        padding: 15px;
        background-color: #111727;
        border: 1px solid #2d323e;
        margin-bottom: 20px;
    }
    .sec-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if st.session_state.results is None:
        st.markdown("""
        <div class="info-box" style="text-align:center;padding:2rem;border-left:3px solid #7b61ff">
            <span style="font-size:2rem">📊</span><br><br>
            <b style="color:#e2e8f8;font-size:1rem">No results yet</b><br>
            <span style="color:#4a5568">Run a prediction in the <b>Live Prediction</b> tab first.</span>
        </div>""", unsafe_allow_html=True)
    else:
        r = st.session_state.results
        st.markdown(f'<div class="sec-header">📊 Performance Report · {r["name"]}</div>', unsafe_allow_html=True)

        # ── Error Metrics ──
        mape = float(np.mean(np.abs((r['actual'] - r['predictions']) / r['actual'])) * 100)
        max_err = float(np.max(np.abs(r['actual'] - r['predictions'])))
        residuals = (r['actual'] - r['predictions']).flatten()

        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            st.markdown('<div class="sec-header" style="font-size:0.78rem">Error Metrics</div>', unsafe_allow_html=True)
            # Unified block to prevent raw text display
            st.markdown(f"""
            <div class="stat-grid">
              <div class="stat-item"><div class="stat-val">₹{r['rmse']:,.1f}</div><div class="stat-lbl">RMSE</div></div>
              <div class="stat-item"><div class="stat-val">₹{r['mae']:,.1f}</div><div class="stat-lbl">MAE</div></div>
              <div class="stat-item"><div class="stat-val">{mape:.2f}%</div><div class="stat-lbl">MAPE</div></div>
              <div class="stat-item"><div class="stat-val">{r['r2']:.4f}</div><div class="stat-lbl">R² Score</div></div>
              <div class="stat-item"><div class="stat-val">{r['acc']:.1f}%</div><div class="stat-lbl">Accuracy</div></div>
              <div class="stat-item"><div class="stat-val">₹{max_err:,.1f}</div><div class="stat-lbl">Max Error</div></div>
            </div>
            <div class="info-box" style="margin-top:1rem">
                <b>Metrics Guide:</b> RMSE penalizes large outliers, while MAE shows average "real-world" deviation.<br>
                An <b>R² Score</b> near 1.0 indicates excellent trend capture.
            </div>""", unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="sec-header" style="font-size:0.78rem">Residuals Distribution</div>', unsafe_allow_html=True)
            
            fig_hist = px.histogram(
                residuals, nbins=40, 
                color_discrete_sequence=['#7b61ff'],
                opacity=0.7,
                labels={'value': 'Error Amount (₹)'}
            )
            fig_hist.update_layout(
                template="plotly_dark", height=220, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="#00f5c3")
            st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

            res_bias = "under-prediction" if residuals.mean() > 0 else "over-prediction"
            st.markdown(f"""
            <div class="info-box">
                Current mean residual: <b style="color:#00f5c3">₹{residuals.mean():.2f}</b><br>
                <i>Slight {res_bias}</i>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

        # ── 1. Full Prediction Chart ──
        st.markdown('<div class="sec-header">Full Test Period — Actual vs Predicted</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(y=r['actual'].flatten(), name='Actual', line=dict(color='#4fc3f7', width=2), fill='tozeroy', fillcolor='rgba(79, 195, 247, 0.05)'))
        fig3.add_trace(go.Scatter(y=r['predictions'].flatten(), name='Predicted', line=dict(color='#00f5c3', width=1.5, dash='dash')))
        fig3.update_layout(
            template="plotly_dark", height=350, margin=dict(l=0, r=0, t=20, b=0),
            hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig3, use_container_width=True)
        # ── 2. Scatter: Actual vs Predicted (Fixed Labels) ──
        sc_col1, sc_col2 = st.columns([2, 1])
        with sc_col1:
            st.markdown('<div class="sec-header" style="font-size:0.78rem">Scatter · Price Correlation</div>', unsafe_allow_html=True)
            # labels param fixes the hover, update_layout fixes the axis titles
            fig4 = px.scatter(
                x=r['actual'].flatten(), 
                y=r['predictions'].flatten(), 
                opacity=0.4,
                labels={'x': 'Actual Price', 'y': 'Predicted Price'}
            )
            fig4.add_shape(type="line", x0=r['actual'].min(), y0=r['actual'].min(), x1=r['actual'].max(), y1=r['actual'].max(), line=dict(color="#00f5c3", dash="dash"))
            fig4.update_traces(marker=dict(color='#7b61ff', size=6))
            fig4.update_layout(
                template="plotly_dark", height=300, margin=dict(l=0, r=0, t=10, b=30), 
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Actual Market Price (₹)",
                yaxis_title="Model Prediction (₹)"
            )
            st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})

        with sc_col2:
            st.markdown(f"""
            <div class="info-box" style="margin-top:2.5rem">
            <b>Model Confidence:</b><br>
            Tight clustering near the diagonal line indicates high reliability.<br><br>
            <b>Data Specs:</b><br>
            • Test samples: {r['X_test_len']:,} days<br>
            • Lookback: {r['seq_len']} days
            </div>""", unsafe_allow_html=True)

        # ── 3. Rolling Error ──
        st.markdown('<div class="sec-header">Rolling 20-Day MAE (Volatility Analysis)</div>', unsafe_allow_html=True)
        rolling_err = pd.Series(np.abs(residuals)).rolling(20).mean()
        
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(y=rolling_err, name='Rolling MAE', line=dict(color='#ff4d6d', width=2), fill='tozeroy', fillcolor='rgba(255, 77, 109, 0.1)'))
        fig5.update_layout(
            template="plotly_dark", height=250, margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis_title="Avg Error (₹)"
        )
        st.plotly_chart(fig5, use_container_width=True)
# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — How It Works
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns([3, 2], gap="large")

    with c1:
        st.markdown('<div class="sec-header">🔬 Model Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="info-box">
<b>Support Vector Regression (SVR)</b> is a supervised learning algorithm derived 
from Support Vector Machines. Unlike SVMs for classification, SVR fits a 
hyperplane in a high-dimensional space that deviates from the true values 
by at most ε (epsilon). It is robust to outliers and generalizes well on 
financial time-series with limited data.
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-header" style="font-size:0.78rem">Processing Pipeline</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="pipeline">
  <div class="pipe-step">
    <div class="pipe-num">01</div>
    <div><div class="pipe-title">Data Ingestion</div>
    <div class="pipe-desc">5 years of daily OHLCV data is downloaded from Yahoo Finance. 
    Only the closing price series is used as the prediction target.</div></div>
  </div>
  <div class="pipe-step">
    <div class="pipe-num">02</div>
    <div><div class="pipe-title">MinMax Scaling</div>
    <div class="pipe-desc">All closing prices are normalized to [0, 1] using MinMaxScaler. 
    This prevents large price values from dominating the kernel distance 
    computation and speeds up convergence.</div></div>
  </div>
  <div class="pipe-step">
    <div class="pipe-num">03</div>
    <div><div class="pipe-title">Sliding Window Sequencing</div>
    <div class="pipe-desc">A 60-day lookback window creates flat feature vectors. 
    For day t, the input X = [close(t-60), close(t-59), ..., close(t-1)] 
    and target Y = close(t). This gives the model temporal context.</div></div>
  </div>
  <div class="pipe-step">
    <div class="pipe-num">04</div>
    <div><div class="pipe-title">Train / Test Split (80/20)</div>
    <div class="pipe-desc">Data is split chronologically — first 80% for training, 
    last 20% for evaluation. Random shuffling is not used to preserve 
    temporal ordering and prevent data leakage.</div></div>
  </div>
  <div class="pipe-step">
    <div class="pipe-num">05</div>
    <div><div class="pipe-title">SVR Training (RBF Kernel)</div>
    <div class="pipe-desc">The RBF (Radial Basis Function) kernel maps inputs into 
    infinite-dimensional space. Hyperparameters: C=1000 (regularization), 
    γ=0.1 (kernel bandwidth), ε=0.01 (tube width).</div></div>
  </div>
  <div class="pipe-step">
    <div class="pipe-num">06</div>
    <div><div class="pipe-title">Inference & Inverse Transform</div>
    <div class="pipe-desc">The last 60 scaled prices form the live input. The model 
    outputs a scaled prediction which is inverse-transformed back to 
    actual ₹ price using the fitted scaler.</div></div>
  </div>
</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="sec-header">⚙ Hyperparameters</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="stat-grid" style="grid-template-columns:1fr 1fr">
  <div class="stat-item"><div class="stat-val" style="font-size:1.1rem">RBF</div><div class="stat-lbl">Kernel</div></div>
  <div class="stat-item"><div class="stat-val" style="font-size:1.1rem">1000</div><div class="stat-lbl">C (regularization)</div></div>
  <div class="stat-item"><div class="stat-val" style="font-size:1.1rem">0.1</div><div class="stat-lbl">Gamma (γ)</div></div>
  <div class="stat-item"><div class="stat-val" style="font-size:1.1rem">0.01</div><div class="stat-lbl">Epsilon (ε)</div></div>
  <div class="stat-item"><div class="stat-val" style="font-size:1.1rem">60</div><div class="stat-lbl">Lookback (days)</div></div>
  <div class="stat-item"><div class="stat-val" style="font-size:1.1rem">80/20</div><div class="stat-lbl">Train/Test split</div></div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-header" style="margin-top:1.5rem">📐 Key Formulas</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="info-box" style="font-size:0.75rem">
<b style="color:#7b61ff">RBF Kernel:</b><br>
K(x, x') = exp(−γ ‖x − x'‖²)<br><br>
<b style="color:#7b61ff">SVR Objective:</b><br>
Minimize ½‖w‖² + C Σξᵢ<br>
subject to |yᵢ − f(xᵢ)| ≤ ε + ξᵢ<br><br>
<b style="color:#7b61ff">RMSE:</b><br>
√( (1/n) Σ(ŷᵢ − yᵢ)² )<br><br>
<b style="color:#7b61ff">MAPE:</b><br>
(100/n) Σ |yᵢ − ŷᵢ| / |yᵢ|<br><br>
<b style="color:#7b61ff">R²:</b><br>
1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-header" style="margin-top:1.5rem">⚠ Limitations</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="info-box" style="border-left-color:#ff4d6d">
<b style="color:#ff4d6d">SVR does not know:</b><br>
• News events or earnings surprises<br>
• Macroeconomic changes (RBI rates, inflation)<br>
• Market sentiment or FII flows<br>
• Corporate actions (splits, dividends)<br><br>
It only learns from past closing prices. Use as one 
signal among many — never as sole trading advice.
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-header">📚 Why SVR for Stock Prediction?</div>', unsafe_allow_html=True)
    col3a, col3b, col3c = st.columns(3)
    col3a.markdown("""
<div class="info-box">
<b>vs LSTM / RNN</b><br>
SVR trains much faster with small datasets. 
LSTM needs thousands of samples and GPUs 
to outperform SVR. For short sequences, 
SVR is often competitive.
</div>""", unsafe_allow_html=True)
    col3b.markdown("""
<div class="info-box">
<b>vs Linear Regression</b><br>
Linear models assume price is a linear 
combination of past prices. The RBF kernel 
lets SVR capture non-linear patterns like 
momentum, reversals, and volatility clustering.
</div>""", unsafe_allow_html=True)
    col3c.markdown("""
<div class="info-box">
<b>vs ARIMA</b><br>
ARIMA assumes stationarity and models 
autocorrelation explicitly. SVR makes no 
such assumptions — it learns the mapping 
function directly from data, making it 
more flexible for trending markets.
</div>""", unsafe_allow_html=True)
