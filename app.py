import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantVision Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Enhanced CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@300;400;600&display=swap');

:root {
  --bg: #03050a;
  --surface: rgba(13, 17, 23, 0.8);
  --card-bg: rgba(255, 255, 255, 0.03);
  --accent: #00f5c3;
  --accent-glow: rgba(0, 245, 195, 0.4);
}

/* Glassmorphism Cards */
.st-emotion-cache-1r6slb0, .metric-card {
    background: var(--card-bg) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    background: linear-gradient(to right, #00f5c3, #7b61ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    height: 40px;
    border-radius: 20px !important;
    background-color: var(--card-bg) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #080d17 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

</style>
""", unsafe_allow_html=True)

# ─── Sidebar Config ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛠 Engine Config")
    stock_options = {
        "TCS · Tata Consultancy": "TCS.NS",
        "Infosys": "INFY.NS",
        "Wipro": "WIPRO.NS",
        "Reliance Industries": "RELIANCE.NS",
        "HDFC Bank": "HDFCBANK.NS",
    }
    selected_name = st.selectbox("Market Asset", list(stock_options.keys()))
    ticker = stock_options[selected_name]
    
    custom = st.text_input("Custom NSE Ticker", placeholder="e.g., SBIN.NS")
    if custom.strip():
        ticker = custom.strip().upper()
        selected_name = ticker

    st.divider()
    seq_len = st.select_slider("Temporal Window (Days)", options=[30, 60, 90], value=60)
    predict_btn = st.button("⚡ EXECUTE INFERENCE", use_container_width=True, type="primary")
    
    st.markdown("---")
    st.caption("v2.1.0 · Powered by SVR-RBF")

# ─── Hero Section ─────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">QuantVision Pro</div>', unsafe_allow_html=True)
st.markdown('<p style="color:#8892b0; letter-spacing: 2px;">NEURAL ARCHITECTURE FOR NSE MARKET VOLATILITY</p>', unsafe_allow_html=True)

# ─── Core Logic ───────────────────────────────────────────────────────────────
def make_sequences(data, length):
    X, Y = [], []
    for i in range(length, len(data)):
        X.append(data[i-length:i].flatten())
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

if 'results' not in st.session_state:
    st.session_state.results = None

tab1, tab2, tab3 = st.tabs(["🔮 Live Terminal", "📈 Analytics", "🧪 Documentation"])

# ─── Tab 1: Terminal ──────────────────────────────────────────────────────────
with tab1:
    if predict_btn:
        with st.status("Initializing Quant Engine...", expanded=True) as status:
            st.write("Fetching historical OHLCV data...")
            df = yf.download(ticker, start="2019-01-01", progress=False)
            
            st.write("Normalizing tensors...")
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[['Close']].values)
            
            X, Y = make_sequences(scaled_data, seq_len)
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            Y_train, Y_test = Y[:split], Y[split:]
            
            st.write("Fitting SVR Hyperplane...")
            model = SVR(kernel='rbf', C=1e3, gamma=0.1)
            model.fit(X_train, Y_train)
            
            # Predictions
            pred_scaled = model.predict(X_test).reshape(-1, 1)
            next_scaled = model.predict(scaled_data[-seq_len:].flatten().reshape(1, -1))
            
            # Inverse
            predictions = scaler.inverse_transform(pred_scaled)
            actual = scaler.inverse_transform(Y_test.reshape(-1, 1))
            next_price = float(scaler.inverse_transform(next_scaled.reshape(-1, 1))[0][0])
            curr_price = float(df['Close'].iloc[-1])
            
            st.session_state.results = {
                "curr": curr_price, "next": next_price, 
                "actual": actual, "pred": predictions,
                "df": df, "change": ((next_price - curr_price)/curr_price)*100
            }
            status.update(label="Inference Complete", state="complete", expanded=False)

    if st.session_state.results:
        res = st.session_state.results
        
        # Metric Grid
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Asset Value", f"₹{res['curr']:,.2f}")
        
        delta_color = "normal" if res['change'] > 0 else "inverse"
        c2.metric("Target Projection", f"₹{res['next']:,.2f}", f"{res['change']:.2f}%", delta_color=delta_color)
        
        # Advanced Plotly Chart
        st.markdown("### 📊 Interactive Flux Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=res['actual'].flatten()[-100:], name="Ground Truth", line=dict(color='#4fc3f7', width=2)))
        fig.add_trace(go.Scatter(y=res['pred'].flatten()[-100:], name="AI Prediction", line=dict(color='#00f5c3', width=2, dash='dot')))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            height=400,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Select a ticker and press 'Execute Inference' in the sidebar to begin.")

# ─── Tab 3: Documentation (Educational Section) ───────────────────────────────
with tab3:
    st.markdown("### 🧠 How the SVR Engine Thinks")
    st.markdown("""
    The Support Vector Regressor (SVR) doesn't just look at the last price; it maps a **60-day window** into a high-dimensional space to find the 'tube' of best fit.
    """)
    
    # Triggering educational diagram for user understanding of SVR
    st.write("This diagram illustrates how the SVR model maintains an 'Epsilon-tube' to allow for minor fluctuations while capturing the broader trend:")
    

    st.markdown("""
    - **Kernel (RBF):** Handles non-linear market "noise."
    - **C Parameter:** Controls the trade-off between error and model simplicity.
    - **Epsilon:** Defines the margin of tolerance where no penalty is given to errors.
    """)
