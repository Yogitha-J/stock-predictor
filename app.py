import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR

# ─── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="QuantVision Pro",
    page_icon="⚡",
    layout="wide"
)

# ─── HERO ───────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center;'>⚡ QuantVision Pro</h1>
<p style='text-align:center;color:gray;'>AI Stock Intelligence System</p>
<hr>
""", unsafe_allow_html=True)

# ─── INPUT ──────────────────────────────────────────────────
col1, col2 = st.columns([1,2])

with col1:
    stock = st.selectbox("Select Stock", [
        "TCS.NS","INFY.NS","RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS"
    ])

    seq_len = st.slider("Lookback Window", 30, 90, 60)

    run = st.button("🚀 Run Prediction")

with col2:
    st.info("This system predicts stock movement using SVR with RBF kernel and provides confidence + risk analysis.")

# ─── FUNCTION ───────────────────────────────────────────────
def make_sequences(data, seq_len):
    X, Y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i].flatten())
        Y.append(data[i])
    return np.array(X), np.array(Y)

# ─── MAIN ───────────────────────────────────────────────────
if run:

    # ── LOADING EXPERIENCE ──
    progress = st.progress(0)
    status = st.empty()

    steps = ["Fetching data...", "Scaling...", "Training model...", "Predicting..."]

    for i, step in enumerate(steps):
        status.text(step)
        for j in range(25):
            time.sleep(0.01)
            progress.progress(i*25 + j + 1)

    # ── DATA ──
    df = yf.download(stock, start="2018-01-01", progress=False)

    if df.empty:
        st.error("Invalid ticker")
        st.stop()

    close = df[['Close']].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    X, Y = make_sequences(scaled, seq_len)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    # ── MODEL ──
    model = SVR(kernel='rbf', C=1000, gamma=0.1, epsilon=0.01)
    model.fit(X_train, Y_train)

    pred_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(pred_scaled.reshape(-1,1))
    actual = scaler.inverse_transform(Y_test.reshape(-1,1))

    # ── NEXT PREDICTION ──
    last_seq = scaled[-seq_len:].flatten().reshape(1,-1)
    next_price = scaler.inverse_transform(model.predict(last_seq).reshape(-1,1))[0][0]
    current_price = close[-1][0]

    change = ((next_price - current_price)/current_price)*100

    # ── METRICS ──
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    r2 = r2_score(actual, predictions)

    confidence = max(0, min(100, r2 * 100))
    risk = "LOW" if rmse < 20 else "MEDIUM" if rmse < 50 else "HIGH"

    # ── STORY BANNER ──
    direction = "rise" if change > 0 else "fall"

    st.success(f"""
    📊 {stock} is expected to **{direction} by {abs(change):.2f}%** 
    based on historical pattern recognition.
    """)

    # ── METRICS ROW ──
    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Current Price", f"₹{current_price:.2f}")
    c2.metric("Predicted Price", f"₹{next_price:.2f}")
    c3.metric("Change %", f"{change:.2f}%")
    c4.metric("Confidence", f"{confidence:.1f}%")

    # ── RISK ──
    st.markdown(f"### ⚠ Risk Level: `{risk}`")

    # ── SCENARIO SIMULATION ──
    st.markdown("### 🔮 Scenario Simulation")

    shock = st.slider("Market Shock (%)", -5, 5, 0)
    sim_price = next_price * (1 + shock/100)

    st.metric("Simulated Price", f"₹{sim_price:.2f}")

    # ── INTERACTIVE CHART ──
    st.markdown("### 📈 Prediction vs Actual")

    df_plot = pd.DataFrame({
        "Actual": actual.flatten(),
        "Predicted": predictions.flatten()
    })

    fig = px.line(df_plot, template="plotly_dark")
    fig.update_layout(hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True)

    # ── RESIDUAL CHART ──
    st.markdown("### 📊 Residual Distribution")

    residuals = actual.flatten() - predictions.flatten()

    fig2 = px.histogram(residuals, nbins=40, template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

    # ── SCATTER ──
    st.markdown("### 🔍 Actual vs Predicted")

    fig3 = px.scatter(
        x=actual.flatten(),
        y=predictions.flatten(),
        template="plotly_dark"
    )

    fig3.add_shape(
        type="line",
        x0=min(actual)[0], y0=min(actual)[0],
        x1=max(actual)[0], y1=max(actual)[0]
    )

    st.plotly_chart(fig3, use_container_width=True)

    # ── EXPLAINABILITY ──
    st.markdown("### 🧠 Model Insight")

    trend = "UPTREND" if change > 0 else "DOWNTREND"

    st.info(f"""
    The model detected a **{trend} momentum** in the last {seq_len} days.
    SVR captured non-linear patterns resembling previous market behavior.
    """)

    st.caption("⚠ Not financial advice")
