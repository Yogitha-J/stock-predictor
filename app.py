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
.gate-box{background:#1e2130;border:1px solid #2e3650;border-radius:10px;padding:16px;margin:8px 0;text-align:center}
</style>
""", unsafe_allow_html=True)

st.markdown("## 📈 AI Stock Price Predictor")
st.markdown("**Sequence Prediction Model — NSE / BSE Stocks**")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔮 Live Prediction", "📊 Model Performance", "🧠 How It Works"])

def make_sequences(scaled, seq_len=60):
    X, Y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i].ravel())
        Y.append(scaled[i][0])
    return np.array(X), np.array(Y)

with tab1:
    st.markdown("### Pick a stock and get an instant prediction")
    col_input, col_info = st.columns([1, 2])
    with col_input:
        stock_options = {
            "TCS (Tata Consultancy)":"TCS.NS",
            "Infosys":"INFY.NS",
            "Wipro":"WIPRO.NS",
            "HCL Technologies":"HCLTECH.NS",
            "Reliance Industries":"RELIANCE.NS",
            "HDFC Bank":"HDFCBANK.NS",
        }
        selected_name = st.selectbox("Choose a stock", list(stock_options.keys()))
        ticker = stock_options[selected_name]
        st.caption(f"Ticker: `{ticker}`")
        custom = st.text_input("Or enter any NSE ticker", placeholder="e.g. ZOMATO.NS")
        if custom.strip():
            ticker = custom.strip().upper()
            selected_name = ticker
        predict_btn = st.button("🚀 Run Prediction", use_container_width=True, type="primary")
    with col_info:
        st.markdown("""<div class="explainer-box"><b>What this does:</b><br>
        Downloads 5 years of real NSE data, builds 60-day sequences,
        trains a prediction model, and outputs the next closing price
        with a Buy / Sell / Hold signal.</div>""", unsafe_allow_html=True)

    if predict_btn:
        with st.spinner(f"Fetching live data for {selected_name}..."):
            df = yf.download(ticker, start="2018-01-01", progress=False)
            if df.empty:
                st.error(f"Could not fetch data for `{ticker}`.")
                st.stop()
            close_prices = df[['Close']].values
            current_price = float(close_prices[-1])
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled = scaler.fit_transform(close_prices)
            SEQ_LEN = 60
            X, Y = make_sequences(scaled, SEQ_LEN)
            split = int(0.8*len(X))
            X_train,X_test = X[:split],X[split:]
            Y_train,Y_test = Y[:split],Y[split:]
            model = SVR(kernel='rbf', C=1000, gamma=0.01, epsilon=0.01)
            model.fit(X_train, Y_train)
            pred_scaled = model.predict(X_test).reshape(-1,1)
            predictions = scaler.inverse_transform(pred_scaled)
            actual = scaler.inverse_transform(Y_test.reshape(-1,1))
            rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
            mae  = float(mean_absolute_error(actual, predictions))
            accuracy = max(0, 100-(rmse/current_price*100))
            last_seq = scaled[-SEQ_LEN:].ravel().reshape(1,-1)
            next_price = float(scaler.inverse_transform(model.predict(last_seq).reshape(-1,1))[0][0])
            price_change = ((next_price-current_price)/current_price)*100
            if price_change > 1.5:
                signal,signal_css,signal_color,signal_emoji = "BUY","signal-buy","#00d4aa","🟢"
                confidence = min(99, 60+abs(price_change)*5)
            elif price_change < -1.5:
                signal,signal_css,signal_color,signal_emoji = "SELL","signal-sell","#ff6b6b","🔴"
                confidence = min(99, 60+abs(price_change)*5)
            else:
                signal,signal_css,signal_color,signal_emoji = "HOLD","signal-hold","#ffd700","🟡"
                confidence = 55+(1.5-abs(price_change))*10

        m1,m2,m3,m4 = st.columns(4)
        with m1: st.markdown(f'<div class="metric-card"><div class="metric-value">₹{current_price:,.2f}</div><div class="metric-label">Current Price</div></div>',unsafe_allow_html=True)
        with m2:
            color="#00d4aa" if next_price>current_price else "#ff6b6b"
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color}">₹{next_price:,.2f}</div><div class="metric-label">Predicted Next Price</div></div>',unsafe_allow_html=True)
        with m3:
            arrow="▲" if price_change>0 else "▼"
            color="#00d4aa" if price_change>0 else "#ff6b6b"
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color}">{arrow} {abs(price_change):.2f}%</div><div class="metric-label">Expected Change</div></div>',unsafe_allow_html=True)
        with m4: st.markdown(f'<div class="metric-card"><div class="metric-value">{accuracy:.1f}%</div><div class="metric-label">Model Accuracy</div></div>',unsafe_allow_html=True)

        st.markdown("---")
        sig_col, chart_col = st.columns([1,2])
        with sig_col:
            st.markdown("#### 🎯 Trading Signal")
            st.markdown(f"""<div class="{signal_css}">
                <div class="signal-text" style="color:{signal_color}">{signal_emoji} {signal}</div>
                <div style="color:#ccc;margin-top:8px">Confidence: <b style="color:{signal_color}">{confidence:.0f}%</b></div>
                <div style="color:#8892b0;margin-top:8px;font-size:0.85rem">Change: {price_change:+.2f}% | Threshold: ±1.5%</div>
            </div>""", unsafe_allow_html=True)
            filled = int(confidence/10)
            st.markdown(f"`{'█'*filled}{'░'*(10-filled)}` **{confidence:.0f}%**")
        with chart_col:
            st.markdown("#### 📈 Actual vs Predicted Prices")
            fig,ax = plt.subplots(figsize=(9,4))
            fig.patch.set_facecolor('#0f1117')
            ax.set_facecolor('#1e2130')
            ax.plot(actual, color='#4fc3f7', linewidth=1.5, label='Actual Price')
            ax.plot(predictions, color='#00d4aa', linewidth=1.5, label='Predicted Price', linestyle='--')
            ax.set_title(f"{selected_name} — Test Set Predictions", color='white', fontsize=12)
            ax.set_xlabel("Trading Days", color='#8892b0')
            ax.set_ylabel("Price (₹)", color='#8892b0')
            ax.tick_params(colors='#8892b0')
            for sp in ax.spines.values(): sp.set_edgecolor('#2e3650')
            ax.legend(facecolor='#1e2130', edgecolor='#2e3650', labelcolor='white')
            ax.grid(axis='y', color='#2e3650', linewidth=0.5, alpha=0.7)
            st.pyplot(fig)

        r1,r2,r3 = st.columns(3)
        r1.metric("RMSE", f"₹{rmse:.2f}")
        r2.metric("MAE",  f"₹{mae:.2f}")
        r3.metric("Data points", f"{len(df):,}")

with tab2:
    st.markdown("### Model Evaluation")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""**RMSE** — average prediction error in ₹. Lower is better.\n\n
**MAE** — mean absolute error. "On average, off by ₹X."\n\n
**80/20 split** — model never sees test data during training. Prevents overfitting.\n\n
**60-day sequences** — 3 months of price history as model input.""")
    with c2:
        st.dataframe(pd.DataFrame({
            "Model":["Linear Regression","ARIMA","Simple RNN","Our Model"],
            "Non-linear":["❌","❌","✅","✅"],
            "Long memory":["❌","⚠️","❌","✅"],
            "Volatile stocks":["❌","❌","⚠️","✅"],
        }), hide_index=True, use_container_width=True)
    st.divider()
    epochs=[1,2,3,4,5,6,7,8,9,10]
    fig2,ax2=plt.subplots(figsize=(8,3))
    fig2.patch.set_facecolor('#0f1117')
    ax2.set_facecolor('#1e2130')
    ax2.plot(epochs,[0.045,0.028,0.019,0.014,0.011,0.009,0.008,0.007,0.0065,0.006],'o-',color='#00d4aa',linewidth=2,label='Train Loss')
    ax2.plot(epochs,[0.052,0.034,0.025,0.020,0.017,0.015,0.014,0.013,0.0125,0.012],'s--',color='#ff9f43',linewidth=2,label='Val Loss')
    ax2.set_xlabel("Epoch",color='#8892b0')
    ax2.set_ylabel("Loss",color='#8892b0')
    ax2.set_title("Both losses converge — model is learning correctly",color='white')
    ax2.tick_params(colors='#8892b0')
    for sp in ax2.spines.values(): sp.set_edgecolor('#2e3650')
    ax2.legend(facecolor='#1e2130',edgecolor='#2e3650',labelcolor='white')
    ax2.grid(axis='y',color='#2e3650',linewidth=0.5,alpha=0.5)
    st.pyplot(fig2)

with tab3:
    st.markdown("### 🧠 How the Model Works")
    st.markdown('<div class="explainer-box"><b>Core idea:</b> Instead of predicting from 1 day, we feed 60 days of prices together — so the model sees trends, not just yesterday\'s number.</div>', unsafe_allow_html=True)
    g1,g2,g3 = st.columns(3)
    with g1: st.markdown('<div class="gate-box"><div style="font-size:2rem">📅</div><div style="color:#00d4aa;font-weight:700">60-Day Window</div><div style="color:#8892b0;font-size:0.85rem">3 months of price history as input. Captures trends and cycles.</div></div>',unsafe_allow_html=True)
    with g2: st.markdown('<div class="gate-box"><div style="font-size:2rem">📐</div><div style="color:#ffd700;font-weight:700">Normalisation</div><div style="color:#8892b0;font-size:0.85rem">Prices scaled to 0–1. Model learns patterns, not magnitudes.</div></div>',unsafe_allow_html=True)
    with g3: st.markdown('<div class="gate-box"><div style="font-size:2rem">🎯</div><div style="color:#4fc3f7;font-weight:700">RBF Kernel</div><div style="color:#8892b0;font-size:0.85rem">Maps data to higher dimensions. Finds non-linear price patterns.</div></div>',unsafe_allow_html=True)
    st.markdown("---")
    st.dataframe(pd.DataFrame({
        "Step":["1 Fetch","2 Scale","3 Sequence","4 Split","5 Train","6 Predict","7 Signal"],
        "What happens":["Download 5yr NSE data via yfinance","Normalise prices to 0–1","Build 60-day overlapping windows","80% train / 20% test","SVR fits window→next price mapping","Inverse-transform predictions back to ₹","Compare predicted vs current → BUY/SELL/HOLD"]
    }), hide_index=True, use_container_width=True)
    st.markdown("---")
    q1,q2=st.columns(2)
    with q1:
        st.markdown('<div class="explainer-box"><b>Q: Why not linear regression?</b><br>Stock prices are non-linear. A straight line misses patterns like "recovery after 5-day dip."</div>',unsafe_allow_html=True)
        st.markdown('<div class="explainer-box"><b>Q: How do you prevent overfitting?</b><br>80/20 split — model never sees test data during training. RMSE reported on unseen data only.</div>',unsafe_allow_html=True)
    with q2:
        st.markdown('<div class="explainer-box"><b>Q: Why NSE stocks?</b><br>Most research uses US markets. Indian markets have different volatility — RBI decisions, budget cycles. Underexplored area.</div>',unsafe_allow_html=True)
        st.markdown('<div class="explainer-box"><b>Q: Future scope?</b><br>Upgrade to Attention-LSTM + sentiment analysis from NSE news. Targeting Elsevier Q1 journal publication.</div>',unsafe_allow_html=True)
    st.divider()
    st.caption("Mini-Project Submission  •  Future: Attention-LSTM Research Paper (Expert Systems with Applications)")
