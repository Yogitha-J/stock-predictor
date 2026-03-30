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
.news-card{background:#1e2130;border:1px solid #2e3650;border-radius:10px;padding:14px 16px;margin:8px 0}
</style>
""", unsafe_allow_html=True)

st.markdown("## 📈 AI Stock Price Predictor")
st.markdown("**Sequence Prediction Model — NSE / BSE / US Stocks**")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Live Prediction",
    "📊 Market Indicators",
    "📰 News & Sentiment",
    "🧠 How It Works"
])

def make_sequences(scaled_data, seq_len=60):
    X, Y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i].flatten())
        Y.append(scaled_data[i, 0])
    return np.array(X), np.array(Y)

def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fear_greed_score(df):
    close      = df['Close'].squeeze()
    rsi        = compute_rsi(close).iloc[-1]
    ma20       = close.rolling(20).mean().iloc[-1]
    ma50       = close.rolling(50).mean().iloc[-1]
    vol        = close.pct_change().rolling(20).std().iloc[-1] * 100
    trend_s    = 70 if ma20 > ma50 else 30
    vol_s      = max(0, 100 - vol * 10)
    score      = int(float(rsi) * 0.4 + trend_s * 0.4 + vol_s * 0.2)
    score      = max(0, min(100, score))
    if score >= 75:   label, color = "Extreme Greed", "#ff6b6b"
    elif score >= 55: label, color = "Greed",         "#ff9f43"
    elif score >= 45: label, color = "Neutral",       "#ffd700"
    elif score >= 25: label, color = "Fear",          "#4fc3f7"
    else:             label, color = "Extreme Fear",  "#00d4aa"
    return score, label, color

def get_news(ticker):
    try:
        t    = yf.Ticker(ticker)
        news = t.news
        if news: return news[:6]
    except Exception:
        pass
    return []

def sentiment_label(title):
    pos_words = ['rise','gain','profit','growth','surge','up','bull','record','high','beat','strong','boost']
    neg_words = ['fall','loss','drop','down','crash','bear','weak','cut','miss','low','decline','risk','warn']
    t   = title.lower()
    pos = sum(1 for w in pos_words if w in t)
    neg = sum(1 for w in neg_words if w in t)
    if pos > neg:   return "🟢 Positive", "#00d4aa"
    elif neg > pos: return "🔴 Negative", "#ff6b6b"
    else:           return "🟡 Neutral",  "#ffd700"

# ═══ TAB 1 — LIVE PREDICTION ═══
with tab1:
    st.markdown("### Pick a stock and get an instant prediction")
    col_input, col_info = st.columns([1, 2])
    with col_input:
        stock_options = {
            "TCS (Tata Consultancy)": "TCS.NS",
            "Infosys":                "INFY.NS",
            "Wipro":                  "WIPRO.NS",
            "HCL Technologies":       "HCLTECH.NS",
            "Reliance Industries":    "RELIANCE.NS",
            "HDFC Bank":              "HDFCBANK.NS",
            "Zomato":                 "ZOMATO.NS",
            "SBI":                    "SBIN.NS",
            "Bajaj Finance":          "BAJFINANCE.NS",
            "Adani Ports":            "ADANIPORTS.NS",
        }
        selected_name = st.selectbox("Choose a stock", list(stock_options.keys()))
        ticker        = stock_options[selected_name]
        st.caption(f"Ticker: `{ticker}`")
        custom = st.text_input("Or enter any ticker", placeholder="e.g. SWIGGY.NS / AAPL / TSLA")
        if custom.strip():
            ticker        = custom.strip().upper()
            selected_name = ticker
        st.caption("💡 NSE stocks: add `.NS` — US stocks: just ticker (AAPL, TSLA)")
        predict_btn = st.button("🚀 Run Prediction", use_container_width=True, type="primary")
    with col_info:
        st.markdown("""<div class="explainer-box"><b>What this does:</b><br>
        Downloads 5 years of real stock data, builds 60-day sequences,
        trains an RBF-Kernel SVR model, and predicts the next closing price.
        Also shows RSI, Fear & Greed index, live news, and lets you download price data.</div>""",
        unsafe_allow_html=True)

    if predict_btn:
        with st.spinner(f"Fetching & analyzing {selected_name}..."):
            df = yf.download(ticker, start="2018-01-01", progress=False)
            if df.empty:
                st.error(f"Could not fetch data for `{ticker}`.")
                st.warning("NSE stocks need `.NS` suffix. US stocks: just AAPL, TSLA etc.")
                st.stop()
            if len(df) < 100:
                st.warning(f"Only {len(df)} days of data — need 200+ for best accuracy.")

            close_prices  = df[['Close']].values
            current_price = float(close_prices[-1][0])
            scaler        = MinMaxScaler(feature_range=(0, 1))
            scaled        = scaler.fit_transform(close_prices)
            SEQ_LEN       = min(60, len(df) // 3)
            X, Y          = make_sequences(scaled, SEQ_LEN)
            split         = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            Y_train, Y_test = Y[:split], Y[split:]
            model = SVR(kernel='rbf', C=1e3, gamma=0.1)
            model.fit(X_train, Y_train)
            pred_scaled  = model.predict(X_test).reshape(-1, 1)
            predictions  = scaler.inverse_transform(pred_scaled)
            actual       = scaler.inverse_transform(Y_test.reshape(-1, 1))
            rmse         = float(np.sqrt(mean_squared_error(actual, predictions)))
            mae          = float(mean_absolute_error(actual, predictions))
            accuracy     = max(0, 100 - (rmse / current_price * 100))
            last_seq     = scaled[-SEQ_LEN:].flatten().reshape(1, -1)
            next_price   = float(scaler.inverse_transform(model.predict(last_seq).reshape(-1, 1))[0][0])
            price_change = ((next_price - current_price) / current_price) * 100

            if price_change > 1.5:
                signal, signal_css = "BUY",  "signal-buy"
                signal_color, signal_emoji = "#00d4aa", "🟢"
                confidence = min(99, 60 + abs(price_change) * 5)
            elif price_change < -1.5:
                signal, signal_css = "SELL", "signal-sell"
                signal_color, signal_emoji = "#ff6b6b", "🔴"
                confidence = min(99, 60 + abs(price_change) * 5)
            else:
                signal, signal_css = "HOLD", "signal-hold"
                signal_color, signal_emoji = "#ffd700", "🟡"
                confidence = 55 + (1.5 - abs(price_change)) * 10

            st.session_state['df']            = df
            st.session_state['ticker']        = ticker
            st.session_state['selected_name'] = selected_name
            st.session_state['current_price'] = current_price

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-value">₹{current_price:,.2f}</div><div class="metric-label">Current Price</div></div>', unsafe_allow_html=True)
        color = "#00d4aa" if next_price > current_price else "#ff6b6b"
        m2.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color}">₹{next_price:,.2f}</div><div class="metric-label">Predicted Price</div></div>', unsafe_allow_html=True)
        arrow = "▲" if price_change > 0 else "▼"
        m3.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color}">{arrow} {abs(price_change):.2f}%</div><div class="metric-label">Exp. Change</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-value">{accuracy:.1f}%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        sig_col, chart_col = st.columns([1, 2])
        with sig_col:
            st.markdown("#### 🎯 Trading Signal")
            st.markdown(f"""<div class="{signal_css}">
                <div class="signal-text" style="color:{signal_color}">{signal_emoji} {signal}</div>
                <div style="color:#ccc;margin-top:8px">Confidence: <b style="color:{signal_color}">{confidence:.0f}%</b></div>
                <div style="color:#8892b0;margin-top:8px;font-size:0.85rem">Change: {price_change:+.2f}% | Threshold: ±1.5%</div>
            </div>""", unsafe_allow_html=True)
            filled = int(confidence / 10)
            st.markdown(f"`{'█'*filled}{'░'*(10-filled)}` **{confidence:.0f}%**")
            st.markdown("---")
            r1, r2 = st.columns(2)
            r1.metric("RMSE", f"₹{rmse:.2f}")
            r2.metric("MAE",  f"₹{mae:.2f}")
        with chart_col:
            st.markdown("#### 📈 Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(9, 4))
            fig.patch.set_facecolor('#0f1117')
            ax.set_facecolor('#1e2130')
            ax.plot(actual[-150:],      color='#4fc3f7', linewidth=1.8, label='Actual Price')
            ax.plot(predictions[-150:], color='#00d4aa', linewidth=1.8, label='Predicted Price', linestyle='--')
            ax.set_title(f"{selected_name} — Last 150 test days", color='white', fontsize=11)
            ax.set_xlabel("Trading Days", color='#8892b0')
            ax.set_ylabel("Price",        color='#8892b0')
            ax.tick_params(colors='#8892b0')
            for sp in ax.spines.values(): sp.set_edgecolor('#2e3650')
            ax.legend(facecolor='#1e2130', edgecolor='#2e3650', labelcolor='white')
            ax.grid(axis='y', color='#2e3650', linewidth=0.5, alpha=0.7)
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("#### 📥 Download Price History")
        csv = df[['Open','High','Low','Close','Volume']].to_csv().encode('utf-8')
        st.download_button(
            label=f"⬇️ Download {selected_name} price history (CSV)",
            data=csv,
            file_name=f"{ticker}_price_history.csv",
            mime='text/csv',
            use_container_width=True
        )
        st.caption(f"Contains {len(df):,} trading days of OHLCV data from Yahoo Finance")

# ═══ TAB 2 — MARKET INDICATORS ═══
with tab2:
    st.markdown("### 📊 Market Indicators")
    if 'df' not in st.session_state:
        st.info("Run a prediction in Tab 1 first to load stock data.")
    else:
        df_i         = st.session_state['df']
        name_i       = st.session_state['selected_name']
        current_i    = st.session_state['current_price']
        close_series = df_i['Close'].squeeze()

        st.markdown("#### 📉 RSI — Relative Strength Index")
        rsi_series = compute_rsi(close_series)
        rsi_val    = float(rsi_series.iloc[-1])
        if rsi_val >= 70:   rsi_status, rsi_color = "🔴 Overbought — possible pullback ahead",    "#ff6b6b"
        elif rsi_val <= 30: rsi_status, rsi_color = "🟢 Oversold — possible bounce ahead",        "#00d4aa"
        else:               rsi_status, rsi_color = "🟡 Neutral — no strong signal right now",    "#ffd700"

        rc1, rc2 = st.columns([1, 3])
        with rc1:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{rsi_color}">{rsi_val:.1f}</div><div class="metric-label">RSI (14-day)</div></div>', unsafe_allow_html=True)
            st.markdown(f"**{rsi_status}**")
            st.markdown('<div class="explainer-box" style="font-size:0.82rem;margin-top:12px"><b>RSI guide:</b><br>• Above 70 = Overbought 🔴<br>• 30–70 = Neutral 🟡<br>• Below 30 = Oversold 🟢</div>', unsafe_allow_html=True)
        with rc2:
            fig_rsi, ax_rsi = plt.subplots(figsize=(9, 3))
            fig_rsi.patch.set_facecolor('#0f1117')
            ax_rsi.set_facecolor('#1e2130')
            rsi_plot = rsi_series.dropna().iloc[-200:]
            ax_rsi.plot(rsi_plot.values, color='#ff9f43', linewidth=1.5, label='RSI')
            ax_rsi.axhline(70, color='#ff6b6b', linewidth=1, linestyle='--', label='Overbought (70)')
            ax_rsi.axhline(30, color='#00d4aa', linewidth=1, linestyle='--', label='Oversold (30)')
            ax_rsi.fill_between(range(len(rsi_plot)), 70, 100, alpha=0.08, color='#ff6b6b')
            ax_rsi.fill_between(range(len(rsi_plot)),  0,  30, alpha=0.08, color='#00d4aa')
            ax_rsi.set_ylim(0, 100)
            ax_rsi.set_title(f"{name_i} — RSI (last 200 days)", color='white', fontsize=11)
            ax_rsi.set_xlabel("Trading Days", color='#8892b0')
            ax_rsi.set_ylabel("RSI",          color='#8892b0')
            ax_rsi.tick_params(colors='#8892b0')
            for sp in ax_rsi.spines.values(): sp.set_edgecolor('#2e3650')
            ax_rsi.legend(facecolor='#1e2130', edgecolor='#2e3650', labelcolor='white', fontsize=8)
            ax_rsi.grid(axis='y', color='#2e3650', linewidth=0.5, alpha=0.5)
            st.pyplot(fig_rsi)

        st.markdown("---")
        st.markdown("#### 😱 Fear & Greed Index")
        fg_score, fg_label, fg_color = fear_greed_score(df_i)
        fg1, fg2 = st.columns([1, 2])
        with fg1:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{fg_color};font-size:3rem">{fg_score}</div><div style="color:{fg_color};font-weight:700;font-size:1.1rem;margin-top:6px">{fg_label}</div><div class="metric-label">out of 100</div></div>', unsafe_allow_html=True)
        with fg2:
            fig_fg, ax_fg = plt.subplots(figsize=(6, 1.2))
            fig_fg.patch.set_facecolor('#0f1117')
            ax_fg.set_facecolor('#0f1117')
            for i, c in enumerate(['#00d4aa','#4fc3f7','#ffd700','#ff9f43','#ff6b6b']):
                ax_fg.barh(0, 20, left=i*20, color=c, height=0.5)
            ax_fg.axvline(fg_score, color='white', linewidth=3)
            ax_fg.set_xlim(0, 100)
            ax_fg.set_yticks([])
            ax_fg.set_xticks([0,25,50,75,100])
            ax_fg.set_xticklabels(['Extreme\nFear','Fear','Neutral','Greed','Extreme\nGreed'], color='#8892b0', fontsize=8)
            ax_fg.tick_params(colors='#8892b0')
            for sp in ax_fg.spines.values(): sp.set_visible(False)
            st.pyplot(fig_fg)
            st.markdown(f'<div class="explainer-box" style="font-size:0.82rem"><b>How we calculate it:</b><br>40% RSI momentum + 40% price vs moving average trend + 20% volatility score.<br>Score of <b style="color:{fg_color}">{fg_score} = {fg_label}</b> for {name_i} right now.</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📈 Price with Moving Averages (MA20 & MA50)")
        ma20 = close_series.rolling(20).mean()
        ma50 = close_series.rolling(50).mean()
        fig_ma, ax_ma = plt.subplots(figsize=(10, 4))
        fig_ma.patch.set_facecolor('#0f1117')
        ax_ma.set_facecolor('#1e2130')
        ax_ma.plot(close_series.iloc[-200:].values, color='#4fc3f7', linewidth=1.5, label='Close Price')
        ax_ma.plot(ma20.iloc[-200:].values,         color='#ff9f43', linewidth=1.2, label='MA 20', linestyle='--')
        ax_ma.plot(ma50.iloc[-200:].values,         color='#ff6b6b', linewidth=1.2, label='MA 50', linestyle=':')
        ax_ma.set_title(f"{name_i} — Price & Moving Averages (last 200 days)", color='white', fontsize=11)
        ax_ma.set_xlabel("Trading Days", color='#8892b0')
        ax_ma.set_ylabel("Price",        color='#8892b0')
        ax_ma.tick_params(colors='#8892b0')
        for sp in ax_ma.spines.values(): sp.set_edgecolor('#2e3650')
        ax_ma.legend(facecolor='#1e2130', edgecolor='#2e3650', labelcolor='white')
        ax_ma.grid(axis='y', color='#2e3650', linewidth=0.5, alpha=0.5)
        st.pyplot(fig_ma)
        st.caption("MA20 crossing above MA50 = bullish signal. MA20 crossing below MA50 = bearish signal.")

# ═══ TAB 3 — NEWS & SENTIMENT ═══
with tab3:
    st.markdown("### 📰 Latest News & Sentiment Analysis")
    if 'ticker' not in st.session_state:
        st.info("Run a prediction in Tab 1 first to load news.")
    else:
        ticker_n = st.session_state['ticker']
        name_n   = st.session_state['selected_name']
        st.markdown(f"**Showing latest news for: {name_n}** (`{ticker_n}`)")
        st.caption("Sentiment scored using keyword analysis on headline text.")
        with st.spinner("Fetching news..."):
            news_items = get_news(ticker_n)
        if not news_items:
            st.warning("No news found. Try a popular ticker like TCS.NS or INFY.NS.")
        else:
            pos_count = neg_count = neu_count = 0
            for item in news_items:
                try:
                    title     = item.get('content', {}).get('title', '') or item.get('title', '')
                    link      = item.get('content', {}).get('canonicalUrl', {}).get('url', '') or item.get('link', '#')
                    publisher = item.get('content', {}).get('provider', {}).get('displayName', '') or item.get('publisher', 'News')
                    if not title: continue
                    sent_label, sent_color = sentiment_label(title)
                    if "Positive" in sent_label: pos_count += 1
                    elif "Negative" in sent_label: neg_count += 1
                    else: neu_count += 1
                    st.markdown(f'<div class="news-card"><div style="font-size:0.75rem;color:#8892b0;margin-bottom:4px">{publisher}</div><div style="font-size:0.95rem"><a href="{link}" target="_blank" style="color:#4fc3f7;text-decoration:none">{title}</a></div><div style="margin-top:8px;font-size:0.82rem;color:{sent_color}"><b>{sent_label}</b></div></div>', unsafe_allow_html=True)
                except Exception:
                    continue
            st.markdown("---")
            st.markdown("#### Overall Sentiment Summary")
            total = pos_count + neg_count + neu_count
            if total > 0:
                s1, s2, s3 = st.columns(3)
                s1.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#00d4aa">{pos_count}</div><div class="metric-label">🟢 Positive</div></div>', unsafe_allow_html=True)
                s2.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ffd700">{neu_count}</div><div class="metric-label">🟡 Neutral</div></div>',  unsafe_allow_html=True)
                s3.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ff6b6b">{neg_count}</div><div class="metric-label">🔴 Negative</div></div>', unsafe_allow_html=True)
                overall = "Bullish 🟢" if pos_count > neg_count else ("Bearish 🔴" if neg_count > pos_count else "Neutral 🟡")
                st.markdown(f'<div class="explainer-box"><b>Overall news sentiment: {overall}</b><br>Based on {total} recent headlines. This is why our research paper will integrate news sentiment as a prediction feature — connecting Tab 3 directly to our future Attention-LSTM model.</div>', unsafe_allow_html=True)

# ═══ TAB 4 — HOW IT WORKS ═══
with tab4:
    st.markdown("### 🧠 How the Model Works")
    st.markdown('<div class="explainer-box"><b>Core idea:</b> Instead of predicting from 1 day, we feed 60 days of prices together — so the model sees trends, not just yesterday\'s number.</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    with g1: st.markdown('<div class="explainer-box"><b>📅 60-Day Window</b><br>3 months of price history as input. Captures trends and cycles that single-day models miss.</div>', unsafe_allow_html=True)
    with g2: st.markdown('<div class="explainer-box"><b>📐 Normalisation</b><br>Prices scaled to 0–1. Model learns patterns, not magnitudes.</div>', unsafe_allow_html=True)
    with g3: st.markdown('<div class="explainer-box"><b>🎯 RBF Kernel (SVR)</b><br>Maps data to higher dimensions. Finds non-linear price patterns straight lines cannot.</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.dataframe(pd.DataFrame({
        "Step":["1 Fetch","2 Scale","3 Sequence","4 Split","5 Train","6 Predict","7 Signal","8 Indicators"],
        "What happens":["Download historical data via yfinance","Normalise prices to 0–1 with MinMaxScaler","Build overlapping 60-day windows","80% training / 20% testing — model never sees test data","SVR (RBF kernel) learns window → next price","Inverse-transform predictions back to ₹","Compare predicted vs current → BUY / SELL / HOLD","RSI + Fear & Greed + MA for deeper market context"]
    }), hide_index=True, use_container_width=True)
    st.markdown("---")
    q1, q2 = st.columns(2)
    with q1:
        st.markdown('<div class="explainer-box"><b>Q: Why not linear regression?</b><br>Stock prices are non-linear. A straight line misses patterns like recovery after a 5-day dip.</div>', unsafe_allow_html=True)
        st.markdown('<div class="explainer-box"><b>Q: What is RSI?</b><br>Measures speed of price changes. Above 70 = overbought, below 30 = oversold.</div>', unsafe_allow_html=True)
        st.markdown('<div class="explainer-box"><b>Q: How do you prevent overfitting?</b><br>80/20 split. Model never sees test data. RMSE reported only on unseen data.</div>', unsafe_allow_html=True)
    with q2:
        st.markdown('<div class="explainer-box"><b>Q: What is Fear & Greed Index?</b><br>Custom sentiment indicator — RSI momentum + moving average trend + volatility score combined into 0–100.</div>', unsafe_allow_html=True)
        st.markdown('<div class="explainer-box"><b>Q: Why NSE stocks?</b><br>Most research uses US markets. Indian markets behave differently — RBI decisions, budget cycles. Underexplored area.</div>', unsafe_allow_html=True)
        st.markdown('<div class="explainer-box"><b>Q: Future scope?</b><br>Upgrade to Attention-LSTM + integrate news sentiment as model input feature. Targeting Elsevier Q1 journal.</div>', unsafe_allow_html=True)
    st.divider()
    st.caption("Mini-Project Submission  •  Future: Attention-LSTM + Sentiment Fusion (Research Paper)")
