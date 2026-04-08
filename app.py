import streamlit as st
from streamlit_oauth import OAuth2Component

# =============================
# 🎨 LOAD CSS
# =============================
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =============================
# 🔐 GOOGLE LOGIN
# =============================
import streamlit as st

CLIENT_ID = st.secrets["CLIENT_ID"]
CLIENT_SECRET = st.secrets["CLIENT_SECRET"]

oauth2 = OAuth2Component(
    CLIENT_ID,
    CLIENT_SECRET,
    "https://accounts.google.com/o/oauth2/auth",
    "https://oauth2.googleapis.com/token"
)

st.markdown("<h1 style='text-align:center;'>📈 AI Stock Predictor</h1>", unsafe_allow_html=True)

result = oauth2.authorize_button(
    "Login with Google",
    redirect_uri="http://localhost:8501",
    scope="openid email profile"
)

if not result:
    st.stop()

st.success("Demo Mode - Login Disabled")
# =============================
# 📊 SIDEBAR MENU
# =============================
st.sidebar.title("📊 Dashboard")
menu = st.sidebar.radio("Navigate", ["Home", "Prediction", "About"])

# =============================
# 🏠 HOME PAGE
# =============================
if menu == "Home":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🚀 Welcome")
    st.write("This AI-powered system predicts stock prices using LSTM Deep Learning.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 💡 Features")
    st.write("- Real-time stock data")
    st.write("- LSTM prediction model")
    st.write("- Interactive dashboard")
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# 🔥 MAIN APP STARTS HERE
# ==============================

elif menu == "Prediction":

    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import load_model
    import yfinance as yf
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Advanced Stock Prediction")

    col1, col2 = st.columns(2)

    with col1:
        stock = st.text_input("Stock 1", "RELIANCE.NS")

    with col2:
        stock2 = st.text_input("Stock 2 (optional)", "")

    if st.button("Predict"):

        def process_stock(stock_name):
            data = yf.download(stock_name, start="2015-01-01", end="2024-01-01")

            if data.empty:
                return None, None, None

            data = data[['Close']]

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)

            x = []
            y = []

            for i in range(60, len(scaled_data)):
                x.append(scaled_data[i-60:i])
                y.append(scaled_data[i])

            x, y = np.array(x), np.array(y)

            model = load_model("model/lstm_model.h5")

            predictions = model.predict(x)
            predictions = scaler.inverse_transform(predictions)
            actual = scaler.inverse_transform(y)

            return data, predictions, actual

        # Process first stock
        data1, pred1, act1 = process_stock(stock)

        if data1 is None:
            st.error("Invalid Stock 1")
        else:
            fig, ax = plt.subplots()

            ax.plot(act1, label=f"{stock} Actual")
            ax.plot(pred1, label=f"{stock} Predicted")

            # Accuracy metrics
            rmse = np.sqrt(mean_squared_error(act1, pred1))
            mae = mean_absolute_error(act1, pred1)

            st.write(f"📊 RMSE: {rmse:.2f}")
            st.write(f"📊 MAE: {mae:.2f}")

            # Second stock comparison
            if stock2:
                data2, pred2, act2 = process_stock(stock2)

                if data2 is not None:
                    ax.plot(act2, label=f"{stock2} Actual")

            ax.legend()
            st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)