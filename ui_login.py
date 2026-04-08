import streamlit as st

def show_login_ui():
    # 🌌 Ultra-Modern Background & Centering Logic
    st.markdown("""
        <style>
        /* Force the app to cover the full viewport and center everything */
        .stApp {
            background: radial-gradient(circle at top right, #2c5364, #0f2027);
            display: flex;
            justify-content: center;
            align-items: center;
        }

        /* Container for the login content */
        .login-card {
            text-align: center;
            padding: 40px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            width: 100%;
            max-width: 400px;
        }

        .brand-title {
            font-size: 2.5rem;
            font-weight: 800;
            color: #00f5c3;
            margin-bottom: 0px;
        }

        .brand-subtitle {
            color: #94a3b8;
            font-size: 0.9rem;
            margin-bottom: 30px;
        }

        /* The Bold Centered LOGIN Text */
        .login-header {
            font-size: 2rem;
            font-weight: 900;
            color: #ffffff;
            letter-spacing: 5px;
            text-transform: uppercase;
            margin-bottom: 20px;
            text-shadow: 0 0 15px rgba(255, 255, 255, 0.3);
        }

        /* Style for the Google Button */
        div.stButton > button {
            width: 100% !important;
            background-color: white !important;
            color: #1f2937 !important;
            font-weight: 700 !important;
            border-radius: 50px !important;
            border: none !important;
            padding: 10px 0px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # 🧱 The Actual UI Structure
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    
    st.markdown('<div class="brand-title">⚡ QuantVision</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">AI-powered stock prediction platform</div>', unsafe_allow_html=True)
    
    # This is the bold, centered "LOGIN" you asked for
    st.markdown('<div class="login-header">LOGIN</div>', unsafe_allow_html=True)

    if st.button("🚀 Continue with Google"):
        st.session_state.logged_in = True
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# Navigation logic
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    show_login_ui()
else:
    st.title("Welcome to the Terminal")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
