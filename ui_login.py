import streamlit as st

# 1. Initialize session state at the VERY top
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def show_login_ui():
    st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%);
    }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    .centered-wrapper {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        z-index: 999;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(25px);
        padding: 60px;
        border-radius: 32px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        animation: float 6s ease-in-out infinite;
        max-width: 450px;
    }

    .brand-logo {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f5c3 0%, #4fc3f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    /* Target the button specifically */
    div.stButton > button {
        background-color: white !important;
        color: #1f2937 !important;
        border-radius: 50px !important;
        padding: 12px 40px !important;
        font-weight: 700 !important;
        width: 100%;
        border: none !important;
        transition: 0.3s;
    }
    
    div.stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 10px 20px rgba(0,245,197,0.2) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="centered-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="brand-logo">⚡ QuantVision</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b; letter-spacing:2px; margin-bottom:30px;">INSTITUTIONAL TERMINAL</p>', unsafe_allow_html=True)

    # The Logic Fix
    if st.button("🚀 Continue with Google"):
        st.session_state.logged_in = True
        st.rerun()  # Forces streamlit to refresh and see that logged_in is now True

    st.markdown('</div></div>', unsafe_allow_html=True)

# ─── NAVIGATION LOGIC ───
if not st.session_state.logged_in:
    show_login_ui()
else:
    # This is where your actual app code goes
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("📈 Market Dashboard")
    st.write("Welcome back, Chief. The markets are active.")
    # Add your charts and tabs here...
