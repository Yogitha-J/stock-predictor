import streamlit as st

def show_login_ui():
    st.markdown("""
    <style>
    /* Full Page Override for perfect centering */
    .stApp {
        background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%);
    }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    @keyframes borderGlow {
        0% { border-color: rgba(0, 245, 197, 0.3); box-shadow: 0 0 10px rgba(0, 245, 197, 0.1); }
        50% { border-color: rgba(79, 195, 247, 0.6); box-shadow: 0 0 25px rgba(79, 195, 247, 0.3); }
        100% { border-color: rgba(0, 245, 197, 0.3); box-shadow: 0 0 10px rgba(0, 245, 197, 0.1); }
    }

    .centered-wrapper {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        z-index: 999;
        font-family: 'Inter', sans-serif;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(25px);
        padding: 60px;
        border-radius: 32px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        animation: float 6s ease-in-out infinite, borderGlow 4s infinite;
        max-width: 450px;
    }

    .brand-logo {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #00f5c3 0%, #4fc3f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    .tagline {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 40px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Modern Google Button Style */
    .google-btn-container {
        display: flex;
        justify-content: center;
        transition: transform 0.2s;
    }

    .google-btn-container:hover {
        transform: scale(1.05);
    }

    /* Targeting Streamlit Button specifically */
    div.stButton > button {
        background-color: white !important;
        color: #1f2937 !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 12px 30px !important;
        font-weight: 600 !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Centered Layout
    st.markdown('<div class="centered-wrapper">', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    st.markdown('<div class="brand-logo">⚡ QuantVision</div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">Institutional Intelligence</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="margin-top: 20px;">', unsafe_allow_html=True)
    # Using columns inside the container to force button width behavior if needed
    if st.button("🚀 Continue with Google"):
        st.session_state.logged_in = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p style="color: #475569; font-size: 0.75rem; margin-top: 30px;">Secure One-Tap Authentication</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True) # End Glass Card
    st.markdown('</div>', unsafe_allow_html=True) # End Wrapper

# To call it
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    show_login_ui()
else:
    st.title("Main Dashboard")
