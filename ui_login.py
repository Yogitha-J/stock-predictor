import streamlit as st

def show_login_ui():
    import streamlit as st

    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 80vh;
    }
    .login-box {
        background: rgba(255,255,255,0.05);
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-container"><div class="login-box">', unsafe_allow_html=True)

    st.markdown("## ⚡ QuantVision")
    st.markdown("AI-powered stock prediction platform")
