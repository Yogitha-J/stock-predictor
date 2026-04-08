import streamlit as st

def show_login_ui():
    # 🌌 Background & Specific Text Styling
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        }
        
        /* Centering the entire content block */
        .main-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 80vh;
            text-align: center;
            width: 100%;
        }

        /* Bold Center LOGIN */
        .login-text {
            font-size: 3rem;
            font-weight: 900;
            color: #ffffff;
            margin-top: 20px;
            margin-bottom: 20px;
            text-transform: uppercase;
        }

        .brand-header {
            color: #ffffff;
            margin-bottom: 0px;
        }
        
        .brand-subtitle {
            color: #a0aec0;
            margin-bottom: 10px;
        }

        /* Standard Google Button Style */
        div.stButton > button {
            background-color: white !important;
            color: black !important;
            border-radius: 5px !important;
            padding: 10px 25px !important;
            font-weight: 600 !important;
            border: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # 🧱 Layout
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown('<h2 class="brand-header">⚡ QuantVision</h2>', unsafe_allow_html=True)
    st.markdown('<p class="brand-subtitle">AI-powered stock prediction platform</p>', unsafe_allow_html=True)
    
    # The requested bold centered LOGIN
    st.markdown('<div class="login-text">LOGIN</div>', unsafe_allow_html=True)

    if st.button("Continue with Google"):
        st.session_state.logged_in = True
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# Navigation logic
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    show_login_ui()
else:
    st.write("Dashboard Loaded")
