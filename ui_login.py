import streamlit as st

def show_login_ui():
    st.markdown("""
    <style>
    /* Import Inter font for that tech look */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    .main {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
    }

    /* Animation Keyframes */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes glowPulse {
        0% { text-shadow: 0 0 5px rgba(0, 245, 195, 0.2); }
        50% { text-shadow: 0 0 20px rgba(0, 245, 195, 0.6); }
        100% { text-shadow: 0 0 5px rgba(0, 245, 195, 0.2); }
    }

    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 70vh;
        font-family: 'Inter', sans-serif;
    }

    .login-box {
        background: rgba(255, 255, 255, 0.03);
        padding: 50px;
        border-radius: 24px;
        text-align: center;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        animation: fadeInUp 0.8s ease-out; /* Motion UI trigger */
        max-width: 400px;
    }

    .brand-logo {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #00f5c3, #4fc3f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        animation: glowPulse 3s infinite;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 2rem;
        letter-spacing: 0.5px;
    }

    /* Styling the Streamlit Button to match motion theme */
    div.stButton > button {
        background: linear-gradient(90deg, #00f5c3 0%, #21d4fd 100%);
        color: #0f172a;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        transition: all 0.3s ease;
        width: 100%;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -10px #00f5c3;
        color: #0f172a;
    }
    </style>
    """, unsafe_allow_html=True)

    # Wrap the UI in a container
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        
        st.markdown('<div class="brand-logo">⚡ QuantVision</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Next-Gen Predictive Intelligence</div>', unsafe_allow_html=True)
        
        # Login Inputs
        email = st.text_input("Work Email", placeholder="name@company.com")
        password = st.text_input("Password", type="password")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Enter Terminal"):
            st.success("Access Granted")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
