import streamlit as st

def show_login_ui():
    # 🌌 Background
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        }
        </style>
    """, unsafe_allow_html=True)

    # 🧱 Center layout using columns
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("## ⚡ QuantVision")
        st.markdown("##### AI-powered stock prediction platform")
        st.write("")

        return True  # just UI, no button here
