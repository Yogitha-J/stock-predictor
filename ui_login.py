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

        return True  # just UI, no button here
