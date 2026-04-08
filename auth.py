import streamlit as st
import requests
from streamlit_oauth import OAuth2Component

CLIENT_ID = st.secrets["CLIENT_ID"]
CLIENT_SECRET = st.secrets["CLIENT_SECRET"]

AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"

REDIRECT_URI = "https://stock-predictor-nyo7pgbptgema3qnwhw86n.streamlit.app"

oauth2 = OAuth2Component(
    CLIENT_ID,
    CLIENT_SECRET,
    AUTHORIZE_URL,
    TOKEN_URL,
    REVOKE_URL
)

def login():
    if "user" not in st.session_state:
        result = oauth2.authorize_button(
            name="🔐 Continue with Google",
            redirect_uri=REDIRECT_URI,
            scope="openid email profile",
            key="google_login",
        )

        if result:
            token = result["token"]
            user_info = requests.get(
                "https://www.googleapis.com/oauth2/v1/userinfo",
                params={"access_token": token["access_token"]}
            ).json()

            st.session_state["user"] = user_info
            st.rerun()

        return False
    return True
