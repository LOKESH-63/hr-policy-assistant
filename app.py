import streamlit as st
import pandas as pd
from PIL import Image

from config import USERS, LOGO_PATH, ANALYTICS_PATH
from rag_pipeline import load_rag, answer_query
from analytics import init_analytics, log_query


st.set_page_config(page_title="HR Policy Assistant", page_icon="🏢", layout="wide")

# -------- SESSION --------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# -------- LOGIN --------
def login():
    st.title("🔐 Login – HR Policy Assistant")
    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if u in USERS and USERS[u]["password"] == p:
                st.session_state.logged_in = True
                st.session_state.user = u
                st.session_state.role = USERS[u]["role"]
                st.rerun()
            else:
                st.error("Invalid credentials")


if not st.session_state.logged_in:
    login()
    st.stop()


# -------- HEADER --------
if LOGO_PATH:
    st.image(Image.open(LOGO_PATH), width=80)

st.caption(f"Logged in as **{st.session_state.role}**")

# -------- INIT --------
init_analytics()
embedder, index, texts, llm = load_rag()

# -------- UI --------
tabs = ["💬 Ask HR"]
if st.session_state.role == "HR":
    tabs.append("📊 Admin Analytics")

tab_list = st.tabs(tabs)

with tab_list[0]:
    q = st.text_input("Ask HR Policy Question")
    if q:
        ans = answer_query(q, embedder, index, texts, llm)
        if ans:
            st.success(ans)
            log_query(st.session_state.user, st.session_state.role, q, True)
        else:
            st.warning("Not found in HR policy")
            log_query(st.session_state.user, st.session_state.role, q, False)

if st.session_state.role == "HR":
    with tab_list[1]:
        df = pd.read_csv(ANALYTICS_PATH)
        st.dataframe(df)
