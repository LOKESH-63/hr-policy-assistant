import streamlit as st
import faiss
import numpy as np
import os
import re
import pandas as pd
from datetime import datetime
from PIL import Image

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="HR Policy Assistant", page_icon="🏢", layout="wide")


# ---------------- FILE PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "Sample_HR_Policy_Document.pdf")
LOGO_PATH = os.path.join(BASE_DIR, "nexus_iq_logo.png")
ANALYTICS_PATH = os.path.join(BASE_DIR, "analytics.csv")


# ---------------- USERS ----------------
USERS = {
    "employee": {"password": "employee123", "role": "Employee"},
    "hr": {"password": "hr123", "role": "HR"}
}


# ---------------- SESSION INIT ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ---------------- LOGIN ----------------
def login():
    st.title("🔐 Login – HR Policy Assistant")
    with st.form("login"):
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        if user in USERS and USERS[user]["password"] == pwd:
            st.session_state.logged_in = True
            st.session_state.role = USERS[user]["role"]
            st.session_state.user = user
            st.rerun()
        else:
            st.error("Invalid credentials")


if not st.session_state.logged_in:
    login()
    st.stop()


# ---------------- HEADER ----------------
col1, col2 = st.columns([1, 6])
if os.path.exists(LOGO_PATH):
    col1.image(Image.open(LOGO_PATH), width=70)
col2.markdown("## **Enterprise RAG-Based HR Policy Assistant**")

st.caption(f"Logged in as **{st.session_state.role}**")

if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()


# ---------------- LOAD ANALYTICS ----------------
if not os.path.exists(ANALYTICS_PATH):
    pd.DataFrame(
        columns=["timestamp", "user", "role", "question", "answered"]
    ).to_csv(ANALYTICS_PATH, index=False)


def log_query(question, answered):
    df = pd.read_csv(ANALYTICS_PATH)
    df.loc[len(df)] = [
        datetime.now(),
        st.session_state.user,
        st.session_state.role,
        question,
        answered
    ]
    df.to_csv(ANALYTICS_PATH, index=False)


# ---------------- LOAD RAG ----------------
@st.cache_resource
def load_rag():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]

    embedder = SentenceTransformer(
        "BAAI/bge-base-en-v1.5",
        normalize_embeddings=True
    )
    embeddings = embedder.encode(texts)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings))

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    return embedder, index, texts, llm


embedder, index, texts, llm = load_rag()


# ---------------- HELPERS ----------------
def ensure_bullets(text):
    sentences = re.split(r"\.\s+", text)
    bullets = [f"- {s.strip()}" for s in sentences if len(s.strip()) > 5]
    return "\n".join(bullets[:5])


def answer_query(question):
    q_emb = embedder.encode([question], normalize_embeddings=True)
    scores, idx = index.search(np.array(q_emb), k=5)

    if scores[0][0] < 0.30:
        log_query(question, False)
        return (
            "- I checked the HR policy document, but this information is not mentioned.\n"
            "- Please reach out to the HR team for clarification."
        )

    context = "\n".join(
        [texts[i] for i, s in zip(idx[0], scores[0]) if s > 0.30][:3]
    )

    prompt = f"""
You are a professional HR assistant.

Rules:
- Answer ONLY from the HR policy
- Do NOT invent information
- Use clear bullet points
- 3 to 5 bullets only

Policy:
{context}

Question:
{question}

Answer:
"""

    response = llm(
        prompt,
        max_length=180,
        temperature=0.0,
        do_sample=False
    )[0]["generated_text"]

    clean = ensure_bullets(response)
    log_query(question, True)
    return clean


# ---------------- UI TABS ----------------
tabs = ["💬 Ask HR"]
if st.session_state.role == "HR":
    tabs.append("📊 Admin Analytics")

tab = st.tabs(tabs)


# ---------------- CHAT TAB ----------------
with tab[0]:
    st.subheader("Ask a Question")
    q = st.text_input("Enter your HR policy question")

    if q:
        st.success(answer_query(q))


# ---------------- ADMIN ANALYTICS (HR ONLY) ----------------
if st.session_state.role == "HR":
    with tab[1]:
        st.subheader("📊 HR Admin Analytics")

        df = pd.read_csv(ANALYTICS_PATH)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Questions", len(df))
        col2.metric("Answered", df["answered"].sum())
        col3.metric("Unanswered", len(df) - df["answered"].sum())

        st.divider()

        st.markdown("### 🔥 Top Asked Questions")
        st.dataframe(
            df["question"].value_counts().head(5).reset_index(),
            use_container_width=True
        )

        st.divider()

        st.markdown("### ❌ Unanswered Queries")
        st.dataframe(
            df[df["answered"] == False][["timestamp", "question"]],
            use_container_width=True
        )

        st.divider()

        st.markdown("### 📥 Full Query Log")
        st.dataframe(df, use_container_width=True)
