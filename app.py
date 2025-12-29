import streamlit as st
import faiss
import numpy as np
import os
import re
from PIL import Image

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="HR Policy Chatbot", page_icon="🏢")

# ---------------- CONSTANTS ----------------
PDF_FILE = "Sample_HR_Policy_Document.pdf"
LOGO_FILE = "nexus_iq_logo.png"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, PDF_FILE)
LOGO_PATH = os.path.join(BASE_DIR, LOGO_FILE)

# ---------------- LOGIN SYSTEM ----------------
USERS = {
    "employee": {"password": "employee123", "role": "Employee"},
    "hr": {"password": "hr123", "role": "HR"}
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login – HR Policy Assistant")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        if username.strip() in USERS and USERS[username.strip()]["password"] == password.strip():
            st.session_state.logged_in = True
            st.session_state.role = USERS[username.strip()]["role"]
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- HEADER ----------------
if os.path.exists(LOGO_PATH):
    logo = Image.open(LOGO_PATH)
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image(logo, width=65)
    with col2:
        st.markdown("## **Enterprise RAG-based HR Chatbot**")
else:
    st.markdown("## **Enterprise RAG-based HR Chatbot**")

st.caption(f"Logged in as: **{st.session_state.role}**")

if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ---------------- CHECK PDF ----------------
if not os.path.exists(PDF_PATH):
    st.error("❌ HR Policy PDF not found in repository root.")
    st.stop()

# ---------------- LOAD RAG PIPELINE ----------------
@st.cache_resource
def load_pipeline():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    texts = [c.page_content for c in chunks]

    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    embeddings = embedder.encode(texts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    return embedder, index, texts, llm

embedder, index, texts, llm = load_pipeline()

# ---------------- HELPERS ----------------
def is_greeting(text):
    return text.lower().strip() in [
        "hi", "hello", "hey",
        "good morning", "good afternoon", "good evening"
    ]

def ensure_bullets(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines and lines[0].startswith("-"):
        return "\n".join(lines[:5])

    sentences = re.split(r"\.\s+", text)
    bullets = [f"- {s.strip()}" for s in sentences if len(s.strip()) > 5]
    return "\n".join(bullets[:5])

# ---------------- RAG QUERY ----------------
def answer_query(question):
    q_emb = embedder.encode([question])
    distances, idx = index.search(np.array(q_emb), k=5)

    context = " ".join([texts[i] for i in idx[0][:3]])

    # Keyword fallback for section titles (e.g., Grievance)
    if distances[0][0] > 1.6 and not any(
        kw in context.lower() for kw in question.lower().split()
    ):
        return "- This information is not mentioned in the HR policy document."

    prompt = f"""
You are an HR assistant.

Answer ONLY in clear bullet points.
Do NOT copy policy clauses word by word.
Use professional, simple language.
Limit to 3–5 short bullet points.

Policy Content:
{context}

Question:
{question}

Format:
- Point 1
- Point 2
- Point 3
"""

    raw = llm(prompt, max_length=160, temperature=0.1)[0]["generated_text"]
    clean = ensure_bullets(raw)

    if clean.lower().strip() in ["- point 1", "point 1", ""]:
        return "- This information is not mentioned in the HR policy document."

    return clean

# ---------------- CHAT UI ----------------
st.subheader("💬 Ask HR Policy Question")
question = st.text_input("Enter your question")

if question:
    if is_greeting(question):
        st.info(
            "Hello 👋 I’m your HR Policy Assistant.\n\n"
            "You can ask:\n"
            "- What is the leave policy?\n"
            "- What is the WFH policy?\n"
            "- What is the Grievance Redressal Policy?"
        )
    else:
        st.success(answer_query(question))
