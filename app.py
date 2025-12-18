import streamlit as st
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="🏢",
    layout="centered"
)

st.title("🏢 HR Policy Assistant")
st.caption("Answers are based only on official HR policy documents")


# ------------------ LOAD RAG PIPELINE ------------------
@st.cache_resource
def load_rag_pipeline():

    # Load HR Policy PDF
    loader = PyPDFLoader("Sample_HR_Policies.pdf")
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    texts = [chunk.page_content for chunk in chunks]

    # Create embeddings
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    embeddings = embedder.encode(texts)

    # Store in FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Load LLM
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=200
    )

    return embedder, index, texts, llm


embedder, index, texts, llm = load_rag_pipeline()


# ------------------ ANSWER FUNCTION ------------------
def get_answer(question):

    query_embedding = embedder.encode([question])
    distances, indices = index.search(query_embedding, k=3)

    context = " ".join([texts[i] for i in indices[0]])

    if context.strip() == "":
        return (
            "I checked the HR policy document, but this information is not mentioned. "
            "Please reach out to the HR team for further clarification."
        )

    prompt = f"""
You are an HR assistant.
Answer the question using ONLY the context below.
If the answer is not present, say politely that it is not mentioned.

Context:
{context}

Question:
{question}
"""

    response = llm(prompt)[0]["generated_text"]
    return response


# ------------------ CHAT UI ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask your HR policy question...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("assistant"):
        with st.spinner("Checking HR policies..."):
            answer = get_answer(question)
            st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
