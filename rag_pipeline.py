import faiss
import numpy as np
import re

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import PDF_PATH


def load_rag():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]

    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    return embedder, index, texts, llm


def ensure_bullets(text):
    sentences = re.split(r"\.\s+", text)
    bullets = [f"- {s.strip()}" for s in sentences if len(s.strip()) > 5]
    return "\n".join(bullets[:5])


def answer_query(question, embedder, index, texts, llm):
    q_emb = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, idx = index.search(q_emb, k=5)

    if scores[0][0] < 0.30:
        return None

    context = "\n".join(
        [texts[i] for i, s in zip(idx[0], scores[0]) if s > 0.30][:3]
    )

    prompt = f"""
You are a professional HR assistant.

Rules:
- Answer ONLY from policy
- Do NOT invent information
- 3 to 5 bullet points only

Policy:
{context}

Question:
{question}

Answer:
"""

    result = llm(
        prompt,
        max_length=180,
        temperature=0.0,
        do_sample=False
    )[0]["generated_text"]

    return ensure_bullets(result)
