# 🏢 HR Policy Assistant (RAG Based)

This is a Retrieval-Augmented Generation (RAG) based HR Policy Assistant.
Employees can ask questions and get accurate answers strictly from official HR policy documents.

## 🔧 Tech Stack
- Streamlit
- FAISS
- Sentence Transformers
- HuggingFace Transformers
- LangChain

## 🚀 How It Works
1. HR policy PDF is loaded
2. Text is split into chunks
3. Embeddings are created
4. Stored in FAISS vector database
5. User question retrieves relevant chunks
6. LLM generates answer using retrieved context

## 🛡️ Safety
- No hallucinations
- No training on private data
- Polite fallback response if answer is not found

## ▶ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
