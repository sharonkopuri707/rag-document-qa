# 📄 RAG Document Q&A

An advanced AI-powered document chatbot using **Retrieval Augmented Generation (RAG)** to answer questions from any PDF with high accuracy.

## 🚀 Live Demo
[**Click here to try it →**](https://sharonkopuri-rag.streamlit.app)

## ⚙️ How It Works
1. PDF is split into semantic chunks
2. Each chunk is converted to vector embeddings using Sentence Transformers
3. Embeddings stored in a FAISS vector database
4. User question is embedded and matched via similarity search
5. Most relevant chunks retrieved and sent to Groq LLM (LLaMA 3.3) for answer generation

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Core language |
| Streamlit | Frontend UI |
| Groq API (LLaMA 3.3) | LLM inference |
| FAISS | Vector database |
| Sentence Transformers | Text embeddings |
| PyPDF | PDF parsing |

## ✨ Features
- True RAG pipeline with semantic search
- FAISS vector similarity search
- Context-aware answers from any PDF
- Fast inference via Groq API
- No API key needed — just upload and ask

## 💻 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run appdoc.py
```
