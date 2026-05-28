# RAG Document Q&A

An advanced AI-powered document chatbot using Retrieval Augmented Generation (RAG) 
to answer questions from any PDF with high accuracy.

## Live Demo
[Click here to try it](https://sharonkopuri-rag.streamlit.app)

## How It Works
1. PDF is split into chunks
2. Each chunk is converted to vector embeddings using Sentence Transformers
3. Embeddings stored in FAISS vector database
4. User question is embedded and matched against stored vectors
5. Most relevant chunks retrieved and sent to Groq LLM for answer generation

## Tech Stack
- Python
- Streamlit
- Groq API (LLaMA 3.3)
- FAISS (Vector Database)
- Sentence Transformers (Embeddings)
- PyPDF

## Features
- True RAG pipeline with semantic search
- FAISS vector similarity search
- Context-aware answers from any PDF
- Fast inference via Groq API

## How to Run Locally
pip install -r requirements.txt
streamlit run appdoc.py
