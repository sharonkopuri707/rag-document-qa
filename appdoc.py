import streamlit as st
import pypdf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

st.title("📄 RAG Document Q&A")
st.write("Upload a PDF and ask it anything")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    pdf_reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    words = text.split()
    chunk_size = 200
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    st.success(f"PDF loaded! Created {len(chunks)} chunks.")

    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    question = st.text_input("Ask a question about your document")

    if question:
        question_embedding = model.encode([question])
        distances, indices = index.search(np.array(question_embedding), k=3)
        relevant_chunks = [chunks[i] for i in indices[0]]
        context = "\n\n".join(relevant_chunks)

        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": f"Answer the question based only on the context below:\n\nContext:\n{context}\n\nQuestion: {question}"
                }
            ]
        )
        st.write("**Answer:**", response.choices[0].message.content)
