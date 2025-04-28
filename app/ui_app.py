# app/ui_app.py

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
index_path = "../embeddings/mini_faiss_index"
vector_store = FAISS.load_local(
    index_path,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# Streamlit UI
st.set_page_config(page_title="Mini RAG Search", page_icon="🔎")

st.title("🔎 Mini RAG - Search Your Documents")

query = st.text_input("Enter your query here:")

if query:
    # FAISS Similarity Search
    retrieved_docs = vector_store.similarity_search(query, k=2)
    retrieved_texts = "\n".join([doc.page_content for doc in retrieved_docs])

    st.subheader("📚 Retrieved Context:")
    st.write(retrieved_texts)

    # Gemini Response
    response = genai.GenerativeModel('gemini-1.5-flash').generate_content(
        f"Answer the following question based on the given context:\n\nContext:\n{retrieved_texts}\n\nQuestion: {query}"
    )

    st.subheader("🤖 Gemini's Answer:")
    st.write(response.text)
