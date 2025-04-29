# app/ui_app.py

import os
import json
import streamlit as st
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google.generativeai import GenerativeModel
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="RAG-QA with Gemini", page_icon="🔎", layout="wide")

# Paths
INDEX_PATH = "../embeddings/full_faiss_index"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Load Gemini API Key (Single key only now)
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("No Gemini API key found! Please set GEMINI_API_KEY in .env")
    st.stop()

# Load FAISS index
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)
    return vectorstore

vectorstore = load_vectorstore()

# Initialize Gemini Model
genai.configure(api_key=API_KEY)
model = GenerativeModel("gemini-1.5-pro")

# Streamlit App
st.title("🔎 RAG Q&A System with Gemini")

query = st.text_input("Ask a question related to Computer Science:", "")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Searching..."):
            try:
                # Step 1: Search FAISS
                retrieved_docs = vectorstore.similarity_search(query, k=3)
                retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

                # Step 2: Prepare prompt
                prompt = f"""Answer the following question based on the given context.
                
Context:
{retrieved_context}

Question:
{query}

Answer:"""

                # Step 3: Generate Answer
                response = model.generate_content(prompt)
                answer = response.text

                # Step 4: Display Answer
                st.success("Answer generated successfully!")
                st.markdown(f"### 💬 Answer:\n\n{answer}")

                # Step 5: Save Log
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_data = {
                    "timestamp": timestamp,
                    "query": query,
                    "context": retrieved_context,
                    "answer": answer
                }
                with open(f"{LOG_DIR}/log_{timestamp}.json", "w", encoding="utf-8") as f:
                    json.dump(log_data, f, ensure_ascii=False, indent=4)

            except Exception as e:
                st.error(f"Error generating answer: {e}")
