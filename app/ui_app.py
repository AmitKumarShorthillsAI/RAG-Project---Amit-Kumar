import os
import json
import requests
import streamlit as st
import traceback
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# Constants
INDEX_PATH = "../embeddings/full_faiss_index"
LOG_DIR = "logs"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:8b"
MAX_CONTEXT_LEN = 1200

# Setup
os.makedirs(LOG_DIR, exist_ok=True)

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    return FAISS.load_local(INDEX_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

vectorstore = load_vectorstore()
reranker = load_reranker()

def rerank(query, docs, top_k=3):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked_docs[:top_k]]

def stream_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
        stream=True
    )
    response.raise_for_status()
    partial_answer = ""
    for line in response.iter_lines():
        print("Raw line:", line)
        if line:
            try:
                json_line = json.loads(line)
                token = json_line.get("response", "")
                yield token
            except Exception as e:
                print("JSON parse error:", e)
    return partial_answer

# UI
st.title("🔎 RAG Q&A System with Gemini")

query = st.text_input("Ask a question related to Computer Science:", "")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Searching..."):
            try:
                docs = vectorstore.similarity_search(query, k=3)
                top_docs = rerank(query, docs)

                context = "\n\n".join([doc.page_content for doc in top_docs])[:MAX_CONTEXT_LEN]
                prompt = f"""Answer the following question based on the context below.

Context:
{context}

Question:
{query}

Answer:"""

                print("Retrieved docs:", len(docs))
                print("Top reranked docs:", len(top_docs))
                print("Prompt:", prompt)

                # st.success("Answer generated below ⬇️")
                # st.markdown("### 💬 Answer:")

                # Streaming response (fixing st.write and partial display)
                full_response = ""
                response_placeholder = st.empty()
                for token in stream_ollama(prompt):
                    full_response += token
                    response_placeholder.markdown(f"### 💬 Answer:\n\n{full_response}")

                # Logging
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_data = {"timestamp": ts, "query": query, "context": context, "answer": full_response}
                with open(f"{LOG_DIR}/log_{ts}.json", "w", encoding="utf-8") as f:
                    json.dump(log_data, f, indent=2, ensure_ascii=False)

            except Exception as e:
                st.error("❌ An error occurred.")
                st.code(traceback.format_exc())  # Shows full traceback
