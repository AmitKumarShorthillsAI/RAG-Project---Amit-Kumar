import os
import faiss
import pickle
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import for HuggingFace embeddings
from langchain_community.vectorstores import FAISS  # Updated FAISS import from langchain_community
import google.generativeai as genai

# Load .env
load_dotenv()

# Load Gemini API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Step 1: Dummy Documents
documents = [
    "The history of India is about five thousand years old. It includes the Indus Valley Civilization and the later Vedic age.",
    "India became independent from British rule on 15th August 1947.",
    "The Mughal Empire was an early-modern empire that controlled much of South Asia between the 16th and 18th centuries.",
    "Mahatma Gandhi led India’s non-violent independence movement against British colonial rule."
]

# Step 2: Initialize HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Create Embeddings and FAISS Index
vector_store = FAISS.from_texts(documents, embedding_model)

# Save FAISS db for future use
vector_store.save_local("../embeddings/mini_faiss_index")

# Step 4: Search Query
query = "Who led the independence movement of India?"

# Embed query and search
retrieved_docs = vector_store.similarity_search(query, k=2)
retrieved_texts = "\n".join([doc.page_content for doc in retrieved_docs])

print("Retrieved Context:\n", retrieved_texts)

# Step 5: Ask Gemini
response = genai.GenerativeModel('gemini-1.5-flash').generate_content(
    f"Answer the following question based on the given context:\n\nContext:\n{retrieved_texts}\n\nQuestion: {query}"
)

print("\nGemini Response:\n", response.text)
