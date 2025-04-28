import faiss
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load FAISS index from local file
index_path = "/home/shtlp_0058/Desktop/RAG Project/embeddings/mini_faiss_index"

# Check if the FAISS index directory exists
if not os.path.exists(index_path):
    print(f"FAISS index directory not found at {index_path}")
else:
    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the FAISS index with dangerous deserialization allowed (since you trust the file)
    vector_store = FAISS.load_local(
        index_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True  # ADD THIS
    )

    # Test search to verify the index
    query = "Who led the independence movement of India?"
    retrieved_docs = vector_store.similarity_search(query, k=2)

    # Print the retrieved documents
    retrieved_texts = "\n".join([doc.page_content for doc in retrieved_docs])
    print("Retrieved Context from FAISS:\n", retrieved_texts)
