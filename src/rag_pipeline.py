# src/rag_pipeline.py

import os
import json
import shutil
import faiss
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.cluster import KMeans
import numpy as np

DATA_DIR = "../scrapers/data/raw_scraped"
INDEX_SAVE_PATH = "../embeddings/full_faiss_index"

class DocumentLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_documents(self):
        documents = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                path = os.path.join(self.data_dir, filename)
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    title = data.get("title", "")
                    content = data.get("content", "")
                    documents.append(Document(page_content=content, metadata={"title": title}))
        return documents

class Chunker:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            # chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def split_documents(self, documents):
        return self.splitter.split_documents(documents)

class Embedder:
    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        self.model = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, documents):
        return self.model.embed_documents([doc.page_content for doc in documents])

class VectorStoreManager:
    def __init__(self, embedder):
        self.embedder = embedder

    def create_vector_store(self, documents):
        vectorstore = FAISS.from_documents(documents, self.embedder.model)
        return vectorstore

    def save_vector_store(self, vectorstore, save_path):
        vectorstore.save_local(save_path)

def clear_old_index(path):
    """
    Delete the existing FAISS index directory if it exists.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def cluster_documents(docs, embedder, query, num_clusters=3):
    if not docs:
        return []

    # Embed the documents
    doc_texts = [doc.page_content for doc in docs]
    doc_embeddings = embedder.embed_documents(doc_texts)
    doc_embeddings = np.array(doc_embeddings)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    cluster_ids = kmeans.fit_predict(doc_embeddings)

    # Find cluster whose centroid is closest to query
    query_embedding = np.array(embedder.embed_query(query))
    centroid_distances = np.linalg.norm(kmeans.cluster_centers_ - query_embedding, axis=1)
    best_cluster = np.argmin(centroid_distances)

    # Pick documents from best cluster
    clustered_docs = [doc for doc, cid in zip(docs, cluster_ids) if cid == best_cluster]

    return clustered_docs

def main():
    print("Loading documents...")
    loader = DocumentLoader(DATA_DIR)
    documents = loader.load_documents()
    print(f"Loaded {len(documents)} documents.")

    print("Splitting into chunks...")
    chunker = Chunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split_documents(documents)
    print(f"Generated {len(chunks)} chunks.")

    print("Initializing embedder...")
    embedder = Embedder()

    print("Clearing old FAISS index...")
    clear_old_index(INDEX_SAVE_PATH)  # <<<<<< Added clearing function here!

    print("Creating FAISS vectorstore...")
    manager = VectorStoreManager(embedder)
    vectorstore = manager.create_vector_store(chunks)

    print("Saving vectorstore locally...")
    manager.save_vector_store(vectorstore, INDEX_SAVE_PATH)

    print(f"Vectorstore successfully saved at {INDEX_SAVE_PATH}")

if __name__ == "__main__":
    main()
