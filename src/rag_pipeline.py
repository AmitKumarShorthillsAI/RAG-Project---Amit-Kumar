import os
import json
import shutil
import time
from pathlib import Path
from typing import List
import re
import difflib
import logging

from tqdm import tqdm

from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class RagPipeline:
    def __init__(self, raw_data_path="../scrapers/data/raw_scraped", index_path="../embeddings/full_faiss_index", chunk_size=500, chunk_overlap=50):
        self.raw_data_path = raw_data_path
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Clear old FAISS index
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
            logging.info(f"Removed existing FAISS index at {index_path}")
        os.makedirs(index_path, exist_ok=True)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5"
        )

    def clean_text(self, text):
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def fuzzy_duplicate(self, chunk, previous_chunks, threshold=0.9):
        for prev in previous_chunks:
            similarity = difflib.SequenceMatcher(None, chunk, prev).ratio()
            if similarity > threshold:
                return True
        return False

    def load_and_chunk_documents(self) -> List[Document]:
        files = list(Path(self.raw_data_path).glob("*.json"))
        logging.info(f"Found {len(files)} JSON files to process.")

        all_chunks = []
        seen_chunks = []
        duplicate_filtered = 0
        chunk_lengths = []

        for file in tqdm(files, desc="📄 Chunking files"):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            title = data.get("title", "Untitled")
            content = self.clean_text(data.get("content", ""))

            if not content or len(content) < 100:
                continue

            chunks = self.splitter.split_text(content)
            for chunk in chunks:
                cleaned_chunk = self.clean_text(chunk)

                if self.fuzzy_duplicate(cleaned_chunk, seen_chunks):
                    duplicate_filtered += 1
                    continue

                seen_chunks.append(cleaned_chunk)
                chunk_lengths.append(len(cleaned_chunk))

                doc = Document(
                    page_content=f"{title}\n{cleaned_chunk}",
                    metadata={"title": title, "source": str(file)}
                )
                all_chunks.append(doc)

        logging.info(f"✅ Chunking complete: {len(all_chunks)} chunks")
        logging.info(f"🚫 Skipped {duplicate_filtered} duplicate/similar chunks")
        logging.info(f"📏 Avg chunk length: {sum(chunk_lengths) // len(chunk_lengths) if chunk_lengths else 0}")
        return all_chunks

    def embed_and_store(self, chunks: List[Document]):
        logging.info("🔍 Starting embedding...")

        start = time.time()

        # Wrap documents in a tqdm progress bar
        class tqdmEmbedding:
            def __init__(self, embedder, total):
                self.embedder = embedder
                self.pbar = tqdm(total=total, desc="💡 Embedding chunks")

            def embed_documents(self, texts):
                embeddings = []
                for text in texts:
                    embeddings.append(self.embedder.embed_documents([text])[0])
                    self.pbar.update(1)
                self.pbar.close()
                return embeddings

        embedder = tqdmEmbedding(self.embedding_model, total=len(chunks))
        db = FAISS.from_documents(chunks, embedder)
        db.save_local(self.index_path)

        end = time.time()
        logging.info(f"✅ Embedding complete in {round(end - start, 2)} seconds.")
        logging.info(f"📦 FAISS index stored at {self.index_path}")

    def run(self):
        chunks = self.load_and_chunk_documents()
        self.embed_and_store(chunks)

if __name__ == "__main__":
    pipeline = RagPipeline()
    pipeline.run()
    logging.info("✅ RAG pipeline completed successfully.")