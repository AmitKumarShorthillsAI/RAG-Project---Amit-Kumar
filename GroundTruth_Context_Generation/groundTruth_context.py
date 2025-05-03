import os
import csv
import json
import requests
import time
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

class GroundTruthContextProcessor:
    INDEX_PATH = "../embeddings/full_faiss_index"
    INPUT_CSV_PATH = "eval_data.csv"
    OUTPUT_CSV_PATH = "eval_data_enriched.csv"
    LOG_DIR = "logs_eval"
    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "llama3:8b"
    MAX_CONTEXT_LEN = 1200
    TOP_K = 5

    def __init__(self):
        os.makedirs(self.LOG_DIR, exist_ok=True)
        self.vectorstore = self.load_vectorstore()
        self.reranker = self.load_reranker()

    def load_vectorstore(self):
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        return FAISS.load_local(self.INDEX_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)

    def load_reranker(self):
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, docs, top_k=3):
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked_docs[:top_k]]

    def generate_answer(self, prompt):
        try:
            response = requests.post(
                self.OLLAMA_URL,
                json={"model": self.OLLAMA_MODEL, "prompt": prompt, "stream": False}
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "[ERROR]"

    def process_and_save(self):
        if os.path.exists(self.OUTPUT_CSV_PATH):
            df = pd.read_csv(self.OUTPUT_CSV_PATH)
            print(f"🔁 Resuming from existing file: {self.OUTPUT_CSV_PATH}")
        else:
            df = pd.read_csv(self.INPUT_CSV_PATH)
            df["generated_answer"] = ""
            df["retrieved_context"] = ""
            print(f"🆕 Starting fresh from: {self.INPUT_CSV_PATH}")

        print(f"Processing {len(df)} questions...")

        for i, row in tqdm(df.iterrows(), total=len(df)):
            if pd.notna(row["generated_answer"]) and str(row["generated_answer"]).strip():
                continue

            query = str(row["Question"])

            try:
                docs = self.vectorstore.similarity_search(query, k=self.TOP_K)
                top_docs = self.rerank(query, docs)
                context = "\n\n".join([doc.page_content for doc in top_docs])[:self.MAX_CONTEXT_LEN]

                prompt = f"""Answer the following question based on the context below.

Context:
{context}

Question:
{query}

Answer:"""

                answer = self.generate_answer(prompt)

                df.at[i, "generated_answer"] = answer
                df.at[i, "retrieved_context"] = context

                df.to_csv(self.OUTPUT_CSV_PATH, index=False)

                log_path = os.path.join(self.LOG_DIR, f"log_{i:04d}.json")
                if not os.path.exists(log_path):
                    log_data = {
                        "timestamp": datetime.now().isoformat(),
                        "index": i,
                        "query": query,
                        "generated_answer": answer,
                        "retrieved_context": context
                    }
                    with open(log_path, "w", encoding="utf-8") as f:
                        json.dump(log_data, f, indent=2, ensure_ascii=False)

            except Exception as e:
                print(f"❌ Error processing question {i}: {e}")
                df.at[i, "generated_answer"] = "[ERROR]"
                df.at[i, "retrieved_context"] = "[ERROR]"
                df.to_csv(self.OUTPUT_CSV_PATH, index=False)

            time.sleep(0.5)

        print(f"\n✅ Done. Results saved to: {self.OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    processor = GroundTruthContextProcessor()
    processor.process_and_save()
