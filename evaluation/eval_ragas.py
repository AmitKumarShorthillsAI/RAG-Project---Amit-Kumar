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

# === Constants ===
INDEX_PATH = "../embeddings/full_faiss_index"
INPUT_CSV_PATH = "eval_data.csv"
OUTPUT_CSV_PATH = "eval_data_enriched.csv"
LOG_DIR = "logs_eval"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:8b"
MAX_CONTEXT_LEN = 1200
TOP_K = 5

# === Setup ===
os.makedirs(LOG_DIR, exist_ok=True)

# === Load vectorstore and reranker ===
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    return FAISS.load_local(INDEX_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)

def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

vectorstore = load_vectorstore()
reranker = load_reranker()

# === Reranking ===
def rerank(query, docs, top_k=3):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked_docs[:top_k]]

# === Answer generation ===
def generate_answer(prompt):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "[ERROR]"

# === Main Process with checkpointing ===
def process_and_save():
    # Load enriched data if it exists
    if os.path.exists(OUTPUT_CSV_PATH):
        df = pd.read_csv(OUTPUT_CSV_PATH)
        print(f"🔁 Resuming from existing file: {OUTPUT_CSV_PATH}")
    else:
        df = pd.read_csv(INPUT_CSV_PATH)
        df["generated_answer"] = ""
        df["retrieved_context"] = ""
        print(f"🆕 Starting fresh from: {INPUT_CSV_PATH}")

    print(f"Processing {len(df)} questions...")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        if pd.notna(row["generated_answer"]) and str(row["generated_answer"]).strip():
            continue  # Skip already completed

        query = str(row["Question"])

        try:
            docs = vectorstore.similarity_search(query, k=TOP_K)
            top_docs = rerank(query, docs)
            context = "\n\n".join([doc.page_content for doc in top_docs])[:MAX_CONTEXT_LEN]

            prompt = f"""Answer the following question based on the context below.

Context:
{context}

Question:
{query}

Answer:"""

            answer = generate_answer(prompt)

            df.at[i, "generated_answer"] = answer
            df.at[i, "retrieved_context"] = context

            # Save after each entry to prevent loss
            df.to_csv(OUTPUT_CSV_PATH, index=False)

            # Save log only if not already present
            log_path = os.path.join(LOG_DIR, f"log_{i:04d}.json")
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
            df.to_csv(OUTPUT_CSV_PATH, index=False)

        time.sleep(0.5)

    print(f"\n✅ Done. Results saved to: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    process_and_save()
