# bert_evaluation/bert_eval_metrics.py

import os
import pandas as pd
import logging
from bert_score import score
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

# Setup logging
log_path = "bert_evaluation/bert_score.log"
os.makedirs("bert_evaluation", exist_ok=True) # This will create the directory if it doesn't exist
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Load environment variables
load_dotenv(dotenv_path=".env")

# Load input CSV
input_path = "input.csv"
df = pd.read_csv(input_path)
logging.info(f"Loaded input CSV with {len(df)} rows.")

# Drop any rows with missing values
df.dropna(subset=["ground_truth", "generated_answer"], inplace=True)
logging.info(f"Dropped {len(df[df.isnull().any(axis=1)])} rows due to missing values.")

# Prepare model for semantic similarity
logging.info("Loading sentence transformer model for cosine similarity and semantic analysis...")
# sim_model = SentenceTransformer('all-MiniLM-L6-v2')
# sim_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
sim_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Prepare lists
references = df["ground_truth"].tolist()
candidates = df["generated_answer"].tolist()

# BERTScore Precision
logging.info("Running BERTScore using 'microsoft/deberta-base' (balanced model)...")
P, _, _ = score(candidates, references, model_type="microsoft/deberta-base", lang="en", verbose=True) # other model_type includes 'bert-base-uncased', 'roberta-base', 'distilbert-base-uncased', 'microsoft/deberta-v3-base', 'microsoft/deberta-v3-large' but 'microsoft/deberta-base' is the best for this task

# Cosine Similarity + Semantic Analysis
cosine_scores = []
semantic_match = []

for ref, gen in zip(references, candidates):
    emb_ref = sim_model.encode(ref, convert_to_tensor=True)
    emb_gen = sim_model.encode(gen, convert_to_tensor=True)
    cos_sim = float(util.pytorch_cos_sim(emb_ref, emb_gen))
    cosine_scores.append(cos_sim)
    semantic_match.append("yes" if cos_sim > 0.7 else "no")

# Save results
output_df = pd.DataFrame({
    "bert_precision": P.tolist(),
    "cosine_similarity": cosine_scores,
    "semantic_match": semantic_match
})

# Append averages at the bottom
avg_row = pd.DataFrame({
    "bert_precision": [output_df["bert_precision"].mean()],
    "cosine_similarity": [output_df["cosine_similarity"].mean()],
    "semantic_match": ["average"]
})

output_df = pd.concat([output_df, avg_row], ignore_index=True)
output_df.to_csv("bert_evaluation/bert_precision_scores.csv", index=False)

logging.info("✅ Evaluation complete. Results saved to 'bert_evaluation/bert_precision_scores.csv'")
logging.info("📊 Average Scores: BERT Precision: %.4f | Cosine Similarity: %.4f" % (
    output_df["bert_precision"].iloc[-1],
    output_df["cosine_similarity"].iloc[-1]
))
