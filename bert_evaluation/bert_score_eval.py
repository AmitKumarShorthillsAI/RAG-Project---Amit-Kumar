import os
import pandas as pd
import logging
from bert_score import score
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from dotenv import load_dotenv

# ----------------------------- Setup -----------------------------
# Create output folders
os.makedirs("bert_eval_output/logs", exist_ok=True)
os.makedirs("bert_eval_output/outputs", exist_ok=True)

# Setup logging
log_path = "bert_eval_output/logs/bert_score.log"
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

# ----------------------------- Load Data -----------------------------
# Load input CSV
input_path = "input.csv"
df = pd.read_csv(input_path)
logging.info(f"Loaded input CSV with {len(df)} rows.")

# Drop missing values
initial_len = len(df)
df.dropna(subset=["ground_truth", "generated_answer"], inplace=True)
dropped = initial_len - len(df)
logging.info(f"Dropped {dropped} rows due to missing values.")

# Prepare data
references = df["ground_truth"].tolist()
candidates = df["generated_answer"].tolist()

# ----------------------------- Load Models -----------------------------
logging.info("Loading models for evaluation...")
sim_model = SentenceTransformer('BAAI/bge-small-en-v1.5')  # For cosine similarity

# ----------------------------- LLM-based Metric: BERTScore -----------------------------
logging.info("Running BERTScore using 'microsoft/deberta-base'...")
P, _, _ = score(candidates, references, model_type="microsoft/deberta-base", lang="en", verbose=True)

# ----------------------------- Non-LLM Metrics -----------------------------
# Initialize scorers and smoothing
cosine_scores, semantic_match = [], []
bleu_scores, rouge1_scores, rougeL_scores = [], [], []
rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
smoothie = SmoothingFunction().method4

logging.info("Running non-LLM metrics (cosine, BLEU, ROUGE)...")

for ref, gen in zip(references, candidates):
    # Cosine Similarity (using sentence embeddings)
    emb_ref = sim_model.encode(ref, convert_to_tensor=True)
    emb_gen = sim_model.encode(gen, convert_to_tensor=True)
    cos_sim = float(util.pytorch_cos_sim(emb_ref, emb_gen))
    cosine_scores.append(cos_sim)
    semantic_match.append("yes" if cos_sim > 0.7 else "no")

    # BLEU Score
    bleu = sentence_bleu([ref.split()], gen.split(), smoothing_function=smoothie)
    bleu_scores.append(bleu)

    # ROUGE Scores
    scores = rouge.score(gen, ref)
    rouge1_scores.append(scores["rouge1"].fmeasure)
    rougeL_scores.append(scores["rougeL"].fmeasure)

# ----------------------------- Save Output -----------------------------
output_df = pd.DataFrame({
    "bert_precision": P.tolist(),
    "cosine_similarity": cosine_scores,
    "semantic_match": semantic_match,
    "bleu": bleu_scores,
    "rouge_1": rouge1_scores,
    "rouge_l": rougeL_scores
})

# Add average row
avg_row = {
    "bert_precision": output_df["bert_precision"].mean(),
    "cosine_similarity": output_df["cosine_similarity"].mean(),
    "semantic_match": "average",
    "bleu": output_df["bleu"].mean(),
    "rouge_1": output_df["rouge_1"].mean(),
    "rouge_l": output_df["rouge_l"].mean()
}
output_df = pd.concat([output_df, pd.DataFrame([avg_row])], ignore_index=True)

# Save to CSV
output_path = "bert_eval_output/outputs/bert_precision_scores.csv"
output_df.to_csv(output_path, index=False)

# ----------------------------- Final Logs -----------------------------
logging.info("✅ Evaluation complete.")
logging.info(f"📁 Results saved to '{output_path}'")
logging.info("📊 Average Scores: BERT Precision: %.4f | Cosine Similarity: %.4f | BLEU: %.4f | ROUGE-1: %.4f | ROUGE-L: %.4f" % (
    avg_row["bert_precision"],
    avg_row["cosine_similarity"],
    avg_row["bleu"],
    avg_row["rouge_1"],
    avg_row["rouge_l"]
))
