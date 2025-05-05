import pandas as pd
from tqdm import tqdm
import os
from prompts import combined_eval_prompt
from gemini_client import call_combined_metrics

INPUT_CSV = "input/enriched_eval_data.csv"
OUTPUT_CSV = "score/enriched_eval_with_scores.csv"
LOG_FILE = "logs_errors/logs/gemini_logs.txt"

def ensure_dirs():
    os.makedirs("logs_errors/logs", exist_ok=True)
    os.makedirs("logs_errors/bad_outputs", exist_ok=True)
    os.makedirs("score", exist_ok=True)

def main():
    ensure_dirs()
    df = pd.read_csv(INPUT_CSV)

    # Check for existing output to resume
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV)
        processed_indices = set(existing_df.index)
        print(f"Resuming... {len(processed_indices)} rows already processed.")
    else:
        existing_df = pd.DataFrame()
        processed_indices = set()

    # Load Gemini model once
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Open log file in append mode
    with open(LOG_FILE, "a", encoding="utf-8") as log_f:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if idx in processed_indices:
                continue  # Skip already processed rows

            question = row["Question"]
            ground_truth = row["Answer"]
            context = row["retrieved_context"]
            generated_answer = row["generated_answer"]

            prompt = combined_eval_prompt.format(
                question=question,
                ground_truth=ground_truth,
                generated_answer=generated_answer,
                context=context
            )

            scores = call_combined_metrics(prompt, model)

            # Log and save
            output_row = row.to_dict()
            if scores:
                output_row.update(scores)
                log_f.write(f"[Index {idx}] Scores: {scores}\n\n")
            else:
                # If Gemini failed to return valid scores, mark as null or -1
                output_row.update({
                    "faithfulness": None,
                    "answer_relevance": None,
                    "answer_correctness": None,
                    "context_precision": None,
                    "context_recall": None,
                })
                log_f.write(f"[Index {idx}] ❌ Failed to extract combined scores\nPrompt:\n{prompt}\n\n")

            # Append to output CSV
            pd.DataFrame([output_row]).to_csv(
                OUTPUT_CSV,
                mode='a',
                header=not os.path.exists(OUTPUT_CSV),
                index=False
            )

# Final average score calculation
print("\nCalculating final average scores...")
result_df = pd.read_csv(OUTPUT_CSV)

metrics = ["faithfulness", "answer_relevance", "answer_correctness", "context_precision", "context_recall"]
valid_scores_df = result_df.dropna(subset=metrics)

# Convert to float just in case
for m in metrics:
    valid_scores_df[m] = valid_scores_df[m].astype(float)

# Simple average
avg_scores = valid_scores_df[metrics].mean().to_dict()

# Weighted average
weights = {
    "faithfulness": 0.30,
    "answer_relevance": 0.25,
    "answer_correctness": 0.25,
    "context_precision": 0.10,
    "context_recall": 0.10
}
weighted_avg = sum(avg_scores[m] * weights[m] for m in metrics)

summary_lines = [
    "\n--- Final Evaluation Summary ---",
    *(f"{metric:20}: {avg_scores[metric]:.4f}" for metric in metrics),
    f"\nSimple Average Score     : {sum(avg_scores.values()) / len(avg_scores):.4f}",
    f"Weighted Average Score   : {weighted_avg:.4f}",
    "-" * 40
]

# Print to console
for line in summary_lines:
    print(line)

# Append to log file
with open(LOG_FILE, "a", encoding="utf-8") as log_f:
    log_f.write("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
