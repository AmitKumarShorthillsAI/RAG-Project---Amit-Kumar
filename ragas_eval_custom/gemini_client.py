## ragas_eval_custom/gemini_client.py
import google.generativeai as genai
import time
import os
import google.api_core.exceptions
import json
import re

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

def call_gemini(prompt, model):
    max_retries = 5
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except google.api_core.exceptions.ResourceExhausted as e:
            print(f"Rate limit hit: {e.message}")
            wait_time = (attempt + 1) * retry_delay
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Other error in call_gemini: {e}")
            break

    return "Error: Failed after retries"


def call_combined_metrics(prompt, model):
    response = call_gemini(prompt, model)

    if response.startswith("Error:"):
        return None

    try:
        # Expecting JSON-like output
        parsed = extract_clean_json(response)
        metrics = {
            "faithfulness": parsed.get("faithfulness"),
            "answer_relevance": parsed.get("answer_relevance"),
            "answer_correctness": parsed.get("answer_correctness"),
            "context_precision": parsed.get("context_precision"),
            "context_recall": parsed.get("context_recall"),
        }
        return metrics
    except json.JSONDecodeError:
        print("❌ Failed to parse Gemini response as JSON.")
        print(f"Response: {response}")
        return None


def extract_clean_json(response_text: str) -> dict:
    # Remove ```json ... ``` wrapper if present
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = response_text.strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("❌ Failed to parse cleaned response as JSON.")
        print("Cleaned response:", json_str)
        return None
