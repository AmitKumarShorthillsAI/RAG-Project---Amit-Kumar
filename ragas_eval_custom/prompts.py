combined_eval_prompt = """
You are a helpful evaluator tasked with rating the quality of an AI-generated answer using several metrics.

Evaluate the following based on the definitions below and return only a JSON object with the five scores (as floats between 0 and 1):

Definitions:
- Faithfulness: How well does the answer stay true to the context?
- Answer Relevance: How relevant is the answer to the user's question?
- Answer Correctness: How correct is the answer compared to the expected (ground truth)?
- Context Precision: How well does the context support the answer (avoid extra info)?
- Context Recall: How complete is the answer given the context?

Input:
Question: {question}

Ground Truth Answer:
{ground_truth}

Retrieved Context:
{context}

Generated Answer:
{generated_answer}

Output Format:
{{
  "faithfulness": float,
  "answer_relevance": float,
  "answer_correctness": float,
  "context_precision": float,
  "context_recall": float
}}
""".strip()
