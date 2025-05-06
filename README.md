# 🧠 RAG-LLM: Wikipedia Q\&A System with Evaluation

## Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Folder Structure](#2-folder-structure)
- [3. Architecture Overview](#3-architecture-overview)
- [4. Scraping Pipeline](#4-scraping-pipeline)
- [5. Chunking, Embedding & FAISS Storage](#5-chunking-embedding-faiss-storage)
- [6. Streamlit UI Chatbot](#6-streamlit-ui-chatbot)
- [7. Evaluation Strategy](#7-evaluation-strategy)
- [8. Improvements and Enhancements](#8-improvements-and-enhancements)
- [9. How to Run](#9-how-to-run)
- [10. Future Work](#10-future-work)
- [11. License](#11-license)
- [12. Acknowledgements](#12-acknowledgements)

## 1. 🚀 Project Overview

This project implements an end-to-end Retrieval-Augmented Generation (RAG) system designed to generate high-quality answers from scraped Wikipedia pages under the "Outline of computer science" category. The solution leverages state-of-the-art tools and models to build a production-ready pipeline, supporting everything from data scraping to question-answer evaluation.

### 🌐 Scraped Pages:

* Pages under Wikipedia's "Outline of Computer Science"
* Categories: Algorithms, Data Structures, Programming Languages, Databases, Networking, AI, etc.

### 📑 Key Features:

* **Scraping** Wikipedia pages and storing them as clean JSON
* **Chunking** the content with `RecursiveCharacterTextSplitter`
* **Embedding** using **BAAI/bge-large-en-v1.5** model
* **Storing** embeddings into **FAISS** vector database
* **Reranking** with `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **Answer generation** with `llama3:8b` via **Ollama**
* **UI** built in **Streamlit**
* **Evaluation** using both LLM-based (RAGAS) and non-LLM-based (BERTScore, ROUGE, BLEU) methods

### New Evaluation Method:

* **Q&A Generation**: Generated through OpenAI by manually copy-pasting the content of scraped data into ChatGPT.
* **Answer and Context Generation**: Generated from the RAG pipeline using LLaMA3 via Ollama.
* **Evaluation**: Conducted using Gemini-2.0-flash, ensuring no bias from LLMs.

## 2. 🗂️ Folder Structure

```bash
RAG Project/
│
├── app/                        # Streamlit UI
│   ├── ui_app.py              # Main chatbot UI
│   ├── test_chatbot_ui.py     # UI test cases
│   └── logs/                  # Chat interaction logs
│
├── scrapers/                  # Scraping Wikipedia content
│   ├── wiki_scraper.py        # Core scraper module
│   ├── test_scraper.py        # Unit tests
│   ├── test_result_scraper/   # Logs and results
│   └── data/                  # Raw JSON output from scraping
│
├── src/                       # Core RAG pipeline
│   ├── rag_pipeline.py        # Chunking + embedding + saving
│   ├── mini_rag.py            # Minimal version for testing
│   ├── test_ragPipeline.py    # Pipeline unit tests
│   └── embeddings/            # FAISS index stored here
│
├── GroundTruth_Context_Generation/
│   ├── eval_data.csv               # Ground-truth Q&A
│   ├── eval_data_enriched.csv     # Enriched with generated answers/context
│   ├── groundTruth_context.py     # Script to enrich with generation
│ 
│
├── bert_evaluation/               # Traditional metric-based evaluation
│   ├── bert_score_eval.py         # Main script (BERTScore, ROUGE, BLEU)
│   ├── input.csv                  # Evaluation input
│   └── bert_eval_output/
│       ├── logs/                  # Logs
│       └── outputs/               # Evaluation results
│
├── ragas_eval_custom/            # Gemini/LLM-based custom evaluator(RAGAS Mimic)
│   ├── evaluate.py               # Main evaluation script
│   ├── gemini_client.py          # Gemini API client
│   ├── prompts.py                # Evaluation prompts
│   ├── score                     # Metric score saved
│   ├── logs_error                # Logs and errors get saved here
│   └── input/                    # Input data for evaluation
│
├── requirements.txt              # Dependencies
├── .gitignore                    # git ignore file
├── .env                          # for keeping environment variables
├── venv                          # virtual env. folder, make sure to add it in gitignore
└── README.md
```

## 3. 🧭 Architecture Overview

[![Cursor Logo](/assets/RAG-Architecture.png)](https://app.eraser.io/workspace/MbIksknysMwQ7GE9LrdF)

---

## 4. 🕸️ Scraping Pipeline

### Core Module: `wiki_scraper.py`

* Uses `wikipediaapi` and `BeautifulSoup` to extract clean text
* Targets pages from the "Outline of Computer Science"
* Saves JSON content in `scrapers/data/raw_scraped/`
* Logs every scraped URL

### Workflow:

1. Delete old scraped files to save space
2. Recursively navigate linked pages
3. Extract paragraphs and skip unrelated content
4. Save clean structured output in JSON

### Key Methods:

* `scrape_outline()`: Starts the full scraping session
* `parse_page()`: Extracts clean text from one page
* `save_json()`: Dumps structured content to disk

---

## 5. ✂️ Chunking, Embedding & FAISS Storage

### Chunking:

* **Tool:** `RecursiveCharacterTextSplitter`
* **Config:** chunk size = 512, overlap = 50

### Comparison of Chunking Strategies:

| Method                         | Pros                                | Cons                        |
| ------------------------------ | ----------------------------------- | --------------------------- |
| Fixed-size (naïve)             | Simple, fast                        | Breaks context mid-sentence |
| RecursiveCharacterTextSplitter | Preserves structure (headings etc.) | Slightly slower             |
| Token-based (e.g., tiktoken)   | More accurate with LLMs             | More complex                |

➡️ **Selected RecursiveCharacterTextSplitter** to ensure sentence boundaries and heading preservation.

### Embedding:

* **Model Used:** `BAAI/bge-large-en-v1.5`
* **Why Chosen:**

  * Optimized for English
  * Trained for retrieval tasks
  * Better semantic separation

### Model Specifications:

* Parameters: 300M+
* Dimensionality: 1024
* Pretrained on English web/corpora

### Comparison of Embedding Models:

| Model                                    | Dim  | Retrieval Quality | Speed  | Comments                      |
| ---------------------------------------- | ---- | ----------------- | ------ | ----------------------------- |
| `sentence-transformers/all-MiniLM-L6-v2` | 384  | Medium            | Fast   | Lightweight, baseline         |
| `BAAI/bge-small-en-v1.5`                 | 384  | Medium-High       | Fast   | Good for compact systems      |
| `BAAI/bge-large-en-v1.5`                 | 1024 | **High**          | Medium | Best semantic matching so far |
| `multi-qa-mpnet-base-dot-v1`             | 768  | High              | Medium | Multi-lingual support         |

### Workflow:

1. Load JSON data
2. Clean and deduplicate chunks
3. Apply recursive chunking
4. Generate embeddings via `bge-large-en-v1.5`
5. Save to FAISS index in `/src/embeddings/full_faiss_index/`

---

## 6. 🧑‍💻 Streamlit UI Chatbot

### File: `app/ui_app.py`

* Loads FAISS index
* Retrieves top-k chunks
* Reranks using `cross-encoder/ms-marco-MiniLM-L-6-v2`
* Sends best-ranked chunks to LLaMA 3 (8B) via Ollama
* Displays chat interface and logs interactions

### UI Features:

* Clear historical context display
* Real-time generation
* Logging user queries and responses

---

## 7. 📊 Evaluation Strategy

### A. BERT-based Traditional Evaluation

#### File: `bert_score_eval.py`

* Input: `input.csv` with Q, Ground Truth A, Generated A
* Output: ROUGE, BLEU, BERTScore

#### Metrics Used:

| Metric        | Compares                     | Score Range | Meaning                                   |
| ------------- | ---------------------------- | ----------- | ----------------------------------------- |
| **BLEU**      | n-gram overlap               | 0–1         | Precision-oriented score                  |
| **ROUGE**     | recall-oriented overlap      | 0–1         | Measures recall of n-grams                |
| **BERTScore** | contextual embeddings (BERT) | 0–1         | Semantic similarity (better than n-grams) |

* **BLEU**: Measures how much the generated answer overlaps with ground truth using n-gram matching.
* **ROUGE**: Measures how much of the ground truth is captured in the generated answer.
* **BERTScore**: Uses BERT embeddings to compare semantic similarity of sentences.

### B. LLM-Based (RAGAS-style)

#### File: `ragas_eval_custom/`

* Uses Gemini to score:

  * Faithfulness
  * Context relevance
  * Answer correctness
  * Context precision
  * Context recall

#### Metrics and Scores:

| Metric              | Score Range | Meaning                                   |
| ------------------- | ----------- | ----------------------------------------- |
| **Faithfulness**    | 0–1         | How well the answer stays true to context |
| **Relevance**       | 0–1         | How relevant the answer is to the question|
| **Correctness**     | 0–1         | How correct the answer is compared to ground truth|
| **Precision**       | 0–1         | How well the context supports the answer  |
| **Recall**          | 0–1         | How complete the answer is given the context|

* **Scores Obtained**: 
  - Faithfulness: 0.9851
  - Relevance: 0.9902
  - Correctness: 0.9480
  - Precision: 0.9644
  - Recall: 0.9030

* **Interpretation**: Higher scores indicate better performance. The scores obtained are above average, indicating a strong performance in maintaining context and relevance.

* **Reliability**: The method is reliable for contextual and semantic evaluation but may lack precision in isolated metric evaluation compared to traditional methods.

#### Average Score Calculation:

* **Simple Average**: Calculated as the mean of all metric scores.
* **Weighted Average**: Calculated using the following weights:
  - Faithfulness: 0.30
  - Relevance: 0.25
  - Correctness: 0.25
  - Precision: 0.10
  - Recall: 0.10

* **Final Scores**:
  - Simple Average Score: 0.9581
  - Weighted Average Score: 0.9668

### Comparison:

| Metric Type | Tool Used | Key Insight                 |
| ----------- | --------- | --------------------------- |
| Non-LLM     | BERTScore | Precise but surface level   |
| LLM-based   | Gemini    | Deeper, contextual judgment |

#### Pros and Cons:

* **Pros**:
  * Provides a deeper, more contextual evaluation.
  * Can capture nuances missed by traditional metrics.

* **Cons**:
  * May not be as precise in isolated metric evaluation.
  * Relies on the quality of the LLM and prompt design.

---

## 8. 🛠️ Improvements and Enhancements

* Rewrote scraper to log and clean more effectively
* Replaced `bge-small` with **bge-large-en-v1.5** for better semantic quality
* Integrated reranker to boost answer relevance
* Improved chunking strategy (overlap + recursion)
* Streamlit UI now shows previous questions and logs interactions
* Enhanced evaluation: both LLM and traditional metrics now included

---

## 9. ➡️ How to Run

1. **Clone the Repository**:
   ```bash
   git clone [repository-url]
   cd RAG-Project
   ```

2. **Set Up Environment**:
   * Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   * Set up Ollama for LLaMA3:
     * [Instructions for setting up Ollama]

3. **Run the Pipeline**:
   * Scrape Wikipedia:
     ```bash
     python scrapers/wiki_scraper.py
     ```
   * Run chunking and embedding:
     ```bash
     python src/rag_pipeline.py
     ```
   * Launch chatbot UI:
     ```bash
     streamlit run app/ui_app.py
     ```

4. **Evaluate**:
   * With BERT:
     ```bash
     python bert_evaluation/bert_score_eval.py
     ```
   * With Gemini:
     ```bash
     python ragas_eval_custom/evaluate.py
     ```

---

## ✅ Future Work

* Complete RAGA based testing
* Add support for multilingual scraping and embedding
* Experiment with hybrid reranking (BM25 + cross-encoder)
* Extend evaluation framework with Hallucination detection (e.g., TruthfulQA)
* Integrate feedback loop to improve QA generation over time

---

## 📌 License

MIT License

## �� Acknowledgements

* BAAI for BGE model
* HuggingFace for Transformers & evaluation
* Langchain for inspiration on RAG design
* Google Gemini, and Meta for LLM APIs
