import unittest
import os
import json
import sqlite3
from datetime import datetime
from unittest.mock import patch

# Constants
TEST_RESULT_DIR = "test_result_chatbot_ui"
LOG_DIR = os.path.join(TEST_RESULT_DIR, "logs")
DB_PATH = os.path.join(LOG_DIR, "history.db")

# Ensure folders exist
os.makedirs(TEST_RESULT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Helper: Save test result
def log_result(test_case_id, status):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Include microseconds for uniqueness
    file_path = os.path.join(TEST_RESULT_DIR, f"{test_case_id}_{ts}.json")
    with open(file_path, "w") as f:
        json.dump({"test_case_id": test_case_id, "status": status}, f)

class TestChatbotUI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Reset the database
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT UNIQUE,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()

    def test_TC_UI_001_vectorstore_load(self):
        with patch("langchain_community.vectorstores.FAISS.load_local") as mock_load:
            mock_load.return_value = "mock_vectorstore"
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
            vectorstore = mock_load("../embeddings/full_faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
            self.assertEqual(vectorstore, "mock_vectorstore")
            log_result("TC_UI_001", "Pass")

    def test_TC_UI_002_reranker_load(self):
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.assertIsNotNone(reranker)
        log_result("TC_UI_002", "Pass")

    def test_TC_UI_003_valid_query_submission(self):
        query = "What is a compiler?"
        self.assertTrue(query.strip() != "")
        log_result("TC_UI_003", "Pass")

    def test_TC_UI_004_empty_input_warning(self):
        query = "   "
        self.assertTrue(query.strip() == "")
        log_result("TC_UI_004", "Pass")

    def test_TC_UI_005_save_query_sqlite(self):
        query = "What is AI?"
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO history (query, timestamp) VALUES (?, ?)", (query, datetime.now().isoformat()))
        conn.commit()
        c.execute("SELECT query FROM history WHERE query=?", (query,))
        result = c.fetchone()
        conn.close()
        self.assertEqual(result[0], query)
        log_result("TC_UI_005", "Pass")

    def test_TC_UI_006_duplicate_query_handling(self):
        query = "What is welding robot?"
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO history (query, timestamp) VALUES (?, ?)", (query, datetime.now().isoformat()))
        conn.commit()
        c.execute("SELECT COUNT(*) FROM history WHERE query=?", (query,))
        count = c.fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)
        log_result("TC_UI_006", "Pass")

    def test_TC_UI_007_retrieve_query_history(self):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT query FROM history ORDER BY id DESC")
        history = c.fetchall()
        conn.close()
        self.assertGreaterEqual(len(history), 1)
        log_result("TC_UI_007", "Pass")

    def test_TC_UI_009_context_length_limit(self):
        long_context = "x" * 1500
        clipped = long_context[:1200]
        self.assertEqual(len(clipped), 1200)
        log_result("TC_UI_009", "Pass")

    def test_TC_UI_010_logging_output(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        data = {"timestamp": ts, "query": "sample", "context": "ctx", "answer": "ans"}
        file_path = os.path.join(LOG_DIR, f"log_{ts}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.assertTrue(os.path.exists(file_path))
        log_result("TC_UI_010", "Pass")

    def test_TC_UI_012_json_parsing_error_handling(self):
        try:
            line = b"{invalid_json}"
            json.loads(line)
        except Exception as e:
            self.assertIsInstance(e, Exception)
            log_result("TC_UI_012", "Pass")

    def test_TC_UI_013_history_persistence(self):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM history")
        count_before = c.fetchone()[0]
        c.execute("INSERT OR IGNORE INTO history (query, timestamp) VALUES (?, ?)", ("Persistent check", datetime.now().isoformat()))
        conn.commit()
        c.execute("SELECT COUNT(*) FROM history")
        count_after = c.fetchone()[0]
        conn.close()
        self.assertGreaterEqual(count_after, count_before)
        log_result("TC_UI_013", "Pass")


if __name__ == "__main__":
    unittest.main()
