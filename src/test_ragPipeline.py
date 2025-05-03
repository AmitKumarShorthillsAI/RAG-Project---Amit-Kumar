import os
import json
import shutil
import unittest
from pathlib import Path
from unittest import mock
from langchain.schema import Document
from rag_pipeline import RagPipeline
import logging

# Set up logging to a file
logging.basicConfig(
    filename="test_result_ragPipeline/test_results.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# You can still have console output
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

class TestRagPipeline(unittest.TestCase):
    def setUp(self):
        self.test_data_dir = "test_result_rag/temp_data"
        self.test_index_dir = "test_result_rag/index"
        os.makedirs(self.test_data_dir, exist_ok=True)
        self.test_file_path = os.path.join(self.test_data_dir, "test.json")
        content = {
            "title": "Test Title",
            "content": "This is some long enough test content. " * 5  # >100 chars
        }
        with open(self.test_file_path, "w", encoding="utf-8") as f:
            json.dump(content, f)
        self.pipeline = RagPipeline(raw_data_path=self.test_data_dir, index_path=self.test_index_dir)

    def tearDown(self):
        shutil.rmtree("test_result_rag", ignore_errors=True)

    # TC_RAG_001
    def test_init_paths_created(self):
        try:
            self.assertTrue(os.path.exists(self.pipeline.index_path))
            logger.info("Test TC_RAG_001: Passed, Index folder exists at {}".format(self.pipeline.index_path))
        except Exception as e:
            logger.error("Test TC_RAG_001 failed: {}".format(str(e)))
            self.fail("Test failed due to error")

    # TC_RAG_002
    def test_clean_text_removes_citations(self):
        try:
            text = "AI is popular [1] and useful [citation needed]."
            cleaned = self.pipeline.clean_text(text)
            self.assertNotIn("[1]", cleaned)
            self.assertNotIn("[citation needed]", cleaned.lower())
            logger.info("Test TC_RAG_002: Passed, Citations removed correctly")
        except Exception as e:
            logger.error("Test TC_RAG_002 failed: {}".format(str(e)))
            self.fail("Test failed due to error")

    # TC_RAG_003
    def test_clean_text_trims_spaces(self):
        try:
            text = "   Hello    world   "
            cleaned = self.pipeline.clean_text(text)
            self.assertEqual(cleaned, "Hello world")
            logger.info("Test TC_RAG_003: Passed, Spaces trimmed correctly")
        except Exception as e:
            logger.error("Test TC_RAG_003 failed: {}".format(str(e)))
            self.fail("Test failed due to error")

    # TC_RAG_004
    def test_fuzzy_duplicate_detects_similar(self):
        try:
            prev = ["The quick brown fox jumps over the lazy dog."]
            result = self.pipeline.fuzzy_duplicate("The quick brown fox jumps over the lazy dog!", prev)
            self.assertTrue(result)
            logger.info("Test TC_RAG_004: Passed, Detected similar text correctly")
        except Exception as e:
            logger.error("Test TC_RAG_004 failed: {}".format(str(e)))
            self.fail("Test failed due to error")

    # TC_RAG_005
    def test_fuzzy_duplicate_detects_unique(self):
        try:
            prev = ["Completely unrelated sentence."]
            result = self.pipeline.fuzzy_duplicate("Another distinct string.", prev)
            self.assertFalse(result)
            logger.info("Test TC_RAG_005: Passed, Detected unique text correctly")
        except Exception as e:
            logger.error("Test TC_RAG_005 failed: {}".format(str(e)))
            self.fail("Test failed due to error")

    # TC_RAG_006
    def test_load_and_chunk_returns_list(self):
        try:
            chunks = self.pipeline.load_and_chunk_documents()
            self.assertIsInstance(chunks, list)
            logger.info("Test TC_RAG_006: Passed, Returned a list of chunks")
        except Exception as e:
            logger.error("Test TC_RAG_006 failed: {}".format(str(e)))
            self.fail("Test failed due to error")

    # TC_RAG_007
    def test_chunk_elements_are_documents(self):
        try:
            chunks = self.pipeline.load_and_chunk_documents()
            self.assertTrue(all(isinstance(c, Document) for c in chunks))
            logger.info("Test TC_RAG_007: Passed, All chunks are Document instances")
        except Exception as e:
            logger.error("Test TC_RAG_007 failed: {}".format(str(e)))
            self.fail("Test failed due to error")

    # TC_RAG_008
    def test_chunk_has_metadata(self):
        try:
            chunks = self.pipeline.load_and_chunk_documents()
            for c in chunks:
                self.assertIn("title", c.metadata)
                self.assertIn("source", c.metadata)
            logger.info("Test TC_RAG_008: Passed, Chunks contain metadata")
        except Exception as e:
            logger.error("Test TC_RAG_008 failed: {}".format(str(e)))
            self.fail("Test failed due to error")

    # TC_RAG_009
    def test_chunking_generates_chunks(self):
        try:
            chunks = self.pipeline.load_and_chunk_documents()
            self.assertGreater(len(chunks), 0)
            logger.info("Test TC_RAG_009: Passed, Chunks generated successfully")
        except Exception as e:
            logger.error("Test TC_RAG_009 failed: {}".format(str(e)))
            self.fail("Test failed due to error")

    # TC_RAG_010
    @mock.patch("rag_pipeline.FAISS")
    @mock.patch("rag_pipeline.tqdm")
    def test_embed_and_store_runs_successfully(self, mock_tqdm, mock_faiss):
        try:
            chunks = self.pipeline.load_and_chunk_documents()

            # Mock embedding return value to avoid IndexError
            mock_embedder = mock.Mock()
            mock_embedder.embed_documents.return_value = [[0.1] * 768 for _ in chunks]
            self.pipeline.embedding_model = mock_embedder

            # Mock FAISS storage
            mock_faiss.from_documents.return_value = mock.Mock()
            mock_faiss.from_documents.return_value.save_local = mock.Mock()

            self.pipeline.embed_and_store(chunks)

            mock_faiss.from_documents.assert_called_once()
            mock_faiss.from_documents.return_value.save_local.assert_called_once()

            logger.info("Test TC_RAG_010: Passed, Embedding and storing completed successfully")
        except Exception as e:
            logger.error("Test TC_RAG_010 failed: {}".format(str(e)))
            self.fail("Test failed due to error")

    # TC_RAG_011
    def test_run_pipeline_completes(self):
        try:
            # Mock embed_and_store to avoid full embedding
            self.pipeline.embed_and_store = mock.Mock()
            self.pipeline.run()
            self.pipeline.embed_and_store.assert_called_once()
            logger.info("Test TC_RAG_011: Passed, Full pipeline completed successfully")
        except Exception as e:
            logger.error("Test TC_RAG_011 failed: {}".format(str(e)))
            self.fail("Test failed due to error")

    # TC_RAG_012
    def test_file_content_and_title_extraction(self):
        try:
            chunks = self.pipeline.load_and_chunk_documents()
            for c in chunks:
                self.assertTrue(c.page_content.startswith("Test Title"))
            logger.info("Test TC_RAG_012: Passed, Title extracted correctly")
        except Exception as e:
            logger.error("Test TC_RAG_012 failed: {}".format(str(e)))
            self.fail("Test failed due to error")


if __name__ == "__main__":
    unittest.main()
