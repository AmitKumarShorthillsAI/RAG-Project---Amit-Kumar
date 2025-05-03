import unittest
import os
import shutil
import json
from unittest.mock import patch, MagicMock
from wiki_scraper import WikipediaScraper

LOG_DIR = "test_result_scraper"
LOG_FILE = os.path.join(LOG_DIR, "log.json")

# Helper to log result
def log_result(test_case_id, status):
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    log_data = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            log_data = json.load(f)
    log_data.append({"test_case_id": test_case_id, "status": status})
    with open(LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=2)

class TestWikipediaScraper(unittest.TestCase):
    def setUp(self):
        self.test_path = os.path.join(LOG_DIR, "temp_data")
        self.scraper = WikipediaScraper(base_url="http://fakeurl.org", save_path=self.test_path)

    def tearDown(self):
        if os.path.exists(self.test_path):
            shutil.rmtree(self.test_path)

    def test_TC_WS_001_init_clears_old_data(self):
        os.makedirs(self.test_path, exist_ok=True)
        with open(os.path.join(self.test_path, "old.txt"), "w") as f:
            f.write("old")
        WikipediaScraper("http://fakeurl.org", self.test_path)
        self.assertTrue(os.path.exists(self.test_path))
        self.assertEqual(os.listdir(self.test_path), [])
        log_result("TC_WS_001", "Pass")

    @patch("wiki_scraper.requests.get")
    def test_TC_WS_002_get_links_extracts_valid(self, mock_get):
        html = '''
            <div id="mw-content-text">
                <a href="/wiki/AI">AI</a>
                <a href="/wiki/Compiler">Compiler</a>
                <a href="/wiki/File:Image.jpg">Image</a>
                <a href="/wiki/Help:Page">Help</a>
            </div>
        '''
        mock_get.return_value.text = html
        links = self.scraper.get_links()
        self.assertIn("http://fakeurl.org/wiki/AI", links)
        self.assertNotIn("http://fakeurl.org/wiki/File:Image.jpg", links)
        self.assertNotIn("http://fakeurl.org/wiki/Help:Page", links)
        log_result("TC_WS_002", "Pass")

    @patch("wiki_scraper.requests.get")
    def test_TC_WS_003_scrape_and_save_json(self, mock_get):
        mock_html = '''
            <html>
                <h1>Fake Title</h1>
                <div id="mw-content-text">
                    <p>First para</p><p>Second para</p>
                </div>
            </html>
        '''
        mock_get.return_value.text = mock_html
        with patch.object(self.scraper, "get_links", return_value=["http://fakeurl.org/wiki/Fake"]):
            self.scraper.scrape_and_save()
            files = os.listdir(self.test_path)
            self.assertTrue(any("Fake Title" in f for f in files))
            log_result("TC_WS_003", "Pass")

    @patch("wiki_scraper.requests.get")
    def test_TC_WS_004_skip_insufficient_paragraphs(self, mock_get):
        mock_get.return_value.text = '<h1>Skip</h1><div id="mw-content-text"><p>Only one</p></div>'
        with patch.object(self.scraper, "get_links", return_value=["http://fakeurl.org/wiki/Skip"]):
            self.scraper.scrape_and_save()
            self.assertEqual(len(os.listdir(self.test_path)), 0)
            log_result("TC_WS_004", "Pass")

    @patch("wiki_scraper.requests.get")
    def test_TC_WS_005_filename_sanitization(self, mock_get):
        mock_get.return_value.text = '''
            <h1>Some/Title</h1>
            <div id="mw-content-text"><p>Para 1</p><p>Para 2</p></div>
        '''
        with patch.object(self.scraper, "get_links", return_value=["http://fakeurl.org/wiki/Some/Title"]):
            self.scraper.scrape_and_save()
            self.assertTrue(any("Some_Title" in f for f in os.listdir(self.test_path)))
            log_result("TC_WS_005", "Pass")

    @patch("wiki_scraper.requests.get", side_effect=Exception("Failed"))
    def test_TC_WS_006_handle_exceptions(self, mock_get):
        with patch.object(self.scraper, "get_links", return_value=["bad"]):
            try:
                self.scraper.scrape_and_save()
                log_result("TC_WS_006", "Pass")
            except Exception:
                self.fail("Exception not handled properly")

    @patch("wiki_scraper.requests.get")
    @patch("wiki_scraper.time.sleep")
    def test_TC_WS_007_sleep_called(self, mock_sleep, mock_get):
        mock_html = '<h1>Title</h1><div id="mw-content-text"><p>1</p><p>2</p></div>'
        mock_get.return_value.text = mock_html
        with patch.object(self.scraper, "get_links", return_value=[f"http://x.com/{i}" for i in range(10)]):
            self.scraper.scrape_and_save()
            mock_sleep.assert_called_once_with(2)
            log_result("TC_WS_007", "Pass")

    def test_TC_WS_008_log_file_created(self):
        log_result("TC_WS_008", "Pass")
        self.assertTrue(os.path.exists(LOG_FILE))

if __name__ == "__main__":
    unittest.main()
