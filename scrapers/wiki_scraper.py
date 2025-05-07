import requests
from bs4 import BeautifulSoup
import json
import os
import time
import re
import shutil
from urllib.parse import urljoin
from functools import lru_cache
import logging

# Setup logging directory at script level
SCRIPT_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "wiki_scraper.log")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# File handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class WikipediaScraper:
    def __init__(self, base_url, save_path="data/raw_scraped"):
        self.base_url = base_url
        self.save_path = save_path
        self.clear_previous_data()
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def clear_previous_data(self):
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
            logger.info(f"Deleted previous data from: {self.save_path}")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def get_links(self):
        response = self._cached_get(self.base_url)
        soup = BeautifulSoup(response.text, "html.parser")
        content_div = soup.find("div", {"id": "mw-content-text"})
        links = content_div.find_all("a")
        valid_links = [
            urljoin(self.base_url, link.get("href"))
            for link in links
            if link.get("href") and link.get("href").startswith("/wiki") and ":" not in link.get("href")
        ]
        return list(set(valid_links))

    def scrape_and_save(self):
        links = self.get_links()
        logger.info(f"Found {len(links)} links. Starting to scrape...")

        for idx, link in enumerate(links):
            try:
                logger.info(f"Scraping: {link}")
                response = self._cached_get(link)
                soup = BeautifulSoup(response.text, "html.parser")
                content_div = soup.find("div", {"id": "mw-content-text"})

                if not content_div or len(content_div.find_all("p")) < 2:
                    continue

                title = soup.find("h1").text.strip()
                paragraphs = content_div.find_all("p")
                content = "\n".join([p.text for p in paragraphs if p.text.strip()])

                content = content.replace("\xa0", " ").replace("[edit]", "")
                content = re.sub(r"\[\d+\]", "", content)
                content = re.sub(r"\s+", " ", content).strip()

                filename = os.path.join(self.save_path, f"{title.replace('/', '_')}.json")
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump({"title": title, "content": content}, f, ensure_ascii=False, indent=2)

                logger.info(f"Saved: {title}")

                if idx % 10 == 0:
                    time.sleep(2)

            except Exception as e:
                logger.error("Error scraping %s: %s", link, e)

    @staticmethod
    @lru_cache(maxsize=None)
    def _cached_get(url):
        return requests.get(url)

if __name__ == "__main__":
    base_url = "https://en.wikipedia.org/wiki/Outline_of_computer_science"
    scraper = WikipediaScraper(base_url)
    scraper.scrape_and_save()
