import requests
from bs4 import BeautifulSoup
import json
import os
import time
import re
import shutil
from urllib.parse import urljoin

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
            print(f"Deleted previous data from: {self.save_path}")

    def get_links(self):
        response = requests.get(self.base_url)
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
        print(f"Found {len(links)} links. Starting to scrape...")

        for idx, link in enumerate(links):
            try:
                print(f"Scraping: {link}")
                response = requests.get(link)
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

                print(f"Saved: {title}")

                if idx % 10 == 0:
                    time.sleep(2)

            except Exception as e:
                print(f"Error scraping {link}: {e}")

if __name__ == "__main__":
    base_url = "https://en.wikipedia.org/wiki/Outline_of_computer_science"
    scraper = WikipediaScraper(base_url)
    scraper.scrape_and_save()
