# scrapers/wiki_scraper.py

import os
import requests
from bs4 import BeautifulSoup
import json
import time
import shutil
import re

BASE_URL = "https://en.wikipedia.org"
ROOT_PAGE = "/wiki/Computer_science"
SAVE_DIR = "data/raw_scraped"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAG-LLM-Scraper/1.0; +https://yourdomain.com/)"
}

def fetch_page(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None

def parse_links(html, max_links=50):
    """
    Parse main content and extract related sub-page links under Computer Science.
    """
    soup = BeautifulSoup(html, "html.parser")
    content_div = soup.find("div", {"id": "bodyContent"})
    links = set()

    for link in content_div.find_all("a", href=True):
        href = link["href"]
        if (href.startswith("/wiki/") and not any(x in href for x in [":", "#", "disambiguation", "List_of"])):
            full_url = BASE_URL + href
            links.add(full_url)
            if len(links) >= max_links:
                break

    return list(links)


def clean_text(html):
    soup = BeautifulSoup(html, "html.parser")
    main_content = soup.find("div", class_="mw-parser-output")
    if not main_content:
        print("❌ main_content not found")
        return ""

    content = []

    for elem in main_content.find_all(["h2", "h3", "h4", "p", "ul", "ol"]):
        if elem.name in ["h2", "h3", "h4"]:
            heading = elem.get_text(strip=True)
            if heading:
                content.append(f"\n## {heading}\n")

        elif elem.name == "p":
            para = elem.get_text(strip=True)
            if para:
                content.append(para)

        elif elem.name in ["ul", "ol"]:
            for li in elem.find_all("li", recursive=False):
                li_text = li.get_text(strip=True)
                if li_text:
                    content.append(f"- {li_text}")

    final_text = "\n".join(content).strip()
    print(f"✅ Final cleaned text length: {len(final_text)}")
    return final_text

def save_page(title, text):
    os.makedirs(SAVE_DIR, exist_ok=True)
    safe_title = title.replace("/", "_").replace(" ", "_")
    filepath = os.path.join(SAVE_DIR, f"{safe_title}.json")
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"title": title, "content": text}, f, ensure_ascii=False, indent=2)

def clear_scraped_data():
    """
    Delete all previously scraped files before a new scraping session.
    """
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)  # Deletes the entire directory and files
    os.makedirs(SAVE_DIR, exist_ok=True)  # Recreate the folder

def scrape_wikipedia():
    clear_scraped_data()  # Clear previous data

    main_html = fetch_page(BASE_URL + ROOT_PAGE)
    if not main_html:
        print("Could not fetch main page.")
        return

    links = parse_links(main_html, max_links=150)  # Scrape around 100–150 related pages

    print(f"Found {len(links)} sub-pages to scrape.")
    all_pages = [BASE_URL + ROOT_PAGE] + links  # Include root page too

    for idx, page_url in enumerate(all_pages):
        print(f"[{idx+1}/{len(all_pages)}] Scraping {page_url}")
        html = fetch_page(page_url)
        if not html:
            continue

        title = page_url.split("/wiki/")[-1].replace("_", " ")
        text = clean_text(html)

        if len(text) < 100:
            print(f"Skipping {title} (too short).")
            continue

        if "may refer to:" in text or "disambiguation" in title.lower(): # Skip disambiguation pages
            print(f"Skipping {title} (disambiguation/stub).")
            continue

        save_page(title, text)
        time.sleep(1)  # Be polite: don't overload Wikipedia

    print(f"Scraping finished. Saved {len(os.listdir(SAVE_DIR))} pages.")

if __name__ == "__main__":
    scrape_wikipedia()
