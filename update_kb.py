# ===============================================================
# update_kb_toi.py — Scrape TOI headlines & update FAISS KB
# ===============================================================

import time
import re
import numpy as np
import pandas as pd
import faiss
import joblib
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------
# Load models and existing KB
# ---------------------------------------------------------------
print("Loading models and existing KB...")
embedder = joblib.load("models/embedding_model.pkl")

try:
    kb_index = faiss.read_index("models/kb_faiss.index")
    kb_texts = np.load("models/kb_texts.npy", allow_pickle=True).tolist()
    print(f"✅ Loaded existing KB with {len(kb_texts)} entries.")
except:
    print("⚠️ No existing KB found. Creating a new one...")
    kb_index = faiss.IndexFlatL2(768)
    kb_texts = []

# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------
def clean_text(text):
    """Clean up text by removing extra whitespace and unwanted characters."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------------------------------------------------------
# Selenium Web Scraping — Times of India
# ---------------------------------------------------------------
def scrape_toi_articles(max_scrolls=10):
    print("🌐 Launching headless browser for TOI scraping...")

    chrome_options = Options()
    chrome_options.add_argument("--headless")  
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    url = "https://timesofindia.indiatimes.com/"
    driver.get(url)
    print(" Page loaded:", driver.title)

    # Scroll to load more headlines
    scroll_pause = 2
    for i in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause)
        print(f"🔄 Scrolled {i+1}/{max_scrolls}")

    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, "lxml")
    driver.quit()
    print(" Page parsed and browser closed")

    # Extract article data
    articles = []
    for li in soup.find_all("li", class_="BxDma"):
        a_tag = li.find("a", href=True)
        headline_tag = li.find("div", class_="CRKrj")
        desc_tag = li.find("p", class_="W4Hjm")

        if a_tag and headline_tag:
            href = a_tag["href"]
            if not href.startswith("http"):
                href = "https://timesofindia.indiatimes.com" + href
            text = f"{headline_tag.get_text(strip=True)}. {desc_tag.get_text(strip=True) if desc_tag else ''}"
            if len(text) > 50:
                articles.append(clean_text(text))

    print(f" Extracted {len(articles)} articles from TOI.")
    return articles

# ---------------------------------------------------------------
# Main KB Update Pipeline
# ---------------------------------------------------------------
def update_kb_from_toi():
    all_articles = scrape_toi_articles(max_scrolls=12)  # ~100+ articles

    if not all_articles:
        print("⚠️ No new articles found, exiting.")
        return

    print("\n Embedding new TOI articles and updating FAISS index...")
    new_embeddings = embedder.encode(all_articles, show_progress_bar=True)
    new_embeddings = np.array(new_embeddings).astype('float32')

    kb_index.add(new_embeddings)
    kb_texts.extend(all_articles)

    # Save updated KB
    faiss.write_index(kb_index, "models/kb_faiss.index")
    np.save("models/kb_texts.npy", np.array(kb_texts, dtype=object))

    print(f" KB updated successfully. Total entries now: {len(kb_texts)}")

# ---------------------------------------------------------------
# Run Script
# ---------------------------------------------------------------
if __name__ == "__main__":
    update_kb_from_toi()
