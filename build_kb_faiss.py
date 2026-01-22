# ===============================================================
# BUILD KNOWLEDGE BASE (REAL NEWS EMBEDDINGS)
# ===============================================================

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load your trained embedding model
from joblib import load
embedder = load("models/embedding_model.pkl")

# Use only real news (label == 0)
news_df = pd.read_csv("data/fake_news.csv")
news_df['content'] = news_df['title'].fillna('') + " " + news_df['text'].fillna('')
if news_df['label'].dtype == 'O':
    news_df['label'] = news_df['label'].map({'FAKE': 1, 'REAL': 0, 'fake': 1, 'real': 0})

real_news = news_df[news_df['label'] == 0]['content'].tolist()

print(f"Encoding {len(real_news)} verified articles for knowledge base...")
kb_embeddings = embedder.encode(real_news, show_progress_bar=True)

# Build FAISS index
d = kb_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(kb_embeddings))

# Save index and reference texts
faiss.write_index(index, "models/kb_faiss.index")
np.save("models/kb_texts.npy", np.array(real_news))

print("✅ Knowledge base built successfully!")
