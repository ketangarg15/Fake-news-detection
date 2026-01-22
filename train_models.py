# ===============================================================
# TRAIN MODELS (TEXT MODEL + STYLE MODEL)
# ===============================================================

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib
import re

# -------------------------------
# 1. Load dataset
# -------------------------------
# Dataset columns expected: 'title', 'text', 'label'
news_df = pd.read_csv("data/fake_news.csv")

# Combine title and text for better context
news_df['content'] = news_df['title'].fillna('') + " " + news_df['text'].fillna('')

# Convert label if it's in text form
if news_df['label'].dtype == 'O':
    news_df['label'] = news_df['label'].map({'FAKE': 1, 'REAL': 0, 'fake': 1, 'real': 0})

# -------------------------------
# 2. Generate Text Embeddings
# -------------------------------
print("Generating text embeddings using DistilBERT...")
embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')
X_text = embedder.encode(news_df['content'].tolist(), show_progress_bar=True)

# -------------------------------
# 3. Train Text Model
# -------------------------------
print("Training text-based classifier...")
text_clf = RandomForestClassifier(n_estimators=200, random_state=42)
text_clf.fit(X_text, news_df['label'])

# -------------------------------
# 4. Extract Style Features
# -------------------------------
def extract_style_features(text):
    words = text.split()
    exclamations = text.count('!')
    capitals = sum(1 for c in text if c.isupper())
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    return [len(words), exclamations, capitals, avg_word_len]

print("Extracting style features...")
X_style = np.array([extract_style_features(t) for t in news_df['content']])
y = news_df['label']

# -------------------------------
# 5. Train Style Model
# -------------------------------
print("Training style-based classifier...")
style_clf = RandomForestClassifier(n_estimators=100, random_state=42)
style_clf.fit(X_style, y)

# -------------------------------
# 6. Save Models
# -------------------------------
print("Saving models...")
joblib.dump(text_clf, "models/text_model.pkl")
joblib.dump(style_clf, "models/style_model.pkl")
joblib.dump(embedder, "models/embedding_model.pkl")

print("✅ Models trained and saved successfully!")
