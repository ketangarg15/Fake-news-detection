# ==============================================================
# Evaluate Text, Style, and Ensemble Models on New Dataset
# ==============================================================

import pandas as pd
import numpy as np
import joblib
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Load models and KB
# -------------------------------
print("Loading models...")
embedder = joblib.load('models/embedding_model.pkl')
text_clf = joblib.load('models/text_model.pkl')
style_clf = joblib.load('models/style_model.pkl')

try:
    kb_index = faiss.read_index('models/kb_faiss.index')
    print("✅ Knowledge base loaded successfully.")
except Exception as e:
    kb_index = None
    print(f"⚠️ Could not load KB ({e}) — skipping KB similarity")

# -------------------------------
# Helper functions
# -------------------------------
def extract_style_features(text):
    """Extract simple text style metrics."""
    words = text.split()
    exclamations = text.count('!')
    capitals = sum(1 for c in text if c.isupper())
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    return [len(words), exclamations, capitals, avg_word_len]

def compute_kb_similarity(query_embedding, faiss_index, top_k=1):
    """Compute KB similarity score in range [0,1]."""
    try:
        distances, indices = faiss_index.search(query_embedding.astype('float32'), top_k)
        similarities = 1 / (1 + distances)
        return float(np.clip(similarities.max(axis=1)[0], 0, 1))
    except Exception:
        return 0.0

# -------------------------------
# Load new dataset
# -------------------------------
try:
    df = pd.read_csv('data/test_dataset.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('data/test_dataset.csv', encoding='latin1', on_bad_lines='skip')

print(f"✅ Loaded dataset with {len(df)} rows before cleaning")

# --- Basic sanity check for columns ---
expected_cols = {'title', 'text', 'label'}
missing = expected_cols - set(df.columns.str.lower())
if missing:
    raise ValueError(f"❌ Missing columns in CSV: {missing}. Columns found: {df.columns.tolist()}")

# Normalize column names (handle Title/Text/Label variations)
df.columns = df.columns.str.lower()

# Drop rows missing essential fields
df = df.dropna(subset=['text', 'label'])

# Convert labels safely to int (ignore text noise)
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

print(f"✅ Cleaned dataset: {len(df)} valid rows remain")

texts = df['text'].astype(str).tolist()
labels = df['label'].astype(int).tolist()

# -------------------------------
# Generate embeddings and features
# -------------------------------
print("Generating embeddings...")
text_embeddings = embedder.encode(texts, show_progress_bar=True)

print("Extracting style features...")
style_features = np.array([extract_style_features(t) for t in texts])

# -------------------------------
# Model predictions
# -------------------------------
print("\nPredicting using text model...")
text_probs = [p[1] for p in text_clf.predict_proba(text_embeddings)]

print("Predicting using style model...")
style_probs = [p[1] for p in style_clf.predict_proba(style_features)]

# KB similarity
print("Computing KB similarity...")
kb_sims = []
if kb_index is not None:
    for emb in text_embeddings:
        kb_sims.append(compute_kb_similarity(np.array([emb]), kb_index))
else:
    kb_sims = [0.0] * len(text_embeddings)

# -------------------------------
# Ensemble (Weighted Average)
# -------------------------------
# weights tuned manually from your final app.py
final_scores = [
    (0.45 * text_probs[i]) + (0.45 * style_probs[i]) + (0.10 * (1 - kb_sims[i]))
    for i in range(len(texts))
]

text_preds = [1 if p > 0.5 else 0 for p in text_probs]
style_preds = [1 if p > 0.5 else 0 for p in style_probs]
ensemble_preds = [1 if p > 0.5 else 0 for p in final_scores]

# -------------------------------
# Evaluate each model
# -------------------------------
print("\n==================== RESULTS ====================")
print(f"Text Model Accuracy:     {accuracy_score(labels, text_preds):.3f}")
print(f"Style Model Accuracy:    {accuracy_score(labels, style_preds):.3f}")
print(f"Ensemble Model Accuracy: {accuracy_score(labels, ensemble_preds):.3f}")

print("\n--- Classification Report (Ensemble) ---")
print(classification_report(labels, ensemble_preds, digits=3))
