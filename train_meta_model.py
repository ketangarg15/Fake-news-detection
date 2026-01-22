# ===============================================================
# TRAIN META MODEL (ENSEMBLE LEARNER) + EVALUATION
# ===============================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import faiss
from sentence_transformers import SentenceTransformer

# -------------------------------
# 1. Load trained models
# -------------------------------
print("Loading models...")
text_clf = joblib.load("models/text_model.pkl")
style_clf = joblib.load("models/style_model.pkl")
embedder = joblib.load("models/embedding_model.pkl")

# -------------------------------
# 2. Load dataset
# -------------------------------
print("Loading dataset...")
news_df = pd.read_csv("data/fake_news.csv")
news_df['content'] = news_df['title'].fillna('') + " " + news_df['text'].fillna('')

# Map labels to 0/1
if news_df['label'].dtype == 'O':
    news_df['label'] = news_df['label'].map({'FAKE': 1, 'REAL': 0, 'fake': 1, 'real': 0})
y = news_df['label']

# -------------------------------
# 3. Generate embeddings
# -------------------------------
print("Generating embeddings for meta-model...")
X_embed = embedder.encode(news_df['content'].tolist(), show_progress_bar=True)

# -------------------------------
# 4. Compute KB similarity using FAISS
# -------------------------------
print("Loading FAISS KB...")
index = faiss.read_index("models/kb_faiss.index")
kb_texts = np.load("models/kb_texts.npy", allow_pickle=True)

def compute_kb_similarity(query_embeddings, faiss_index, top_k=1):
    distances, indices = faiss_index.search(query_embeddings.astype('float32'), top_k)
    similarities = 1 / (1 + distances)
    return similarities.max(axis=1)

print("Computing KB similarity scores...")
kb_similarity = compute_kb_similarity(X_embed, index)

# -------------------------------
# 5. Compute base model predictions
# -------------------------------
print("Computing text and style predictions...")

def extract_style_features(text):
    words = text.split()
    exclamations = text.count('!')
    capitals = sum(1 for c in text if c.isupper())
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    return [len(words), exclamations, capitals, avg_word_len]

X_style = np.array([extract_style_features(t) for t in news_df['content']])

text_pred_proba = text_clf.predict_proba(X_embed)[:, 1]
style_pred_proba = style_clf.predict_proba(X_style)[:, 1]

# -------------------------------
# 6. Prepare meta features
# -------------------------------
X_meta = np.column_stack((text_pred_proba, style_pred_proba, kb_similarity))

# -------------------------------
# 7. Split into train/test sets
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_meta, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 8. Train meta-model
# -------------------------------
print("Training meta-model (Logistic Regression)...")
meta_model = LogisticRegression()
meta_model.fit(X_train, y_train)

# -------------------------------
# 9. Evaluate meta-model
# -------------------------------
print("\nEvaluating meta-model...")
y_pred = meta_model.predict(X_test)
y_proba = meta_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("✅ Evaluation Results:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc:.4f}")
print("\nConfusion Matrix:\n", cm)

# -------------------------------
# 10. Save meta-model
# -------------------------------
joblib.dump(meta_model, "models/meta_model.pkl")
print("\n✅ Meta model trained and saved successfully!")