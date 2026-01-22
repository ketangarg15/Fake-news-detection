# # ===============================================================
# # FLASK APP — USES META MODEL
# # ===============================================================

# from flask import Flask, request, jsonify, render_template
# import joblib, numpy as np, faiss
# from sentence_transformers import SentenceTransformer

# app = Flask(__name__)

# print("Loading models...")

# # -------------------------------
# # Load models (keep same paths)
# # -------------------------------
# embedder = joblib.load('../models/embedding_model.pkl')
# text_clf = joblib.load('../models/text_model.pkl')
# style_clf = joblib.load('../models/style_model.pkl')
# meta_model = joblib.load('../models/meta_model.pkl')  # ✅ new
# kb_index = faiss.read_index('../models/kb_faiss.index')
# kb_texts = np.load('../models/kb_texts.npy', allow_pickle=True)

# print("All models loaded successfully!")

# # -------------------------------
# # Helper function
# # -------------------------------
# def extract_style_features(text):
#     words = text.split()
#     exclamations = text.count('!')
#     capitals = sum(1 for c in text if c.isupper())
#     avg_word_len = np.mean([len(w) for w in words]) if words else 0
#     return [len(words), exclamations, capitals, avg_word_len]

# # -------------------------------
# # Routes
# # -------------------------------
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     text = request.form['news_text']

#     # Text model
#     text_emb = embedder.encode([text])
#     text_pred = text_clf.predict_proba(text_emb)[0][1]  # probability of fake

#     # Style model
#     style_feat = np.array(extract_style_features(text)).reshape(1, -1)
#     style_pred = style_clf.predict_proba(style_feat)[0][1]

#     # Knowledge Base similarity
#     D, I = kb_index.search(np.array(text_emb, dtype='float32'), 1)
#     kb_similarity = np.exp(-D[0][0])  # distance → similarity

#     # Meta model prediction
#     meta_features = np.array([[text_pred, style_pred, kb_similarity]])
#     final_pred = meta_model.predict_proba(meta_features)[0][1]
#     verdict = "Fake News" if final_pred > 0.5 else "Real News"

#     # -------------------------------
#     # Return JSON response
#     # -------------------------------
#     return jsonify({
#         "verdict": verdict,
#         "text_score": round(float(text_pred), 3),
#         "style_score": round(float(style_pred), 3),
#         "similarity": round(float(kb_similarity), 3),
#         "final_score": round(float(final_pred), 3)
#     })

# if __name__ == '__main__':
#     app.run(debug=True)

# ===============================================================
# FLASK APP — HYBRID FAKE NEWS DETECTOR (Weighted Ensemble Version)
# ===============================================================

from flask import Flask, request, jsonify, render_template
import joblib, numpy as np, faiss
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# -------------------------------
# Load models
# -------------------------------
print("Loading models...")
embedder = joblib.load('../models/embedding_model.pkl')
text_clf = joblib.load('../models/text_model.pkl')
style_clf = joblib.load('../models/style_model.pkl')

# Load Knowledge Base
try:
    kb_index = faiss.read_index('../models/kb_faiss.index')
    kb_texts = np.load('../models/kb_texts.npy', allow_pickle=True)
    print("✅ Knowledge base loaded successfully.")
except Exception as e:
    kb_index, kb_texts = None, None
    print(f"⚠️ Warning: Could not load knowledge base ({e})")

# -------------------------------
# Helper functions
# -------------------------------
def extract_style_features(text):
    """Extract basic writing style features."""
    words = text.split()
    exclamations = text.count('!')
    capitals = sum(1 for c in text if c.isupper())
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    return [len(words), exclamations, capitals, avg_word_len]


def compute_kb_similarity(query_embedding, faiss_index, top_k=1):
    """Returns similarity in range [0,1]. If KB missing, returns 0."""
    try:
        distances, indices = faiss_index.search(query_embedding.astype('float32'), top_k)
        similarities = 1 / (1 + distances)
        kb_sim = float(np.clip(similarities.max(axis=1)[0], 0, 1))
        return kb_sim
    except Exception:
        return 0.0  # fallback if KB is missing


# -------------------------------
# Flask routes
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news_text']

    # 1️⃣ Generate embeddings
    text_emb = embedder.encode([text])

    # 2️⃣ Get base model predictions
    text_pred = text_clf.predict_proba(text_emb)[0][1]  # probability of fake
    style_feat = np.array(extract_style_features(text)).reshape(1, -1)
    style_pred = style_clf.predict_proba(style_feat)[0][1]

    # 3️⃣ Knowledge Base similarity
    kb_similarity = compute_kb_similarity(np.array(text_emb), kb_index)

    # 4️⃣ Combine using weighted-average ensemble
    # We give text & style equal weight (45% each), and KB 10% (minor penalty for unseen items)
    # Higher (1 - kb_similarity) increases fake probability if article is far from KB.
    final_score = (0.45 * text_pred) + (0.45 * style_pred) + (0.10 * (1 - kb_similarity))

    verdict = "Fake News" if final_score > 0.5 else "Real News"

    return jsonify({
        "verdict": verdict,
        "text_score": round(float(text_pred), 3),
        "style_score": round(float(style_pred), 3),
        "kb_similarity": round(float(kb_similarity), 3),
        "final_score": round(float(final_score), 3)
    })


if __name__ == '__main__':
    app.run(debug=True)


