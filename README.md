# 📰 Fake News Detection System with Knowledge Base (FAISS)

A hybrid machine learning system for detecting fake news using **semantic understanding**, **stylistic analysis**, and **knowledge base similarity**. The system combines multiple models through an **ensemble (meta-learning) approach** to improve prediction accuracy and reliability.

> 🔗 **GitHub Repository:** [github.com/ketangarg15/Fake-news-detection](https://github.com/ketangarg15/Fake-news-detection)

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🔍 Semantic Analysis | Transformer-based embeddings via DistilBERT |
| ✍️ Stylistic Analysis | Linguistic feature extraction using scikit-learn |
| 📚 Knowledge Base Validation | FAISS similarity search over real news articles |
| 🧠 Ensemble Meta-Learning | Logistic Regression combining all model signals |
| 🌐 Web Interface | Real-time news verification via Flask |
| 🔄 KB Update Pipeline | Continuously update the knowledge base with new articles |

---

## 🧠 Model Architecture

```
Input News Article
        │
        ▼
 DistilBERT Embeddings
        │
   ┌────┴─────┐──────────────┐
   ▼          ▼              ▼
Text Model  Style Model   FAISS KB
(Random     (Random       Similarity
 Forest)     Forest)      Search
   │          │              │
   └────┬─────┘──────────────┘
        ▼
  Meta Model
  (Logistic Regression)
        │
        ▼
 Final Prediction
 (FAKE / REAL)
```

---

## ⚙️ Technologies Used

- **Python 3.8+**
- **Scikit-learn** — Random Forest, Logistic Regression
- **Sentence Transformers** — DistilBERT embeddings (`all-MiniLM-L6-v2`)
- **FAISS** — Facebook AI Similarity Search for KB lookup
- **Flask** — Lightweight web application framework
- **NumPy & Pandas** — Data manipulation and preprocessing
- **Joblib** — Model serialization

---

## 📂 Project Structure

```
fake-news-detection/
│
├── app/
│   └── app.py                  # Flask web application
│
├── data/
│   └── fake_news.csv           # Dataset (required — see below)
│
├── models/                     # Saved trained models (.pkl / .index)
│   ├── text_model.pkl
│   ├── style_model.pkl
│   ├── meta_model.pkl
│   └── kb_faiss.index
│
├── train_models.py             # Train base models (Text + Style)
├── build_kb_faiss.py           # Build FAISS knowledge base
├── train_meta_model.py         # Train the ensemble meta model
├── evaluate_models.py          # Evaluate performance metrics
├── update_kb.py                # Update knowledge base with new data
├── weight_check.py             # Inspect feature importance & weights
├── news_articles_into_kb.py    # Prepare and ingest KB articles
│
└── requirements.txt            # Python dependencies
```

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ketangarg15/Fake-news-detection.git
cd Fake-news-detection
```

### 2. Create and Activate a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

Place your dataset at:

```
data/fake_news.csv
```

### Expected Columns

| Column | Type | Description |
|---|---|---|
| `title` | string | News headline |
| `text` | string | Full news article body |
| `label` | int | `0` = Real, `1` = Fake |

> **Recommended Dataset:** [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) or [LIAR Dataset](https://paperswithcode.com/dataset/liar) available on Kaggle.

---

## 📈 Training Pipeline

Run the following scripts **in order**:

### Step 1 — Train Base Models

```bash
python train_models.py
```

Trains the Text Model (semantic) and Style Model (linguistic) using Random Forest classifiers. Saves models to `models/`.

### Step 2 — Build the Knowledge Base

```bash
python build_kb_faiss.py
```

Encodes real news articles using DistilBERT and indexes them with FAISS for fast similarity search.

### Step 3 — Train the Meta Model

```bash
python train_meta_model.py
```

Trains a Logistic Regression meta-model that combines signals from the Text Model, Style Model, and FAISS KB similarity scores.

---

## 🌐 Run the Web Application

```bash
cd app
python app.py
```

Open your browser and navigate to:

```
http://localhost:5000
```

Paste any news article or headline to get an instant prediction.

---

## 📊 Evaluation

```bash
python evaluate_models.py
```

### Output Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted fakes, how many were actually fake |
| **Recall** | Of actual fakes, how many were correctly identified |
| **F1 Score** | Harmonic mean of Precision and Recall |

Results are printed to the console and can optionally be saved as a report.

---

## 🔄 Update Knowledge Base

```bash
python update_kb.py
```

Adds new real-world news articles to the FAISS index to improve prediction accuracy over time. Useful for keeping the system current with recent events.

---

## 🧪 Inspect Model Weights

```bash
python weight_check.py
```

Displays:
- Feature importance scores from the Random Forest models
- Logistic Regression coefficients from the Meta Model
- Top contributing features for fake vs. real classification

---

## 💡 How It Works

1. **Embedding Generation** — Input text is converted into dense vector embeddings using DistilBERT (`sentence-transformers`).

2. **Text Model** — A Random Forest classifier predicts based on the semantic meaning of the embeddings.

3. **Style Model** — A second Random Forest analyzes stylistic and linguistic features such as punctuation density, capitalization patterns, sentence length, and vocabulary complexity.

4. **FAISS KB Search** — The article embedding is compared against a FAISS index of known real news articles. A high cosine similarity score indicates proximity to real news.

5. **Meta Model** — A Logistic Regression model combines the three signals (text prediction, style prediction, KB similarity score) into a final binary prediction: **FAKE** or **REAL**.

---

## 🎯 Key Advantages

- **Hybrid Architecture** — Combines deep learning embeddings with traditional ML classifiers.
- **Knowledge-Grounded** — FAISS similarity search anchors predictions to real-world verified news.
- **Scalable** — FAISS supports millions of vectors with sub-millisecond lookup time.
- **Modular** — Each component (text, style, KB) can be independently updated or replaced.
- **More Robust** — Ensemble approach significantly reduces false positives vs. single-model systems.

---

## ⚠️ Limitations

- Performance depends heavily on **dataset quality and balance**.
- Knowledge base coverage limits detection of **niche or domain-specific** fake news.
- May struggle with **adversarially crafted** articles that mimic real news writing style.
- No real-time news ingestion — KB must be manually updated via `update_kb.py`.

---

## 🚀 Future Improvements

- [ ] Fine-tune transformer models (e.g., RoBERTa, DeBERTa) on domain-specific data
- [ ] Integrate live news APIs (NewsAPI, GDELT) for real-time KB updates
- [ ] Add explainability layer using **LIME** or **SHAP**
- [ ] Multi-class labeling (e.g., Satire / Misleading / False / True)
- [ ] Dockerize the application for easy deployment
- [ ] Add REST API endpoints for programmatic access

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Ketan Garg**
- GitHub: [@ketangarg15](https://github.com/ketangarg15)
- Repository: [Fake-news-detection](https://github.com/ketangarg15/Fake-news-detection)

---

## 🙏 Acknowledgements

- [Hugging Face Sentence Transformers](https://www.sbert.net/) for DistilBERT embeddings
- [Facebook AI Research](https://github.com/facebookresearch/faiss) for FAISS
- [Scikit-learn](https://scikit-learn.org/) for ML utilities
- Open-source fake news datasets from Kaggle and academic research communities
