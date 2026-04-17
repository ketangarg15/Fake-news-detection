📰 Fake News Detection System with Knowledge Base (FAISS)

A hybrid machine learning system for detecting fake news using semantic understanding, stylistic analysis, and knowledge base similarity. The system combines multiple models through an ensemble (meta-learning) approach to improve prediction accuracy and reliability.

🚀 Features
🔍 Semantic Analysis using transformer-based embeddings
✍️ Stylistic Analysis using linguistic features
📚 Knowledge Base Validation using FAISS similarity search
🧠 Ensemble Learning (Meta Model) for final prediction
🌐 Web Interface for real-time news verification
🔄 Knowledge Base Update with new articles
🧠 Model Architecture
Input News Article
        ↓
Text Embeddings (DistilBERT)
        ↓
-----------------------------------
| Text Model (Random Forest)      |
| Style Model (Random Forest)     |
| KB Similarity (FAISS)           |
-----------------------------------
        ↓
Meta Model (Logistic Regression)
        ↓
Final Prediction (Fake / Real)
⚙️ Technologies Used
Python
Scikit-learn
Sentence Transformers (DistilBERT)
FAISS (Facebook AI Similarity Search)
Flask (Web Application)
NumPy, Pandas
📂 Project Structure
├── app/                        # Flask web app
│   └── app.py
├── data/                       # Dataset (required)
│   └── fake_news.csv
├── models/                     # Saved models
├── train_models.py             # Train base models
├── build_kb_faiss.py           # Build knowledge base
├── train_meta_model.py         # Train meta model
├── evaluate_models.py          # Evaluate performance
├── update_kb.py                # Update knowledge base
├── weight_check.py             # Inspect model weights
├── news_articles_into_kb.py    # Prepare KB data
└── requirements.txt            # Dependencies
🛠️ Installation
git clone <your-repo-link>
cd <project-folder>

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
📊 Dataset

The project requires a dataset:

data/fake_news.csv
Expected Columns:
title → News headline
text → News content
label → 0 (Real), 1 (Fake)
📈 Training Pipeline

Run the following in order:

python train_models.py
python build_kb_faiss.py
python train_meta_model.py
🌐 Run Web Application
cd app
python app.py

Open in browser:

http://localhost:5000
📊 Evaluation
python evaluate_models.py

Outputs:

Accuracy
Precision
Recall
F1 Score
🔄 Update Knowledge Base
python update_kb.py

Adds new real-world news articles to improve predictions.

🧪 Inspect Model Weights
python weight_check.py

Displays feature importance and model weights.

💡 How It Works
Text is converted into embeddings using DistilBERT
Text Model predicts based on semantic meaning
Style Model analyzes writing patterns
FAISS KB checks similarity with real news
Meta Model combines all signals for final prediction
🎯 Key Advantages
Combines deep learning + traditional ML
Improves reliability using knowledge base validation
Scalable using FAISS similarity search
More robust than single-model approaches
⚠️ Limitations
Depends on dataset quality
Knowledge base coverage affects performance
May struggle with completely new/unseen topics
🚀 Future Improvements
Fine-tune transformer models
Integrate live news APIs
Enhance knowledge base with real-time data
Add explainability (LIME/SHAP)
