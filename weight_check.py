# ===============================================================
# INSPECT TRAINED MODEL WEIGHTS / FEATURE IMPORTANCES
# ===============================================================

import joblib
import numpy as np
import pandas as pd

# -------------------------------
# 1. Load models
# -------------------------------
text_clf = joblib.load("models/text_model.pkl")
style_clf = joblib.load("models/style_model.pkl")
meta_model = joblib.load("models/meta_model.pkl")

print("\n================ TEXT MODEL (RandomForest) =================")
try:
    importances = text_clf.feature_importances_
    print(f"Number of features: {len(importances)}")
    # Show top 10 features
    top_idx = np.argsort(importances)[-10:][::-1]
    print("Top 10 Feature Importances (Text Model):")
    for i in top_idx:
        print(f"  Feature {i}: {importances[i]:.6f}")
except AttributeError:
    print("⚠️ Text model does not have feature_importances_ (check model type).")

print("\n================ STYLE MODEL (RandomForest) =================")
try:
    style_features = ["Word Count", "Exclamation Marks", "Capital Letters", "Avg Word Length"]
    style_importances = style_clf.feature_importances_
    print("Feature Importances for Style Model:")
    for name, imp in zip(style_features, style_importances):
        print(f"  {name:<20}: {imp:.6f}")
except AttributeError:
    print("⚠️ Style model does not have feature_importances_ (check model type).")

print("\n================ META MODEL (LogisticRegression) =================")
try:
    feature_names = ["Text Model Score", "Style Model Score", "KB Similarity"]
    weights = meta_model.coef_[0]
    intercept = meta_model.intercept_[0]
    print("Meta Model Weights:")
    for name, w in zip(feature_names, weights):
        print(f"  {name:<20}: {w:.6f}")
    print(f"Intercept               : {intercept:.6f}")
except AttributeError:
    print("⚠️ Meta model is not LogisticRegression or lacks coef_ attribute.")

print("\n✅ All model weights and importances displayed successfully!")
