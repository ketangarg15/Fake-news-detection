# ===============================================================
# UPDATE KNOWLEDGE BASE WITH NEW REAL NEWS ARTICLES
# ===============================================================

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from joblib import load

# Load the same embedding model you used originally
embedder = load("models/embedding_model.pkl")

# New real articles to add
new_articles = [
    """LJP implodes as uncle Paras isolates Chirag Paswan; Nitish Kumar's JD(U) says 'you reap what you sow'
    In a surprise turn of events, Chirag Paswan, the president of Lok Janshakti Party (LJP) founded by his father late Ram Vilas Paswan, was on Monday unceremoniusly unseated as the leader of the party in Lok Sabha. 
    Chirag, who took over the mantle of the party in 2020 after the death of Ram Vilas Paswan, stood completely isolated at the top within his party.""",

    """‘Cameras failed to capture faces’: Out with boyfriend, PG student abducted, gang-raped in Coimbatore; 3 arrested after encounter 
    A 20-year-old student was abducted and gang-raped in Coimbatore after her car was attacked by three men. 
    The assailants assaulted her boyfriend before taking her to a secluded area. 
    Police apprehended three suspects from Madurai district after a manhunt. 
    The survivor is receiving medical attention, and investigations are ongoing."""
]

# ---------------------------------------------------------------
# Step 1: Load existing KB index and reference texts
# ---------------------------------------------------------------
index = faiss.read_index("models/kb_faiss.index")
kb_texts = np.load("models/kb_texts.npy", allow_pickle=True)

print(f"🧠 Loaded existing knowledge base with {len(kb_texts)} articles.")

# ---------------------------------------------------------------
# Step 2: Encode the new articles using the same embedder
# ---------------------------------------------------------------
new_embeddings = embedder.encode(new_articles, show_progress_bar=True)

# ---------------------------------------------------------------
# Step 3: Add new embeddings to FAISS index
# ---------------------------------------------------------------
index.add(np.array(new_embeddings))

# ---------------------------------------------------------------
# Step 4: Update stored text references
# ---------------------------------------------------------------
updated_texts = np.concatenate([kb_texts, np.array(new_articles)])
np.save("models/kb_texts.npy", updated_texts)

# ---------------------------------------------------------------
# Step 5: Save the updated FAISS index
# ---------------------------------------------------------------
faiss.write_index(index, "models/kb_faiss.index")

print("✅ Knowledge base updated successfully with 2 new articles.")
print(f"🔹 Total articles in KB now: {len(updated_texts)}")
