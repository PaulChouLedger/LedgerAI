import os
import pickle
import numpy as np
from cuml.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

# === Config ===
DOC_DIR = "data/parsed"
EMBEDDING_DIR = "shared/vector_index"
INDEX_PATH = os.path.join(EMBEDDING_DIR, "index.pkl")
DOCS_PATH = os.path.join(EMBEDDING_DIR, "doc_chunks.npy")
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CHUNKS = 3

# === Utilities ===
def strip_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

# === Ensure output directory ===
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# === Load embedding model ===
print("🔍 Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# === Load and preprocess documents ===
print("📄 Parsing documents...")
doc_chunks = []
for file in os.listdir(DOC_DIR):
    if file.endswith(".txt"):
        with open(os.path.join(DOC_DIR, file), "r", encoding="utf-8") as f:
            text = f.read()
            chunks = text.split("\n\n")
            doc_chunks.extend([strip_html(c.strip()) for c in chunks if c.strip()])

if not doc_chunks:
    print("⚠️ No valid chunks found.")
    exit(1)

# === Encode with normalization (required for cosine) ===
print(f"🧠 Encoding {len(doc_chunks)} chunks...")
embeddings = model.encode(doc_chunks, normalize_embeddings=True)

# === cuML NearestNeighbors index ===
print("🔧 Building cuML index...")
nn_model = NearestNeighbors(n_neighbors=MAX_CHUNKS, metric="cosine")
nn_model.fit(embeddings)

# === Save index and chunks ===
print("💾 Saving index and document chunks...")
with open(INDEX_PATH, "wb") as f:
    pickle.dump(nn_model, f)

np.save(DOCS_PATH, np.array(doc_chunks))

print(f"✅ cuML index saved to {INDEX_PATH} with {len(doc_chunks)} chunks.")
