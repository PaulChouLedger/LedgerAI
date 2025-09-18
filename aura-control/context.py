import os
import faiss
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# === Thresholds and settings ===
CHUNK_SCORE_THRESHOLD = 1.2  # Lower is better (L2 distance)
MAX_CONTEXT_CHUNKS = 3
INDEX_PATH = "data/embeddings/index.faiss"
DOCS_PATH = "data/embeddings/doc_chunks.npy"

# === Load sentence transformer model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === HTML Cleanup ===
def strip_html(html_text):
    return BeautifulSoup(html_text, "html.parser").get_text()

# === Build FAISS index ===
def build_faiss_index(parsed_dir="data/parsed", force_rebuild=False):
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH) and not force_rebuild:
        print("[Aura/context] ✅ FAISS index already exists — skipping rebuild.")
        return

    print("[Aura/context] 🧠 Building FAISS context index...")
    doc_chunks = []

    for filename in os.listdir(parsed_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(parsed_dir, filename), "r", encoding="utf-8") as f:
                text = f.read()
                chunks = text.split("\n\n")
                cleaned_chunks = [strip_html(chunk.strip()) for chunk in chunks if chunk.strip()]
                doc_chunks.extend(cleaned_chunks)

    if not doc_chunks:
        print("[Aura/context] ⚠️ No valid document chunks found — skipping FAISS build.")
        return

    print(f"[Aura/context] 🧠 Encoding {len(doc_chunks)} chunks...")
    embeddings = model.encode(doc_chunks, show_progress_bar=False)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    np.save(DOCS_PATH, np.array(doc_chunks))

    print(f"[Aura/context] ✅ Built FAISS index with {len(doc_chunks)} entries.")

# === Retrieve top relevant context chunks ===
def retrieve_relevant_context(query):
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
        print("[Aura/context] ⚠️ No FAISS index found. Context search disabled.")
        return ""

    query_embedding = model.encode([query]).astype("float32")
    index = faiss.read_index(INDEX_PATH)
    doc_chunks = np.load(DOCS_PATH, allow_pickle=True)

    D, I = index.search(query_embedding, MAX_CONTEXT_CHUNKS)

    context = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1 or score > CHUNK_SCORE_THRESHOLD:
            print(f"[Aura/context] ❌ Skipping chunk (score={score:.2f}): {doc_chunks[idx][:80]}...")
            continue
        print(f"[Aura/context] ✅ Including chunk (score={score:.2f})")
        context.append(strip_html(doc_chunks[idx]))

    return "\n\n".join(context)

def retrieve_context(query: str) -> list:
    return retrieve_relevant_context(query)
