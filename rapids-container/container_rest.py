from flask import Flask, request, jsonify
import numpy as np
import os
import pickle
from cuml.neighbors import NearestNeighbors

# === Constants ===
CHUNK_SCORE_THRESHOLD = 0.4  # ✅ Lower is better (cosine distance)
MAX_CONTEXT_CHUNKS = 3
INDEX_PATH = "/shared/vector_index/index.pkl"
DOCS_PATH = "/shared/vector_index/doc_chunks.npy"

# === Load doc chunks and cuML index ===
with open(INDEX_PATH, "rb") as f:
    nn_model: NearestNeighbors = pickle.load(f)

doc_chunks = np.load(DOCS_PATH, allow_pickle=True)

# === Flask App ===
app = Flask(__name__)

@app.route("/context", methods=["POST"])
def get_context():
    data = request.get_json()
    if not data or "embedding" not in data:
        return jsonify({"error": "Missing embedding"}), 400

    query_embedding = np.array(data["embedding"], dtype=np.float32).reshape(1, -1)

    try:
        distances, indices = nn_model.kneighbors(query_embedding)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if score > CHUNK_SCORE_THRESHOLD or idx >= len(doc_chunks):
            continue
        results.append({
            "chunk": str(doc_chunks[idx]),
            "score": float(score)
        })

    return jsonify({"context": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)
