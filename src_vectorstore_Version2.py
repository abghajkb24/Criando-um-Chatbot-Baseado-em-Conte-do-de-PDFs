"""
Criação e consulta de um índice FAISS simples.
Salva e carrega índice + metadados (metadata: lista de dicts com doc info).
"""
import faiss
import numpy as np
import joblib
from pathlib import Path

def build_faiss_index(embeddings: np.ndarray, metadata: list, index_path: str, metric: str = "cosine"):
    # FAISS expects float32
    emb = embeddings.astype("float32")
    d = emb.shape[1]
    # normalize for cosine similarity (inner product after normalization)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    joblib.dump(metadata, Path(index_path).with_suffix(".metadata.joblib"))
    return index

def load_faiss_index(index_path: str):
    if not Path(index_path).exists():
        raise FileNotFoundError(index_path)
    index = faiss.read_index(index_path)
    metadata = joblib.load(Path(index_path).with_suffix(".metadata.joblib"))
    return index, metadata

def query_faiss(index, embeddings, top_k: int = 5):
    # embeddings shape (1, d) float32 and normalized
    emb = embeddings.astype("float32")
    faiss.normalize_L2(emb)
    D, I = index.search(emb, top_k)
    return D, I