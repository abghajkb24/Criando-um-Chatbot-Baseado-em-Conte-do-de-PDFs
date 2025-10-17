"""
Geração de embeddings usando SentenceTransformers.
Encapsula criação/uso do modelo de embeddings.
"""
from sentence_transformers import SentenceTransformer
import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"

class EmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        # retorna numpy array (n, dim)
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, 0)
        return embeddings