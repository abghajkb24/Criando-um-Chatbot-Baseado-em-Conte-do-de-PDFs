#!/usr/bin/env python3
"""
Script para construir índice FAISS a partir de PDFs em um diretório.

Usage:
python scripts/build_index.py --pdf-dir data/pdfs --index-path models/index.faiss --meta-path models/metadata.joblib
"""
import argparse
from src.extract import build_corpus_from_dir
from src.embeddings import EmbeddingModel
from src.vectorstore import build_faiss_index
import joblib
import numpy as np
from pathlib import Path

def main(pdf_dir, index_path, meta_path=None, embed_model=None):
    print("Extraindo textos dos PDFs...")
    corpus = build_corpus_from_dir(pdf_dir)
    if corpus.empty:
        raise SystemExit("Nenhum PDF encontrado no diretório.")
    texts = corpus["text"].tolist()
    meta = corpus.to_dict(orient="records")
    print(f"Gerando embeddings para {len(texts)} chunks...")
    emb_model = EmbeddingModel(model_name=embed_model) if embed_model else EmbeddingModel()
    embeddings = emb_model.embed(texts).astype("float32")
    print("Construindo índice FAISS...")
    build_faiss_index(embeddings, meta, index_path)
    print("Salvando metadados...")
    if meta_path:
        Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(meta, meta_path)
    print("Índice construído em", index_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--index-path", default="models/index.faiss")
    parser.add_argument("--meta-path", default="models/metadata.joblib")
    parser.add_argument("--embed-model", default=None)
    args = parser.parse_args()
    main(args.pdf_dir, args.index_path, args.meta_path, args.embed_model)