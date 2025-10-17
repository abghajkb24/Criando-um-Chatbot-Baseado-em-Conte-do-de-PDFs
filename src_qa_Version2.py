"""
Pipeline de recuperação + geração de resposta.

- retrieve_candidates(query, top_k)
- generate_answer(query, candidates) -> tenta OpenAI, senão HuggingFace, senão resposta extractiva
"""
import os
import joblib
import numpy as np
from typing import List, Dict
from .embeddings import EmbeddingModel
from .vectorstore import load_faiss_index, query_faiss
from transformers import pipeline
import openai

# carregamento de artefatos
EMBED_MODEL = None
INDEX = None
METADATA = None

def init(emb_model_name: str = None, index_path: str = "models/index.faiss"):
    global EMBED_MODEL, INDEX, METADATA
    EMBED_MODEL = EmbeddingModel(model_name=emb_model_name) if emb_model_name else EmbeddingModel()
    INDEX, METADATA = load_faiss_index(index_path)

def retrieve_candidates(query: str, top_k: int = 5) -> List[Dict]:
    emb = EMBED_MODEL.embed([query])
    D, I = query_faiss(INDEX, emb, top_k=top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        meta = METADATA[idx]
        results.append({
            "score": float(score),
            "text": meta["text"],
            "file_name": meta.get("file_name"),
            "chunk_id": meta.get("chunk_id")
        })
    return results

def _generate_with_openai(prompt: str) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai.api_key = key
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        n=1,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

def _generate_with_hf(prompt: str, model_name: str = "google/flan-t5-small") -> str:
    gen = pipeline("text2text-generation", model=model_name, device=-1)
    out = gen(prompt, max_length=256, do_sample=False)
    return out[0]["generated_text"].strip()

def generate_answer(query: str, candidates: List[Dict]) -> Dict:
    """
    Gera uma resposta usando o contexto dos candidatos.
    Retorna dict com: answer (str), sources (list)
    """
    context = "\n\n---\n\n".join([f"Fonte: {c['file_name']}\nTrecho: {c['text']}" for c in candidates])
    prompt = f"Você é um assistente que responde perguntas usando apenas as informações fornecidas abaixo.\n\nContexto:\n{context}\n\nPergunta: {query}\n\nResponda objetivamente e cite as fontes (file_name) quando possível."
    # prefer OpenAI se chave existir
    try:
        if os.environ.get("OPENAI_API_KEY"):
            answer = _generate_with_openai(prompt)
        else:
            # tentar HuggingFace local
            answer = _generate_with_hf(prompt)
    except Exception:
        # fallback: retornamos os trechos mais relevantes concatenados (modo extractivo)
        snippets = "\n\n".join([f"- ({c['file_name']}) {c['text'][:500]}..." for c in candidates])
        answer = f"Resposta (extraída dos trechos mais relevantes):\n{snippets}"
    sources = list({c["file_name"] for c in candidates})
    return {"answer": answer, "sources": sources, "candidates": candidates}