"""
FastAPI app para upload de PDFs, construção de índice e consulta.

Endpoints:
- POST /upload-pdfs (files) -> salva PDFs em data/pdfs
- POST /build-index -> (re)constrói índice
- POST /query -> {"query": "..."} -> retorna resposta e fontes
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path
import shutil
import joblib
from src.qa import init, retrieve_candidates, generate_answer
import os

DATA_DIR = Path("data/pdfs")
INDEX_PATH = "models/index.faiss"

app = FastAPI(title="PDF QA Chatbot")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 4

@app.post("/upload-pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        dest = DATA_DIR / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(f.filename)
    return {"saved": saved, "count": len(saved)}

@app.post("/build-index")
def build_index():
    # chama script de build de forma simples (import local)
    from scripts.build_index import main as build_main
    build_main(str(DATA_DIR), INDEX_PATH, meta_path="models/metadata.joblib")
    return {"status": "index_built", "path": INDEX_PATH}

@app.post("/query")
def query(req: QueryRequest):
    if not Path(INDEX_PATH).exists():
        raise HTTPException(status_code=503, detail="Índice não encontrado. Execute /build-index primeiro.")
    # inicializa (carrega índice e modelo de embeddings)
    init(index_path=INDEX_PATH)
    cands = retrieve_candidates(req.query, top_k=req.top_k)
    res = generate_answer(req.query, cands)
    return res