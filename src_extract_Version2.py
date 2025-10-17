"""
Extrai texto de PDFs e salva em CSV simples de chunks.

Funções principais:
- extract_text_from_pdf(path) -> str
- chunk_text(text, chunk_size=500, overlap=50) -> List[str]
- build_corpus_from_dir(pdf_dir) -> DataFrame with columns: doc_id, file_name, chunk_id, text
"""
from pathlib import Path
import pdfplumber
import pandas as pd
import uuid
from typing import List

def extract_text_from_pdf(path: str) -> str:
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text_parts.append(txt)
    return "\n".join(text_parts)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def build_corpus_from_dir(pdf_dir: str, out_csv: str = None) -> pd.DataFrame:
    pdf_dir = Path(pdf_dir)
    rows = []
    for path in pdf_dir.glob("**/*.pdf"):
        doc_id = str(uuid.uuid4())
        text = extract_text_from_pdf(str(path))
        chunks = chunk_text(text)
        for idx, c in enumerate(chunks):
            rows.append({
                "doc_id": doc_id,
                "file_name": path.name,
                "chunk_id": f"{doc_id}_{idx}",
                "text": c
            })
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df