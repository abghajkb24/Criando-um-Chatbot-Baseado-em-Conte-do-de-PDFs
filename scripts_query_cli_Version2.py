#!/usr/bin/env python3
"""
CLI simples para perguntar ao índice.
"""
import argparse
from src.qa import init, retrieve_candidates, generate_answer

def main(index_path, model_name=None, top_k=5):
    init(emb_model_name=model_name, index_path=index_path)
    while True:
        q = input("\nPergunta (ou 'sair' para terminar): ").strip()
        if q.lower() in ("sair", "exit", "quit"):
            break
        cands = retrieve_candidates(q, top_k=top_k)
        res = generate_answer(q, cands)
        print("\nResposta:\n", res["answer"])
        print("\nFontes:", res["sources"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", default="models/index.faiss")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    main(args.index_path, args.model_name, args.top_k)