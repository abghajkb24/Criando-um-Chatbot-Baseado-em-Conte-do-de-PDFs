```markdown
# Chatbot de Perguntas sobre PDFs — Projeto

Este repositório contém um projeto completo de exemplo para construir um sistema de chatbot capaz de responder perguntas com base em documentos PDF. O fluxo inclui:

- Extração de texto de arquivos PDF.
- Chunking (divisão) do texto em fragmentos menores.
- Geração de embeddings (Sentence Transformers).
- Indexação em um vetor store (FAISS).
- Recuperação (retrieval) dos trechos mais relevantes.
- Geração de resposta (usando OpenAI se fornecido, senão um modelo HuggingFace se disponível, caso contrário resposta extractiva).
- API REST com FastAPI para subir PDFs, construir índices e responder perguntas em tempo real.
- Scripts CLI para construir o índice e testar consultas localmente.
- Testes básicos e Dockerfile para empacotamento.

Como começar (local)
1. Crie e ative um ambiente virtual:
   python -m venv venv
   source venv/bin/activate  # Linux / macOS
   venv\Scripts\activate     # Windows

2. Instale dependências:
   pip install -r requirements.txt

3. Construa um índice a partir de PDFs:
   python scripts/build_index.py --pdf-dir ./data/pdfs --index-path models/index.faiss --meta-path models/metadata.joblib

4. Rode a API:
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Endpoints principais:
- POST /upload-pdfs — envia PDFs (multipart/form-data) e os adiciona ao diretório de documentos.
- POST /build-index — (opcional) inicia a construção do índice a partir dos PDFs armazenados.
- POST /query — envia {"query": "sua pergunta"} e recebe resposta e fontes.

Configuração opcional:
- Para usar o OpenAI como gerador de respostas, exporte OPENAI_API_KEY no ambiente. Se não houver chave, o sistema tentará usar um modelo HuggingFace (ex.: google/flan-t5-small) se estiver instalado.

Persistência:
- O índice FAISS é salvo em disco (models/index.faiss).
- Metadados (lista de chunks, mapeamento arquivo->chunk) são salvos em models/metadata.joblib.

Boas práticas / próximos passos sugeridos
- Adicionar monitoramento do índice e processo de reindexação incremental.
- Adotar embeddings e geradores com melhores modelos se for necessário maior qualidade.
- Implementar controle de custos e limites se usar APIs pagas.
- Rodar testes com PDFs reais (diversos layouts) e tratamento avançado de OCR (tika/ocr) quando necessário.

```
