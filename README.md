# Financial Report RAG

Question-answering system that enables beginner investors to ask plain English questions about the financial statements of any company.

## Tech Stack
- **LLM:** Qwen2.5:7b (via Ollama, runs locally)
- **Embeddings:** nomic-embed-text (via Ollama)
- **Vector Store:** ChromaDB
- **PDF Parsing:** PyMuPDF

## How to run

1. Install [Ollama](https://ollama.com) and pull models:
```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

2. Install dependencies:
```bash
pip install chromadb pymupdf requests
```

3. Add a financial report PDF to the `docs/` folder, then ingest it:
```bash
python ingestpdf.py
```

4. Ask questions:
```bash
python main.py
```


## Workings of each py file

1. **ingestpdf.py** - Load the PDFs, split them into chunks and embed using 'nomic-embed-text' via Ollama
2. **ragretrieve.py** - When question is asked, most relevant chunks are found using K-NN (ChromaDB)
3. **aiapi.py** - Retrieved chunks and questions get passed through to Qwen2.5:7b that goes ahead and generates a plain English answer

## Roadmap

- [x] Document ingestion pipeline (PDF → chunks → embeddings → ChromaDB)
- [x] Vector search (ChromaDB KNN)
- [ ] Retrieval reranking
- [ ] Evaluation with golden datasets
- [ ] LLM-as-judge for answer quality
- [ ] Production monitoring (latency, retrieval quality, cost)