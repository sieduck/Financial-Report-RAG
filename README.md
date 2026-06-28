# Financial Report RAG

Question-answering system that enables beginner investors to ask plain English questions about the financial statements of any company.

## Tech Stack
- **LLM:** Qwen2.5:7b (via Ollama, runs locally)
- **Embeddings:** nomic-embed-text (via Ollama)
- **Vector Store:** ChromaDB
- **PDF Parsing:** PyMuPDF
- **Re-ranking:** Flashrank
- **Front-end:** Streamlit

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

5. To evalute via golden dataset, load in questions to 'golden_dataset.json' in the format:
```json
{ 
    "question": "Who is the founder of KFC?",
    "expected": "Colonel Sanders"
}
```

and then simply run:
```bash
python evaluate.py
```

## Workings of each py file

1. **ingestpdf.py** - Load the PDFs, split them into chunks and embed using 'nomic-embed-text' via Ollama
2. **ragretrieve.py** - When question is asked, most relevant chunks are found using K-NN (ChromaDB)
3. **aiapi.py** - Retrieved chunks and questions get passed through to Qwen2.5:7b that goes ahead and generates a plain English answer
4. **evaluate.py** - Retrieves questions and answers from golden_dataset, then asks the question to the LLM, to which we compare the answer with the expected answer through both simple keyword matching and judge-by-llm.
5. **app.py** - Streamlit UI interface for the entire application
6. **prodmonitoring.py** - Retrieves statistics regarding generation and saves them into 'monitor_log.json'


## Roadmap

- [x] Document ingestion pipeline (PDF → chunks → embeddings → ChromaDB)
- [x] Vector search (ChromaDB KNN)
- [X] Retrieval reranking
- [X] Evaluation with golden datasets
- [X] LLM-as-judge for answer quality
- [X] Bare-bones frontend via Streamlit
- [ ] Hybrid retrieval (BM25 + vector search)
- [X] Confidence thresholds (reject low-scoring retrievals)
- [X] Cost analysis per query (token usage)
- [ ] config.py for tweakable parameters (chunk size, top-k, model, temperature)
- [ ] CI/CD via GitHub Actions (auto-run eval on every push)
- [ ] Deploy to Streamlit Cloud