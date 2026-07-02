# Financial Report RAG

**[Website deployment](https://sieduckfinancialrag.streamlit.app/)**

Question-answering system that enables beginner investors to ask plain English questions about the financial statements of any company. To access, simply visit 

## Tech Stack
- **LLM:** gemini-2.5-flash // gemini-2.5-flash-lite
- **Embeddings:** gemini-embedding-001 (Via Gemini API)
- **Vector Store:** ChromaDB
- **PDF Parsing:** PyMuPDF
- **Re-ranking:** Flashrank
- **Front-end:** Streamlit

## How to run locally

1. Create .env file in directory and add gemini API key

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run app.py
```bash
streamlit run app.py
```

## Workings of each py file

1. **ingestpdf.py** - Load the PDFs, split them into chunks and embed using 'gemini-embedding-001'
2. **ragretrieve.py** - When question is asked, most relevant chunks are found using K-NN (ChromaDB)
3. **aiapi.py** - Retrieved chunks and questions get passed through to Gemini that goes ahead and generates a plain English answer
4. **evaluate.py** - Retrieves questions and answers from golden_dataset, then asks the question to the LLM, to which we compare the answer with the expected answer through judge-by-llm.
5. **app.py** - Streamlit UI interface for the entire application
6. **prodmonitoring.py** - Retrieves statistics regarding generation and saves them into 'monitor_log.json'
7. **config.py** - Default settings for model


## Roadmap

- [x] Document ingestion pipeline (PDF → chunks → embeddings → ChromaDB)
- [x] Vector search (ChromaDB KNN)
- [X] Retrieval reranking
- [X] Evaluation with golden datasets
- [X] LLM-as-judge for answer quality
- [X] Bare-bones frontend via Streamlit
- [X] Confidence thresholds (reject low-scoring retrievals)
- [X] Cost analysis per query (token usage)
- [X] config.py for tweakable parameters (chunk size, top-k, model, temperature)
- [X] Deploy to Streamlit Cloud