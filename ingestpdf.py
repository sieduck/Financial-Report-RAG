import fitz
import chromadb
import requests
import json


# Loads the pdf and converts it into raw text
def pdf_load(path):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()

        # skipping blank pages
        if text.strip():
            pages.append({"page": i + 1, "text": text, "source": path})
    return pages


# Cleans out the \n and chunks out the text into a list of chunks of clean processed text
def chunking_text(pages, chunk_size=500, overlap=50):
    chunks = []
    for page in pages:
        text = page["text"]
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:1 + chunk_size])
            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "page": page["page"],
                    "source": page["source"]

                })
    return chunks


# Convert into embeddings (in this case using ollama but can change this section for any other model)
def embed(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )
    return response.json()["embedding"]


def create_embedding_index(pdf_path):
    pages = pdf_load(pdf_path)
    print(f"{len(pages)} pages have been loaded\n")

    chunks = chunking_text(pages)
    print(f"{len(chunks)} chunks have been created\n")

    # Storing into chroma_db
    client = chromadb.PersistentClient(path="./chroma_db")

    # Ensure fresh chroma_db collection every time
    try:
        client.delete_collection("financial_docs")
    except:
        pass
    collection = client.create_collection("financial_docs")

    for i, chunk in enumerate(chunks):
        print(f"  Embedding chunk {i+1}/{len(chunks)}...", end="\r")
        embedding = embed(chunk["text"])
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{"page": chunk["page"], "source": chunk["source"]}]
        )


create_embedding_index("docs\FY26_Q1_Consolidated_Financial_Statements.pdf")
