import fitz
import chromadb
import requests
from google import genai
import os
from dotenv import load_dotenv
import json
from config import GEMINI_MODEL_EMBEDDING, CHUNK_SIZE, CHUNK_OVERLAP



load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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
def chunking_text(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
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

    response = client.models.embed_content (
        model=GEMINI_MODEL_EMBEDDING,
        contents=text


    )

    return response.embeddings[0].values


def create_embedding_index(pdf_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    pages = pdf_load(pdf_path)
    print(f"{len(pages)} pages have been loaded\n")

    chunks = chunking_text(pages, chunk_size, chunk_overlap)
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

# return the number of chunks for our opttinos
    return len(chunks)

# create_embedding_index("docs\FY26_Q1_Consolidated_Financial_Statements.pdf")
