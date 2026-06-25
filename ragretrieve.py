import chromadb
import requests

def embed(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )
    return response.json()["embedding"]

def retrieve(question, n_results=3):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("financial_docs")
    
    question_embedding = embed(question)
    
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )
    
    return results["documents"][0]  # list of top 3 matching chunks

# testing
# question = "Should I invest in Apple?"
# chunks = retrieve(question)

# print(f"Question: {question}\n")
# print("Top matching chunks:")
# for i, chunk in enumerate(chunks):
#     print(f"\n--- Chunk {i+1} ---")
#     print(chunk)