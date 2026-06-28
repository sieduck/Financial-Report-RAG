import chromadb
import requests
from flashrank import Ranker, RerankRequest

# Download the reranker model
reranker = Ranker()

def embed(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )
    return response.json()["embedding"]

def retrieve(question, confidence_threshold=0.7, n_results=3):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("financial_docs")

    
    question_embedding = embed(question)
    # Instead of 3-NN we get 10-NN first
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=10
    )

    # Note that passages takes in a dictionary of format "text: content"
    # So here we convert to that dictionary formaat
    top_ten_chunks = results["documents"][0] 
    top_ten_chunks_flashrank = []
    for message in top_ten_chunks:
        top_ten_chunks_flashrank.append({"text": message})


    # Now use flash rank to re-rank
    rerank_req = RerankRequest(query=question, passages=top_ten_chunks_flashrank)

    # Put those top 3 into a new list
    reranked = reranker.rerank(rerank_req)

    test_conf_list = []

    # Print out confidence scores
    for r in reranked:
        print(f"Score: {r['score']:.4f} | Text: {r['text'][:50]}")
        test_conf_list.append({"Score": f"{r['score']:.4f}", "Text": f"{r['text'][:50]}"})

    # Now take the top 3 chunks from the reranked
    final_results = []

    for i in range(0, n_results):
        if reranked[i]["score"] >= confidence_threshold:
            final_results.append(reranked[i]["text"])
    
    return [final_results, test_conf_list]

# testing
# question = "Should I invest in Apple?"
# chunks = retrieve(question)

# print(f"Question: {question}\n")
# print("Top matching chunks:")
# for i, chunk in enumerate(chunks):
#     print(f"\n--- Chunk {i+1} ---")
#     print(chunk)