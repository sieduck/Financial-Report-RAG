import requests
from ragretrieve import retrieve
from prodmonitoring import log_query
import time

def ask(question):

    # Time logging for retrieving chunks
    start_time_chunks = time.time()
    # Retrieve the relevant chunks
    chunks = retrieve(question)
    retrieval_time = time.time() - start_time_chunks
    context = "\n\n".join(chunks)

    # Prompt engineering
    prompt = f"""Act as a financial assistant to help beginner investors interpret financial reports to make investment decisions. 
    Use ONLY the context below to answer the question. If the answer is not in the context, say "I couldn't find that in the document."
    Explain in simple terms a beginner investor would understand.

    Context: {context}
    Question: {question}

    Answer:"""

    # Step 3: send to Qwen via Ollama
    # Time logging for response from LLM
    start_time_llm = time.time()

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False
        }
    )

    generation_time = time.time() - start_time_llm

    answer = response.json()["response"]

    log_query(question, answer, chunks, retrieval_time, generation_time)
    return answer

# test it
# question = "Should I invest in apple based on the financial statements?"
# answer = ask(question)
# print(f"Q: {question}")
# print(f"\nA: {answer}")