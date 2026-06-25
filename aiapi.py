import requests
from ragretrieve import retrieve

def ask(question):
    # Retrieve the relevant chunks
    chunks = retrieve(question)
    context = "\n\n".join(chunks)

    # Prompt engineering
    prompt = f"""Act as a financial assistant to help beginner investors interpret financial reports to make investment decisions. 
    Use ONLY the context below to answer the question. If the answer is not in the context, say "I couldn't find that in the document."
    Explain in simple terms a beginner investor would understand.

    Context: {context}
    Question: {question}

    Answer:"""

    # Step 3: send to Qwen via Ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]

# test it
# question = "Should I invest in apple based on the financial statements?"
# answer = ask(question)
# print(f"Q: {question}")
# print(f"\nA: {answer}")