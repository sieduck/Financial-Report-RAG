import requests
from ragretrieve import retrieve
from prodmonitoring import log_query
import time
from dotenv import load_dotenv
import os
from google import genai
from config import GEMINI_MODEL_LLM, TOP_K_RETRIEVAL, TOP_K_FINAL, CONFIDENCE_THRESHOLD

# Put up here to run once the file loads
load_dotenv()

# print(os.getenv("GEMINI_API_KEY"))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))



def ask(question, confidence_threshold=CONFIDENCE_THRESHOLD, top_k_retrieval=TOP_K_FINAL, top_k_final=TOP_K_FINAL,
        gemini_model_llm=GEMINI_MODEL_LLM, collection_name="financial_docs"):
    
    # Time logging for retrieving chunks
    start_time_chunks = time.time()
    # Retrieve the relevant chunks
    chunks_and_scores = retrieve(question, confidence_threshold=confidence_threshold, top_k_retrieval=top_k_retrieval,
                                  top_k_final=top_k_final, collection_name="financial_docs")

    chunks = chunks_and_scores[0]

    # If due to confidence threshold can't find anything
    if not chunks:
        return f"Could not find relevant information in document to answer question (Confidence Threshold={confidence_threshold}). Scores of nearest relevant clusters are {chunks_and_scores[1]}"
    else:
        print(f"(Threshold={confidence_threshold}, nearest cluster scores are {chunks_and_scores[1]})")


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

    # response = requests.post(
    #     "http://localhost:11434/api/generate",
    #     json={
    #         "model": "qwen2.5:7b",
    #         "prompt": prompt,
    #         "stream": False
    #     }
    # )

    response = client.interactions.create(
        model=gemini_model_llm,
        input=prompt
    )

    generation_time = time.time() - start_time_llm

    answer = response.output_text

    log_query(question, answer, chunks, retrieval_time, generation_time, response.usage.total_input_tokens, 
              response.usage.total_output_tokens)
    return answer

# test it
# question = "Should I invest in apple based on the financial statements?"
# answer = ask(question)
# print(f"Q: {question}")
# print(f"\nA: {answer}")