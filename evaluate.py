import json
from aiapi import ask
import requests

def llm_as_judge(question, expected, answer):
    # Prompt engineering
    prompt = f"""Please evaluate whether an AI assistant's answer includes the expected information. Please score the answer from 1-5,
    where 1 means completely incorrect and missing the point, and 5 means correct, clear and extremely helpful information for a 
    beginner investor.

    Question: {question}
    Expected information: {expected}
    AI assistant answer: {answer}

    

    Ensure to only output the rating from 1-5 and nothing else."""

    #send to Qwen via Ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False
        }
    )

    print(f"LLM GRADES RESPONSE AS: {response.json()["response"]}")
    return int(response.json()["response"])



def evaluate():
    with open("golden_dataset.json") as f:
        dataset = json.load(f)

    # Scores how many correct by just looking for the word in the response
    correct = 0

    # Scores how many correct for llm-by-judge
    correct_llm = 0

    # Note dataset JSON has question expected for each row
    for item in dataset:
        question = item["question"]
        expected = item["expected"]

        print(f"Question: {question}")
        answer = ask(question)
        print(f"Answer: {answer}")

        # process them to get rid of dollar signs and commas to match answer
        processed_answer = answer.lower()

        if expected in answer.replace(",", "").replace("$",""):
            print(f"Expected: {expected}, FOUND IN ANSWER")
            correct += 1
            
        
        correct_llm += llm_as_judge(question, expected, answer)


    total_score = (correct / len(dataset)) * 100
    print(f"Simple Eval: Score in percentage: {total_score}, Raw {correct}/{len(dataset)}\n")

    possible_levels = 5
    total_possible_llm_score = len(dataset) * possible_levels

    total_score_llm = (correct_llm / total_possible_llm_score) * 100
    print(f"LLM Eval: Score in percentage: {total_score_llm}, Raw {correct_llm}/{total_possible_llm_score}")


evaluate()