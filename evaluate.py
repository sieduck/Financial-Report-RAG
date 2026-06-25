import json
from aiapi import ask


def evaluate():
    with open("golden_dataset.json") as f:
        dataset = json.load(f)

    correct = 0

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
            print(f"Expected: {answer}, FOUND IN OUTPUT")
            correct += 1
            continue
        

        print(processed_answer)

    total_score = (correct / len(dataset)) * 100
    print(f"Score in percentage: {total_score}")
evaluate()