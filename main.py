from aiapi import ask

print("Financial Report RAG Assistant w/ QWEN\n")
print("Ask a question about the uploaded financial statements\n")


while True:
    question = input("Input: ")
    if question.lower() == "quit":
        break
    answer = ask(question)
    print(f"\GEMINI: {answer}\n")