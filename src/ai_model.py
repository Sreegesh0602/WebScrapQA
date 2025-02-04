# AI model integration

from langchain_ollama import ChatOllama

def get_ai_response(prompt):
    model = ChatOllama()
    return model.run(prompt)
