
from typing import Literal
from langchain_ollama.chat_models import ChatOllama


type OllamaModel = Literal["llama3.2:latest"]

def build_llm(model: OllamaModel = "llama3.2:latest", temperature=0.2):
    llm = ChatOllama(model=model, temperature=temperature)
    return llm