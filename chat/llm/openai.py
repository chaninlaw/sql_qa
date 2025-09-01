import os
import getpass
import dotenv
from typing import Literal

from langchain.chat_models import init_chat_model


dotenv.load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass(
        "Enter API key for OpenAI: ")


type OpenAIModel = Literal["gpt-4o-mini"]


def build_llm(model: OpenAIModel = "gpt-4o-mini"):
    llm = init_chat_model(model, model_provider="openai")
    return llm
