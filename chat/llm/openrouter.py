import os
import getpass
import dotenv
from typing import Literal
from pydantic import SecretStr
from langchain_openai import ChatOpenAI


dotenv.load_dotenv()
if not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = getpass.getpass(
        "Enter API key for OpenRouter: ")


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


type OpenRouterModel = Literal["openai/gpt-4o-mini", "openai/gpt-oss-20b:free", "meta-llama/llama-3.3-8b-instruct:free"]

def build_llm(model: OpenRouterModel = "openai/gpt-oss-20b:free", temperature=0.2):
    llm = ChatOpenAI(
        api_key=SecretStr(OPENROUTER_API_KEY),
        base_url=OPENROUTER_BASE_URL,
        model=model,
        temperature=temperature,
    )

    return llm
