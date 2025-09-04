from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field


class Clarification(BaseModel):
    summary: str = Field(
        description="1–2 sentences; end with how AI can help.")
    follow_up_question: str = Field(
        description="ONE short, polite, specific question.")


SYSTEM = """You are “Summary+Clarify” for SQL Q&A.
Output JSON ONLY per schema:
- summary: 1–2 sentences. End with one clause of how you (the AI) can help next.
- follow_up_question: ONE short, polite, specific question that resolves the MOST blocking missing info among:
  table/dataset, entity, metric, time_range, filters.

Tables:\n{table_digest}

Conversation:
{conversation_text}"""


def build_clarify_chain(llm: BaseChatModel):
    clarify_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
    ])
    return clarify_prompt | llm.with_structured_output(Clarification)
