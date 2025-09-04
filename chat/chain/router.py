from typing import List, Literal, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field


class Slots(BaseModel):
    group: Optional[str] = Field(
        default=None, description="List of table digest that relates to the user's question.")
    metric: Optional[Literal["count", "sum", "avg", "min", "max"]] = None
    time_range: Optional[str] = None
    missing: List[Literal["group", "metric", "time_range"]
                  ] = Field(default_factory=list)


class Route(BaseModel):
    decided: Literal["clarification", "direct_response", "table_selection_node"] = Field(
        description="Pick 'clarification' if unrelated or missing critical slots. Pick 'direct_response' if the conversation is ready for a direct answer."
    )
    reason: str = Field(
        description="One-sentence why you chose that path."
    )
    slots: Slots


SYSTEM = """You are an intelligent router. Decide if the conversation has ENOUGH info to direct response or build a SQL query. If this conversation need to build a SQL query do the MIN REQUIRED SLOTS, ASSUMPTION POLICY, and DECISION POLICY, or conversation enough info to direct response just do the DECISION POLICY

MIN REQUIRED SLOTS:
- group (table group/dataset)
- metric (e.g., count/sum/avg). If user says “how many/total count” -> metric=count.
OPTIONAL:
- time_range (assume all-time if not specified; DO NOT block on this unless question is time-sensitive)

ASSUMPTION POLICY:
- “how many”, “total”, “in total” => metric = count

DECISION POLICY:
- If conversation enough info to direct response => decided = direct_response
- If group AND entity AND metric are determined => decided = table_selection_node
- Otherwise => decided = clarification (fill `slots.missing`)

TABLE DIGEST:
{table_digest}

CONVERSATION:
{conversation}

OUTPUT:
Return JSON ONLY in the provided schema {{decided, reason, slots}}.
Do your internal analysis silently. Do NOT include chain-of-thought."""


def build_router_chain(llm: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "{current_input}")
    ])
    return prompt | llm.with_structured_output(Route)
