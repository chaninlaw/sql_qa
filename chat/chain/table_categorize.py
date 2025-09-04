from typing import List
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

TABLE_GROUPS = {
    "Music":    ["Album", "Artist", "Genre", "MediaType", "Playlist", "PlaylistTrack", "Track"],
    "Business":   ["Customer", "Employee", "Invoice", "InvoiceLine"],
}

SYSTEM = """Return the names of ALL the SQL tables that MIGHT be relevant to the user question. Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed.
The tables are:

{table_groups}"""


class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")


def build_table_categorize_chain(llm: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "Question: {question}")
    ])

    llm_with_tool = llm.bind_tools([Table], tool_choice="required")
    output_parser = PydanticToolsParser(tools=[Table])
    return prompt | llm_with_tool | output_parser


def groups_to_tables(categories: List[Table]) -> List[str]:
    out = []
    for c in categories:
        out.extend(TABLE_GROUPS.get(c.name, []))
    return sorted(set(out))
