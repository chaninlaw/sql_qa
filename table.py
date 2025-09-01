from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from pydantic import BaseModel, Field

from llm import llm
from db import db

class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")


table_names = "\n".join(db.get_usable_table_names())
system = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:

{table_names}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{input}"),
    ]
)
llm_with_tools = llm.bind_tools([Table])
output_parser = PydanticToolsParser(tools=[Table])

table_chain = prompt | llm_with_tools | output_parser

# Convert "question" key to the "input" key expected by current table_chain.
table_chain = {"input": itemgetter("question")} | table_chain

# table_chain.invoke({"input": "What are all the genres of Alanis Morissette songs"})