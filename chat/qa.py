from typing import TypedDict, Optional
from pydantic import BaseModel, BaseModel, Field
from typing_extensions import Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.runnables import RunnableConfig
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.prompts import ChatPromptTemplate

from chat.db.sql import build_sql_database
from chat.llm.openai import build_llm


class State(BaseModel):
    question: Optional[str] = Field(
        default=None, description="User's question.")
    query: Optional[str] = Field(
        default=None, description="Generated SQL query.")
    result: Optional[str] = Field(
        default=None, description="Result of the SQL query.")
    answer: Optional[str] = Field(
        default=None, description="Final answer to the user's question.")


llm = build_llm()
db = build_sql_database("sqlite:///Chinook.db")
system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}
"""

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state.question,
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    if isinstance(result, BaseModel[QueryOutput]):
        return {"query": result.model_dump().get("query", "<NO QUERY>")}
    return {"query": result.get("query", "<NO QUERY>")}


def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    query = state.query
    if query is None:
        raise ValueError("No SQL query to execute.")
    return {"result": execute_query_tool.invoke(query)}


def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f"Question: {state.question}\n"
        f"SQL Query: {state.query}\n"
        f"SQL Result: {state.result}"
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")

memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory, interrupt_before=["execute_query"])

# Now that we're using persistence, we need to specify a thread ID
# so that we can continue the run after review.
config = RunnableConfig({"configurable": {"thread_id": "1"}})
state = State(question="How many employees are there?")

for step in graph.stream(
    state,
    config,
    stream_mode="updates",
):
    print(step)

try:
    user_approval = input("Do you want to go to execute query? (yes/no): ")
except Exception:
    user_approval = "no"

if user_approval.lower() == "yes":
    # If approved, continue the graph execution
    for step in graph.stream(None, config, stream_mode="updates"):
        print(step)
else:
    print("Operation cancelled by user.")
