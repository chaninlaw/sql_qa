"""
Deal with large databases when doing SQL question-answering
![Link](https://python.langchain.com/docs/how_to/sql_large_db/)
"""

from operator import itemgetter
from typing import List, Optional
import uuid
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from chat.db.sql import SQLDatabase
from chat.chain.query import build_query_chain, BaseChatModel
from chat.utils import draw_graph


class State(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages]
    relevant_tables: Optional[List[str]] = Field(
        default=None, description="List of relevant database tables.")
    sql_query: Optional[str] = Field(
        default=None, description="The SQL query generated for the question.")
    sql_result: Optional[str] = Field(
        default=None, description="The results from executing the SQL query.")
    final_answer: Optional[str] = Field(
        default=None, description="The final natural language answer.")
    retry_count: int = Field(
        default=0, description="Number of retry attempts.")
    error: Optional[str] = Field(
        default=None, description="Any error that occurred during processing.")


class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")


system = """Return the names of any SQL tables that are relevant to the user question.
The tables are:

Music
Business
"""
output_parser = PydanticToolsParser(tools=[Table])
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(system),
    HumanMessage("Question: {input}")
])


def build_chat(llm: BaseChatModel, db: SQLDatabase):
    """Build a StateGraph-based chat system for SQL Q&A."""

    # Create the state graph
    graph = StateGraph(State)

    # Create node functions with access to llm and db
    def normalize_node(state: State) -> State:
        """Ingest user question and relevant context, attach to messages"""
        return state
    
    def guard_node(state: State) -> State:
        """Guardrail, remove PII information, and ensure policy rules."""
        return state
    
    def route_node(state: State) -> State:
        """Decide whether to answer directly, call a table_selection_node, or ask for clarification."""
        return state

    def table_selection_node(state: State) -> State:
        """Node for selecting relevant tables based on user question."""
        try:
            llm_with_tools = llm.bind_tools([Table])
            category_chain = prompt | llm_with_tools | output_parser
            table_chain = {
                "input": itemgetter("messages")} | category_chain | get_tables

            relevant_tables = table_chain.invoke({"messages": state.messages})
            return state.model_copy(update={"relevant_tables": relevant_tables})
        except Exception as e:
            return state.model_copy(update={"error": f"Table selection error: {str(e)}"})

    def sql_generation_node(state: State) -> State:
        """Node for generating SQL query."""
        try:
            if state.error:
                return state  # Skip if there's already an error

            query_chain = build_query_chain(llm, db)

            # Prepare input for query chain
            query_input = {
                "messages": state.messages,
                "table_names_to_use": state.relevant_tables or []
            }

            sql_query = query_chain.invoke(query_input)
            return state.model_copy(update={"sql_query": sql_query})
        except Exception as e:
            return state.model_copy(update={"error": f"SQL generation error: {str(e)}"})

    def sql_execution_node(state: State) -> State:
        """Node for executing SQL query."""
        try:
            if state.error or not state.sql_query:
                return state  # Skip if there's an error or no query

            sql_result = db.run(state.sql_query)
            return state.model_copy(update={"sql_result": sql_result})
        except Exception as e:
            return state.model_copy(update={"error": f"SQL execution error: {str(e)}"})

    def summarization_node(state: State) -> State:
        """Node for generating final natural language answer."""
        try:
            if state.error:
                return state.model_copy(update={
                    "final_answer": f"I encountered an error while processing your question: {state.error}"
                })

            summary_prompt = ChatPromptTemplate.from_messages([
                SystemMessage("""You are a helpful assistant that provides clear, concise answers based on SQL query results.

Given a user question, the SQL query that was executed, and the results, provide a natural language answer that:
1. Directly answers the user's question
2. Is easy to understand
3. Includes relevant details from the results
4. Mentions if no results were found

Do not include the SQL query in your response unless specifically asked."""),
                HumanMessage("""Question: {question}

SQL Query: {sql_query}

Results: {sql_result}

Please provide a clear, natural language answer to the question based on these results.""")
            ])

            summary_chain = summary_prompt | llm
            final_answer = summary_chain.invoke({
                "messages": state.messages,
                "sql_query": state.sql_query,
                "sql_result": state.sql_result
            })

            return state.model_copy(update={"final_answer": final_answer.content})
        except Exception as e:
            return state.model_copy(update={
                "final_answer": f"I encountered an error while generating the summary: {str(e)}"
            })

    def approval_node(state: State) -> State:
        """Ask user to approve SQL query before execution."""
        print(f"About to execute: {state.sql_query}")
        approval = input("Approve? (y/n): ")

        if approval.lower() != 'y':
            return state.model_copy(update={"error": "Query rejected by user"})
        return state

    def retry_logic_node(state: State) -> State:
        """Decide whether to retry or continue."""
        if state.error and state.retry_count < 3:
            print("🔄 Retrying SQL generation...")
            return state.model_copy(update={
                "retry_count": state.retry_count + 1,
                "error": None,  # Clear error for retry
                "sql_query": None  # Clear previous query to force regeneration
            })
        elif state.error:
            print(f"❌ Max retries reached. Final error: {state.error}")
            return state
        else:
            print("✅ Proceeding to summarization")
            return state

    def should_retry(state: State) -> str:
        """Determine if we should retry or proceed to summarization."""
        if state.error and state.retry_count < 3:
            return "sql_generation"  # Retry SQL generation
        else:
            return "summarization"   # Proceed to summarization

    # Add nodes to the graph
    graph.add_node("table_selection", table_selection_node)
    graph.add_node("sql_generation", sql_generation_node)
    graph.add_node("sql_execution", sql_execution_node)
    graph.add_node("summarization", summarization_node)
    graph.add_node("approval", approval_node)
    graph.add_node("retry_logic", retry_logic_node)

    # Define the flow
    graph.set_entry_point("table_selection")
    graph.add_edge("table_selection", "sql_generation")
    graph.add_edge("sql_generation", "approval")
    graph.add_edge("approval", "sql_execution")
    graph.add_edge("sql_execution", "retry_logic")
    graph.add_edge("summarization", END)

    # Define conditional edges for retry logic
    graph.add_conditional_edges(
        "retry_logic",
        should_retry,
        {
            "sql_generation": "sql_generation",  # Retry path
            "summarization": "summarization"     # Success/max retry path
        }
    )

    # Compile the graph
    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)

    # Draw the graph
    try:
        draw_graph(compiled_graph)
    except:
        print("📊 Graph is unavailable.")

    # Return a function that uses the graph, thread_id is necessary for checkpointer
    def chat_function(question: str,
                      config=RunnableConfig({"configurable": {"thread_id": uuid.uuid4()}})) -> str:
        """Main chat function that processes a question and returns an answer."""
        initial_state = State(messages=[HumanMessage(content=question)])
        final_state = compiled_graph.invoke(initial_state, config=config)

        # Handle different return types from the graph
        if isinstance(final_state, dict):
            return final_state.get("final_answer", "I couldn't process your question.")
        else:
            return getattr(final_state, "final_answer", "I couldn't process your question.")

    return chat_function


def get_tables(categories: List[Table]) -> List[str]:
    tables = []
    for category in categories:
        if category.name == "Music":
            tables.extend(
                [
                    "Album",
                    "Artist",
                    "Genre",
                    "MediaType",
                    "Playlist",
                    "PlaylistTrack",
                    "Track",
                ]
            )
        elif category.name == "Business":
            tables.extend(["Customer", "Employee", "Invoice", "InvoiceLine"])
    return tables
