"""
Deal with large databases when doing SQL question-answering
![Link](https://python.langchain.com/docs/how_to/sql_large_db/)
"""

from typing import List, Optional, Literal, cast
import uuid
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages.utils import get_buffer_string
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from chat.db.sql import SQLDatabase
from chat.chain import query, router, clarify, table_categorize, summarization
from chat.utils import draw_graph
from chat.llm import openai, ollama

llm_tool = ollama.build_llm(temperature=0)
llm0 = openai.build_llm(
    # model="meta-llama/llama-3.3-8b-instruct:free"
    temperature=0)


MAX_RETRIES = 3
TABLE_GROUPS = {
    "Music":    ["Album", "Artist", "Genre", "MediaType", "Playlist", "PlaylistTrack", "Track"],
    "Business":   ["Customer", "Employee", "Invoice", "InvoiceLine"],
}


class State(BaseModel):
    current_input: str = Field(description="The current user's input.")
    group: Optional[str] = Field(
        default=None, description="e.g., Business, Music")
    metric: Optional[Literal["count", "sum", "avg", "min", "max"]] = None
    time_range: Optional[str] = None
    summary: Optional[str] = Field(
        default=None, description="Summary of the conversation so far.")
    messages: Annotated[List[AnyMessage],
                        add_messages] = Field(default_factory=list)
    relevant_tables: Optional[List[str]] = Field(
        default=None, description="List of relevant database tables.")
    sql_query: Optional[str] = Field(
        default=None, description="The SQL query generated for the messages.")
    sql_result: Optional[str] = Field(
        default=None, description="The results from executing the SQL query.")
    final_answer: Optional[str] = Field(
        default=None, description="The final natural language answer.")
    retry_count: int = Field(
        default=0, description="Number of retry attempts.")
    error: Optional[str] = Field(
        default=None, description="Any error that occurred during processing.")


def build_chat(llm: BaseChatModel, db: SQLDatabase):
    """Build a StateGraph-based chat system for SQL Q&A."""

    # Create the state graph
    graph = StateGraph(State)

    # Create node functions with access to llm and db
    def router_node(state: State):
        """Ingest user question and relevant context, attach to messages"""
        router_chain = router.build_router_chain(llm0)
        route: router.Route = cast(router.Route, router_chain.invoke({"table_digest": "\n".join(TABLE_GROUPS.keys()),
                                                                      "current_input": state.current_input,
                                                                      "conversation": state.messages}))

        return {
            "group": route.slots.group or state.group,
            "metric": route.slots.metric or state.metric,
            "time_range": route.slots.time_range or state.time_range,
            "route_decision": route.decided,
            "route_reason": route.reason,
        }

    def guard_node(state: State):
        """Guardrail, remove PII information, and ensure policy rules."""
        return {}

    def direct_response_node(state: State):
        """Directly response to user"""
        return {"final_answer": state.final_answer}

    def clarification_node(state: State):
        """Node for asking clarifying questions."""
        transcript = get_buffer_string(state.messages)
        chain = clarify.build_clarify_chain(llm)
        result: clarify.Clarification = cast(clarify.Clarification,
                                             chain.invoke({"conversation_text": transcript,
                                                           "table_digest": "\n".join(TABLE_GROUPS.keys())}))
        return {
            "final_answer": result.follow_up_question,
            "messages": [AIMessage(result.follow_up_question)],
            "summary": result.summary
        }

    def table_categorize_node(state: State):
        """Node for selecting relevant tables based on user question."""
        category_chain = table_categorize.build_table_categorize_chain(llm)
        table: List[table_categorize.Table] = category_chain.invoke({"question": state.current_input,
                                                                     "table_groups": "\n".join(TABLE_GROUPS.keys())})

        relevant_tables = table_categorize.groups_to_tables(table)
        return {"relevant_tables": relevant_tables,
                "messages": [AIMessage(f"Relevant tables: {', '.join(relevant_tables) or '∅'}")]}

    def sql_generation_node(state: State):
        """Node for generating SQL query."""
        try:
            if state.error:
                return {"error": state.error}

            query_chain = query.build_query_chain(llm0, db)
            sql_query = query_chain.invoke({
                "question": state.current_input,
                "relevant_tables": state.relevant_tables or [],
            })
            return {"sql_query": sql_query}
        except Exception as e:
            return {"error": f"SQL generation error: {str(e)}"}

    def sql_execution_node(state: State):
        """Node for executing SQL query."""
        try:
            if state.error or not state.sql_query:
                return {"error": state.error}

            sql_result = db.run(state.sql_query)
            return {"sql_result": sql_result}
        except Exception as e:
            return {"error": f"SQL execution error: {str(e)}"}

    def summarization_node(state: State):
        """Node for generating final natural language answer."""
        try:
            if state.error:
                # TODO
                return {"final_answer": f"I encountered an error while processing your question: {state.error}"}

            summary_chain = summarization.build_summarization_chain(llm)
            final_answer = summary_chain.invoke({
                "question": state.current_input,
                "sql_query": state.sql_query,
                "sql_result": state.sql_result
            })

            return {"final_answer": final_answer.content}
        except Exception as e:
            return {"final_answer": f"I encountered an error while generating the summary: {str(e)}"}

    def human_approval_node(state: State) -> Command[Literal["sql_execution", "direct_response"]]:
        """Ask user to approve SQL query before execution."""
        answer: str = interrupt({
            "sql_query": state.sql_query,
            "question": "Do you would like to run this query? (Y/n): "
        })
        if answer.strip().lower() == 'y':
            return Command(goto="sql_execution")
        else:
            return Command(goto="direct_response", update={"final_answer": "User declined to run the query."})

    def retry_logic_node(state: State):
        """Decide whether to retry or continue."""
        if state.error and state.retry_count < MAX_RETRIES:
            print("🔄 Retrying SQL generation...")
            return {
                "retry_count": state.retry_count + 1,
                "error": None,  # Clear error for retry
                "sql_query": None  # Clear previous query to force regeneration
            }
        elif state.error:
            print(f"❌ Max retries reached. Final error: {state.error}")
            return {"error": state.error}
        else:
            print("✅ Proceeding to summarization")
            return {"retry_count": 0, "error": None}

    def should_retry(state: State) -> Literal["sql_generation", "summarization"]:
        """Determine if we should retry or proceed to summarization."""
        if state.error and state.retry_count < MAX_RETRIES:
            return "sql_generation"
        else:
            return "summarization"

    def router_edge(state: State) -> Literal["clarification", "table_categorize", "direct_response"]:
        """Read last message."""
        last = state.messages[-1].content if state.messages else [
            "Decision: clarification"]
        if "Decision: direct_response" in last:
            return "direct_response"
        if "Decision: clarification" in last:
            return "clarification"
        return "table_categorize"

    # Add nodes to the graph
    graph.add_node("router", router_node)
    # graph.add_node("guard", guard_node)
    graph.add_node("direct_response", direct_response_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("table_categorize", table_categorize_node)
    graph.add_node("sql_generation", sql_generation_node)
    graph.add_node("sql_execution", sql_execution_node)
    graph.add_node("summarization", summarization_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("retry_logic", retry_logic_node)

    # Define the flow
    graph.set_entry_point("router")
    # graph.add_edge("router", "guard")
    graph.add_edge("table_categorize", "sql_generation")
    graph.add_edge("sql_generation", "human_approval")
    graph.add_edge("sql_execution", "retry_logic")
    graph.add_edge("direct_response", END)
    graph.add_edge("summarization", END)
    graph.add_edge("clarification", END)

    # Define the route decision node
    graph.add_conditional_edges(
        "router",
        router_edge,
        {
            "table_categorize": "table_categorize",
            "clarification": "clarification",
            "direct_response": "direct_response"
        }
    )

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
    except Exception as e:
        print("📊 Graph is unavailable.\n", e)

    # Return a function that uses the graph, thread_id is necessary for checkpointer
    def chat_function(current_input: str, thread_id: Optional[str] = None):
        """Main chat function that processes a input and returns an answer."""
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        config = RunnableConfig({"configurable": {"thread_id": thread_id}})

        # Add current input to messages to build conversation history
        input_state = State(
            current_input=current_input,
            messages=[HumanMessage(content=current_input)]
        )
        state = compiled_graph.invoke(input_state, config=config)

        if "__interrupt__" in state:
            sql_query = state["__interrupt__"][0].value["sql_query"]
            question = state["__interrupt__"][0].value["question"]
            print(f"🤖 SQL Query: {sql_query}")
            user_decision = input(question)

            command = Command(goto="human_approval", resume=user_decision)
            result = compiled_graph.invoke(command, config=config)
            if "final_answer" in result:
                print(f"💡{result.get('final_answer', 'No Answer')}")
            elif "error" in result:
                print(
                    f"❌ Error occurred: {result.get('error', 'Unknown error')}")
            else:
                print("Thinking after confirm...")

        elif "error" in state:
            print(f"❌ Error occurred: {state.get('error', 'Unknown error')}")
        elif "final_answer" in state:
            print(f"💡{state.get('final_answer', 'No Answer')}")
        else:
            # Handle different return types from the graph
            print("Thinking...")

    return chat_function
