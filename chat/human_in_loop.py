from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from chain import SQLDatabase, BaseChatModel

system_message = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(dialect="SQLite", top_k=5,)

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate([
    ("system", system_message),
    ("user", user_prompt)
])


def build_chat(llm: BaseChatModel, db: SQLDatabase):
    """
    Example:
        ```python
        agent_executor = build_chat(llm, db)
        for step in agent_executor.stream(
            {"messages": [{"role": "user", "content": "How many employees are there?"}]},
            config=RunnableConfig({"configurable": {"thread_id": "1"}}),
            stream_mode="values",
            interrupt_before=["tools"]
        ):
            step["messages"][-1].pretty_print()
        ```
    """
    toolkit = SQLDatabaseToolkit(llm=llm, db=db)
    tools = toolkit.get_tools()

    agent_executor = create_react_agent(llm, tools, prompt=system_message)
    return agent_executor


def human_in_the_loop(agent_executor: CompiledStateGraph, config: RunnableConfig):
    try:
        user_approval = input("Do you want to go to execute query? (Y/n): ")
    except Exception:
        user_approval = "n"

    if user_approval.lower() == "y":
        # If approved, continue the graph execution
        for step in agent_executor.stream(None, config, stream_mode="updates"):
            print(step)
    else:
        print("Operation cancelled by user.")
