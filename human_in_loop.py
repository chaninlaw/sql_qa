from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from llm import llm
from db import db

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

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
agent_executor = create_react_agent(llm, tools, prompt=system_message)

png_bytes = agent_executor.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)

# How many employees are there?
user_input = input("Enter your question: ")
config = RunnableConfig({"configurable": {"thread_id": "1"}})
for step in agent_executor.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config=config,
    stream_mode="values",
    interrupt_before=["tools"]
):
    step["messages"][-1].pretty_print()

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
