from langchain_core.runnables import RunnablePassthrough

from table import table_chain
from query import query_chain


# Set table_names_to_use using table_chain.
full_chain = RunnablePassthrough.assign(table_names_to_use=table_chain) | query_chain

query = full_chain.invoke(
    {"question": "What are all the genres of Alanis Morissette songs"}
)
print(query)