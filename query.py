from langchain.chains import create_sql_query_chain

from llm import llm
from db import db

query_chain = create_sql_query_chain(llm, db)
