from chat.large_db import build_chat
from chat.db.sql import build_sql_database
from chat.llm.openai import build_llm

db = build_sql_database("sqlite:///Chinook.db")
llm = build_llm()
chat = build_chat(llm, db)

user_input = input("Question: ")

response = chat(user_input)
print(response)
