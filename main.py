from chat import large_db
from chat.db import sql
from chat.llm import openrouter

db = sql.build_sql_database("sqlite:///Chinook.db")
llm = openrouter.build_llm()
chat = large_db.build_chat(llm, db)

while True:
    try:
        user_input = input(">>: ")
        chat(user_input)
    except EOFError:
        print("Bye")
        break
