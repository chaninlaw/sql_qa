from chat import large_db
from chat.db import sql
from chat.llm import openai
import uuid

db = sql.build_sql_database("sqlite:///Chinook.db")
llm = openai.build_llm()
chat = large_db.build_chat(llm, db)

# Use the same thread_id for the entire conversation
thread_id = str(uuid.uuid4())

while True:
    try:
        user_input = input(">>: ")
        chat(user_input, thread_id)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        break
