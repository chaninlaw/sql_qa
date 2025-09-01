from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities.sql_database import SQLDatabase
from langgraph.prebuilt import create_react_agent


SYSTEM = """You are a {dialect} expert. Given an input question, create a syntactically correct {dialect} query to run.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per {dialect}. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".

Only use the following tables:
{table_info}

Write an initial draft of the query. Then double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

Use format:

First draft: <<FIRST_DRAFT_QUERY>>
Final answer: <<FINAL_ANSWER_QUERY>>
"""

FEW_SHOTS = [
    # Positive example — ไม่มี code fence
    ("human", 'Find 3 genres for artist "Adele".'),
    ("ai",
     "First draft: SELECT DISTINCT \"g\".\"Name\" FROM \"Genre\" \"g\" "
     "JOIN \"Track\" \"t\" ON \"g\".\"GenreId\" = \"t\".\"GenreId\" "
     "JOIN \"Album\" \"a\" ON \"t\".\"AlbumId\" = \"a\".\"AlbumId\" "
     "JOIN \"Artist\" \"ar\" ON \"a\".\"ArtistId\" = \"ar\".\"ArtistId\" "
     "WHERE \"ar\".\"Name\" = 'Adele' LIMIT 3;\n"
     "Final answer: SELECT DISTINCT \"g\".\"Name\" FROM \"Genre\" \"g\" "
     "JOIN \"Track\" \"t\" ON \"g\".\"GenreId\" = \"t\".\"GenreId\" "
     "JOIN \"Album\" \"a\" ON \"t\".\"AlbumId\" = \"a\".\"AlbumId\" "
     "JOIN \"Artist\" \"ar\" ON \"a\".\"ArtistId\" = \"ar\".\"ArtistId\" "
     "WHERE \"ar\".\"Name\" = 'Adele' LIMIT 3;"),
    # Contrastive — ชี้ว่าห้ามใส่รั้ว
    ("human", 'List 5 tracks for artist "Radiohead".'),
    ("ai",
     "First draft: ```sql SELECT \"t\".\"Name\" FROM \"Track\" \"t\" ... LIMIT 5; ```\n"
     "Final answer (RAW SQL only): SELECT \"t\".\"Name\" FROM \"Track\" \"t\" "
     "JOIN \"Album\" \"a\" ON \"t\".\"AlbumId\" = \"a\".\"AlbumId\" "
     "JOIN \"Artist\" \"ar\" ON \"a\".\"ArtistId\" = \"ar\".\"ArtistId\" "
     "WHERE \"ar\".\"Name\" = 'Radiohead' LIMIT 5;"),
]


def build_query_chain(llm: BaseChatModel, db: SQLDatabase):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        *FEW_SHOTS,
        ("human", "{input}")
    ]).partial(dialect=db.dialect)

    chain = create_sql_query_chain(llm=llm, db=db, prompt=prompt)

    return chain | parse_final_answer


def parse_final_answer(output: str) -> str:
    return output.split("Final answer: ")[1]
