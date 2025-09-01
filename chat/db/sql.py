from sqlalchemy.engine import URL
from langchain_community.utilities import SQLDatabase


def build_sql_database(database_uri: str | URL):
    db = SQLDatabase.from_uri(database_uri)
    return db
