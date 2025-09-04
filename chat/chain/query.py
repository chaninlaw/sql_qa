import json
import os
from typing import List, Literal, Dict
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.prompts import FewShotPromptTemplate, PromptTemplate, ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

EXAMPLES: List[Dict[str, Dict[Literal["input", "query"], str]]] = []
try:
    config_path = os.getenv('QUERY_EXAMPLES_CONFIG_PATH',
                            'config/query_examples.json')
    with open(config_path, 'r') as f:
        EXAMPLES = json.load(f)
except Exception as e:
    print(f"Error loading query examples: {e}")


def extract_sql_query(text: str) -> str:
    """Extract SQL query from LLM response, removing SQLQuery: prefix if present"""
    # Remove SQLQuery: prefix if present
    if "SQLQuery:" in text:
        sql_part = text.split("SQLQuery:")[1]
    else:
        sql_part = text

    # Clean up
    sql_part.strip()

    # Remove any trailing text after semicolon if there's explanation
    if ';' in sql_part:
        sql_part = sql_part.split(';')[0] + ';'

    return sql_part


def build_query_chain(llm: BaseChatModel, db: SQLDatabase):
    few_shot_prompt = FewShotPromptTemplate(
        examples=EXAMPLES,
        example_prompt=PromptTemplate.from_template(
            "Question: {input}\nSQLQuery: {query}"),
        prefix="Below are a number of examples of questions and their corresponding SQL queries:",
        suffix="",
        input_variables=[]
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("""
You are a {dialect} expert. Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer to the input question. Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per {dialect}. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".

{few_shot_examples}

Relevant tables: {relevant_tables}
                        
Use the following format:

Question: Question here
SQLQuery: SQL query to run

Only use the following tables:
{table_info}\n"""),
            HumanMessagePromptTemplate.from_template("Question: {question}"),
        ],
    ).partial(
        top_k=str(50),
        dialect=db.dialect,
        table_info=db.get_table_info(),
        few_shot_examples=few_shot_prompt.format()
    )

    # Create a chain that includes SQL extraction
    return final_prompt | llm | StrOutputParser() | extract_sql_query
