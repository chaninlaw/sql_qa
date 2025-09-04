from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

SYSTEM = """You are a helpful assistant that provides clear, concise answers based on SQL query results.

Given a user question, the SQL query that was executed, and the results, provide a natural language answer that:
1. Directly answers the user's question
2. Is easy to understand
3. Includes relevant details from the results
4. Mentions if no results were found

Do not include the SQL query in your response unless specifically asked."""

HUMAN = """Question: {question}

SQL Query: {sql_query}

Results: {sql_result}

Please provide a clear, natural language answer to the question based on these results."""


def build_summarization_chain(llm: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", HUMAN)
    ])

    return prompt | llm
