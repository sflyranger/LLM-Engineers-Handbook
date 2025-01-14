import opik
from langchain_openai import ChatOpenAI
from loguru import logger

from llm_engineering.domain.queries import Query
from llm_engineering.settings import settings

from .base import RAGStep
from .prompt_templates import QueryExpansionTemplate

class QueryExpansion(RAGStep):
    # Tracks the execution of the generate method from the QueryExpansion class.
    @opik.track(name="QueryExpansion.generate")
    def generate(self, query: Query, expand_to_n: int) -> list[Query]:
        # Ensure expand_to_n is greater than 0
        assert expand_to_n > 0, f"'expand_to_n' should be greater than 0. Got {expand_to_n}."

        # Return mock queries if in mock mode
        if self._mock:
            return [query for _ in range(expand_to_n)]
        
        # Create a query expansion template
        query_expansion_template = QueryExpansionTemplate()
        prompt = query_expansion_template.create_template(expand_to_n - 1)
        
        # Initialize the OpenAI model
        model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, api_key=settings.OPENAI_API_KEY, temperature=0)

        # Create a chain of prompt and model
        chain = prompt | model

        # Invoke the chain with the query
        response = chain.invoke({"question", query})
        result = response.content

        # Split the result into individual queries
        queries_content = result.strip().split(query_expansion_template.seperator)

        # Create a list of queries
        queries = [query]
        queries += [
            query.replace_content(stripped_content)
            for content in queries_content
            if (stripped_content := content.strip())
        ]

        return queries 

if __name__ == "__main__":
    # Example usage of the QueryExpansion class
    query = Query.from_str("Write an article about the best types of advanced RAG methods.")
    query_expander = QueryExpansion()
    expanded_queries = query_expander.generate(query, expand_to_n=3)  # generating 3 more queries
    for expanded_query in expanded_queries:
        logger.info(expanded_query.content)
