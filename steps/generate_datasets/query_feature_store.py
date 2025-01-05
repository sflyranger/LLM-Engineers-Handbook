from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from qdrant_client.http import exceptions
from typing_extensions import Annotated
from zenml import step

from llm_engineering.domain.base.nosql import NoSQLBaseDocument
from llm_engineering.domain.cleaned_documents import (
    CleanedRepositoryDocument, 
    CleanedArticleDocument, 
    CleanedDocument, 
    CleanedPostDocument,
)

# Create the zenml step to query Mongo db feature store
@step
def query_feature_store() -> Annotated[list, "queried_cleaned_documents"]:
    logger.info("Querying feature store.")

    # Setting the results variable to fetch all of the document types.
    results = fetch_all_data()

    # Query the documents
    cleaned_documents = [doc for query_result in results.values() for doc in query_result]

    return cleaned_documents

# Function to fetch all data from the feature store
def fetch_all_data() -> dict[str, list[NoSQLBaseDocument]]:
    with ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(
                __fetch_articles,
            ): "articles",
            executor.submit(
                __fetch_posts,
            ): "posts",
            executor.submit(
                __fetch_repositories,
            ): "repositories",
        }

        results = {}
        for future in as_completed(future_to_query):
            query_name = future_to_query[future]
            try:
                results[query_name] = future.result()
            except Exception:
                logger.exception(f"'{query_name}' request failed.")

                results[query_name] = []

    return results

# Function to fetch articles from the feature store
def __fetch_articles() -> list[CleanedDocument]:
    return __fetch(CleanedArticleDocument)

# Function to fetch posts from the feature store
def __fetch_posts() -> list[CleanedDocument]:
    return __fetch(CleanedPostDocument)

# Function to fetch repositories from the feature store
def __fetch_repositories() -> list[CleanedDocument]:
    return __fetch(CleanedRepositoryDocument)

# Generic function to fetch documents of a given type from the feature store
def __fetch(cleaned_document_type: type[CleanedDocument], limit: int = 1) -> list[CleanedDocument]:
    try:
        cleaned_documents, next_offset = cleaned_document_type.bulk_find(limit=limit)
    except exceptions.UnexpectedResponse:
        return []

    while next_offset:
        documents, next_offset = cleaned_document_type.bulk_find(limit=limit, offset=next_offset)
        cleaned_documents.extend(documents)

    return cleaned_documents