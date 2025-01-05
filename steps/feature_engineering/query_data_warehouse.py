from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from typing_extensions import Annotated 
from zenml import get_step_context, step 

from llm_engineering.application import utils 
from llm_engineering.domain.base.nosql import NoSQLBaseDocument
from llm_engineering.domain.documents import ArticleDocument, Document, PostDocument, RepositoryDocument, UserDocument

# Zenml step to query the data warehouse
@step
def query_data_warehouse(author_full_name: list[str]) -> Annotated[list, "raw_documents"]:
    
    # Setting up empty lists for documents and authors
    documents = []
    authors = []

    # For every author query the data warehouse and return all results from the query.
    for author_full_name in author_full_name:
        logger.info(f"Querying the data warehouse for user: {author_full_name}")

        first_name, last_name = utils.split_user_full_name(author_full_name)
        logger.info(f"First name: {first_name} , Last name: {last_name}")

        user = UserDocument.get_or_create_user(first_name=first_name, last_name=last_name)
        authors.append(user)

        results = fetch_all_data(user)
        user_documents = [doc for query_result in results.values() for doc in query_result] # Pulls all the documents from the query result.

        documents.extend(user_documents)
    
    # Initialize the Zenml step context
    step_context = get_step_context()

    # Add the metadata for the outputs to the step context
    step_context.add_output_metadata(output="raw_documents", metadata=_get_metadata(documents))

    return documents 

# Function to pull all the user data.
def fetch_all_data(user:UserDocument) -> dict[str, list[NoSQLBaseDocument]]:
    user_id = str(user.id)
    with ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(__fetch_articles, user_id): "articles", # Bulk find the articles. 
            executor.submit(__fetch_posts, user_id): "posts", # Bulk find the posts.
            executor.submit(__fetch_repositories, user_id): "repositories" # Bulk find the repositories.
        }

        # Initialize an empty dictionary for the results.
        results = {}

        for future in as_completed(future_to_query):
            query_name = future_to_query[future]
            try:
                results[query_name] = future.result() # Store the results for the given query.
            except Exception:
                logger.exception(f"'{query_name}' request failed.")

                results[query_name] = [] # Empty list if query fails for the query name.
    
    return results 


def __fetch_articles(user_id) -> list[NoSQLBaseDocument]:
    return ArticleDocument.bulk_find(author_id=user_id)

def __fetch_posts(user_id) -> list[NoSQLBaseDocument]:
    return PostDocument.bulk_find(author_id=user_id)

def __fetch_repositories(user_id) -> list[NoSQLBaseDocument]:
    return RepositoryDocument.bulk_find(author_id=user_id)

def _get_metadata(documents: list[Document]) -> dict:
    metadata = {
        "num_documents": len(documents),
    }
    for document in documents:
        collection = document.get_collection_name()
        if collection not in metadata:
            metadata[collection] = {}
        if "authors" not in metadata[collection]:
            metadata[collection]["authors"] = list()

        metadata[collection]["num_documents"] = metadata[collection].get("num_documents", 0) + 1
        metadata[collection]["authors"].append(document.author_full_name)

    for value in metadata.values():
        if isinstance(value, dict) and "authors" in value:
            value["authors"] = list(set(value["authors"]))

    return metadata
        