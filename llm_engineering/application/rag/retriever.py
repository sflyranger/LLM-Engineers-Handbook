import concurrent.futures


import opik
from loguru import logger
from qdrant_client.models import FieldCondition, Filter, MatchValue

from llm_engineering.application import utils
from llm_engineering.application.preprocessing.dispatchers import EmbeddingDispatcher
from llm_engineering.domain.embedded_chunks import (
    EmbeddedArticleChunk, 
    EmbeddedChunk, 
    EmbeddedPostChunk, 
    EmbeddedRepositoryChunk
)

from llm_engineering.domain.queries import Query

from .query_expansion import QueryExpansion
from .reranking import Reranker
from .self_query import SelfQuery

# Creating the base class to retrieve the relevant context from the vector DB.
class ContextRetriever:
    def __init__(self, mock: bool = False) -> None:
        self._query_expander = QueryExpansion(mock=mock)
        self._metadata_extractor = SelfQuery(mock=mock)
        self._reranker = Reranker(mock=mock)

    # Tracking te search function from the ContextRetriever class.
    @opik.track(name="ContecRetriever.search")
    def search(
        self, 
        query: str, 
        k: int=3, 
        expand_to_n_queries: int=3, 
    ) -> list:
        query_model = Query.from_str(query)
        
        query_model = self._metadata_extractor.generate(query_model)

        logger.info(
            f"Successfully extracted the author_full_name = {query_model.author_full_name} from the query.",
        )

        n_generated_queries = self._query_expander.generate(query_model, expand_to_n=expand_to_n_queries)
        logger.info(
            f"Successfully generated {len(n_generated_queries)} search queries.",
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_tasks = [executor.submit(self._search, _query_model, k) for _query_model in n_generated_queries]

            n_k_documents = [task.result() for task in concurrent.futures.as_completed(search_tasks)]
            n_k_documents = utils.misc.flatten(n_k_documents)
            n_k_documents = list(set(n_k_documents))

        logger.info(f"{len(n_k_documents)} documents retrieved successfully."
                    
    
