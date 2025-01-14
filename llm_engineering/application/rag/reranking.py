import opik

from llm_engineering.application.networks import CrossEncoderModelSingleton
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.domain.queries import Query


from .base import RAGStep

# Creating the reranker to rank the best responses.
class Reranker(RAGStep):
    def __init__(self, mock: bool=False) -> None:
        super().__init__(mock=mock)  # Ensure inherited mock setting.

        # Initialize the cross-encoder model
        self._model = CrossEncoderModelSingleton()

    # Tracking the status of the Reranker.generate process
    @opik.track(name="Reranker.generate")
    def generate(self, query: Query, chunks: list[EmbeddedChunk], keep_top_k: int) -> list[EmbeddedChunk]:
        # Return mock queries if in mock mode
        if self._mock:
            return chunks
        
        # Create tuples of query and chunk content
        query_doc_tuples = [(query.content, chunk.content) for chunk in chunks]
        # Get scores for each query-chunk pair
        scores = self._model(query_doc_tuples)

        # Zip scores with chunks and sort by score in descending order
        scored_query_doc_tuples = list(zip(scores, chunks, strict=False))
        scored_query_doc_tuples.sort(key=lambda x: x[0], reverse=True)

        # Keep only the top k chunks
        reranked_documents = scored_query_doc_tuples[:keep_top_k]
        reranked_documents = [doc for _, doc in reranked_documents]

        return reranked_documents