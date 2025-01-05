from loguru import logger 

from llm_engineering.domain.base import NoSQLBaseDocument, VectorBaseDocument
from llm_engineering.domain.types import DataCategory 

from .chunking_data_handlers import (
    ArticleChunkingHandler, 
    ChunkingDataHandler, 
    PostChunkingHandler, 
    RepositoryChunkingHandler,
)

from .cleaning_data_handlers import (
    ArticleCleaningHandler, 
    CleaningDataHandler, 
    PostCleaningHandler, 
    RepositoryCleaningHandler,
)

from .embedding_data_handlers import (
    ArticleEmbeddingHandler, 
    EmbeddingDataHandler, 
    PostEmbeddingHandler, 
    QueryEmbeddingHandler, 
    RepositoryEmbeddingHandler,
)


class CleaningHandlerFactory:
    """
    Primary Class that will handle all cleaning data handlers.

    """
    @staticmethod
    def create_handler(data_category: DataCategory) -> CleaningDataHandler:
        if data_category==DataCategory.POSTS:
            return PostCleaningHandler()
        elif data_category==DataCategory.ARTICLES:
            return ArticleCleaningHandler()
        elif data_category==DataCategory.REPOSITORIES:
            return RepositoryCleaningHandler()
        else:
            raise ValueError("Unsupported data type.")


class CleaningDispatcher:
    """
    Dispatches the CleaningHandlerFactory to execute cleaning for each data category.
    """

    factory = CleaningHandlerFactory()


    @classmethod
    def dispatch(cls, data_model: NoSQLBaseDocument) -> VectorBaseDocument:
        data_category = DataCategory(data_model.get_collection_name()) # Finds the data category for the collection name upon search.
        handler= cls.factory.create_handler(data_category) # Creates the data handler class.
        clean_model = handler.clean(data_model) # Runs the cleaning on the documents.

        logger.info(
            "Document Cleaned Successfully.", 
            data_category=data_category, 
            cleaned_content_len=len(clean_model.content),
        )

        return clean_model

class ChunkingHandlerFactory:
    """
    Primary Class that will handle all of the chunking data handlers.

    """

    @staticmethod
    def create_handler(self, data_model: DataCategory) -> ChunkingDataHandler:
    if data_category==DataCategory.POSTS:
        return PostChunkingHandler()
    elif data_category==DataCategory.ARTICLES:
        return ArticleChunkingHandler()
    elif data_category==DataCategory.REPOSITORIES:
        return RepositoryChunkingHandler()
    else:
        raise ValueError("Unsupported data type.")

class ChunkingDispatcher:
    """
    Dispatches the ChunkingHandlerFactory to execute chunking for each data category.

    """

    factory = ChunkingHandlerFactory()

    @classmethod
    def dispatch(cls, data_model: VectorBaseDocument) -> list[VectorBaseDocument]:
        data_category = data_model.get_category() # Getting the data category for the documents.
        handler = cls.factory.create_handler(data_category) # Creating the handler based on the data category.
        chunk_models = handler.chunk(data_model) # Perform chunking on the documents.

        logger.info(
            "Document Chunked successfully.", 
            num=len(chunk_models), 
            data_category=data_category
        )

        return chunk_models


class EmbeddingHandlerFactory:
    @staticmethod
    def create_handler(data_category: DataCategory) -> EmbeddingDataHandler:
        if data_category==DataCategory.QUERIES:
            return QueryEmbeddingHandler()
        elif data_category==DataCategory.POSTS:
            return PostEmbeddingHandler()
        elif data_category==DataCategory.ARTICLES:
            return ArticleEmbeddingHandler()
        elif data_category==DataCategory.REPOSITORIES:
            return RepositoryEmbeddingHandler()
        else:
            raise ValueError("Unsupported data type.")
        
class EmbeddingDispatcher:
    """
    Dispatches the EmbeddingHandlerFactory to execute the embeddings for each data category.
    
    """
    
    factory = EmbeddingHandlerFactory()

    @classmethod
    def dispatch(cls, data_model: VectorBaseDocument | list[VectorBaseDocument]) -> VectorBaseDocument | list[VectorBaseDocument]:
        is_list = isinstance(data_model, list) # Setting the list status of the model.
        if not is_list:
            data_model = [data_model] # Changing to a list if not a list.

        # Returning an empty list if there are no documents in the model
        if len(data_model) == 0:
            return []

        data_category = data_model[0].get_category()
        assert all(
            data_model.get_category() == data_category for data_model in data_model # Ensure all models are of the same category.
        ), "Data models must be of the same category.", 
        handler = cls.factory.create_handler(data_category) # Creating the handler for the given data category.

        embedded_chunk_model = handler.embed_batch(data_model) # Getting the embedded chunks for the data model.

        if not is_list:
            embedded_chunk_model = embedded_chunk_model[0]

        logger.info(
            "Data embedded succesfully.", 
            data_category=data_category
        )

        return embedded_chunk_model


