from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from llm_engineering.settings import settings


class QdrantDatabaseConnector:
    """
    Returns an instance of a Qdrant DB connection via either cloud url or host/port settings.
    """
    _instance: QdrantClient | None = None

    def __new__(cls, *args, **kwargs)-> QdrantClient:
        if cls._instance is None:
            try: # trying to connect and create a qdrant cloud instance if in settings
                if settings.USE_QDRANT_CLOUD:
                    cls._instance = QdrantClient(
                        url=settings.QDRANT_CLOUD_URL, 
                        api_key=settings.QDRANT_APIKEY, 
                    )

                    # local connection to view the connection via the cloud
                    uri = settings.QDRANT_CLOUD_URL
                else: # creating a local instance if cloud settings not present
                    cls._instance = QdrantClient(
                        host=settings.QDRANT_DATABASE_HOST,
                        port=settings.QDRANT_DATABASE_PORT
                    )

                    uri = f"{settings.QDRANT_DATABASE_HOST}:{settings.QDRANT_DATABASE_PORT}"

                logger.info(f"Connection to Qdrant DB with uri successful: {uri}")
            except UnexpectedResponse:
                logger.exception(
                    "Couldn't connect to Qdrant.",
                    host=settings.QDRANT_DATABASE_HOST, 
                    port=settings.QDRANT_DATABASE_PORT,
                    url=settings.QDRANT_CLOUD_URL
                )

                raise 


        return cls._instance
    
connection = QdrantDatabaseConnector()