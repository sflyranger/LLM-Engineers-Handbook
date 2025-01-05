from loguru import logger
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from llm_engineering.settings import settings

# Setting up the MongoDatabaseConnector Class to connect to mongodb
class MongoDatabaseConnector:
    _instance: MongoClient | None = None

    def __new__(cls, *args, **kwargs) -> MongoClient:
        if cls._instance is None:
            try:
                cle._instance = MongoClient(settings.DATABASE_HOST)
            except ConnectionFailure as e:
                logger.error(f"Couldn't connect to the database: {e!s}")

                raise
            
            logger.info(f"Connection to MongoDB with URI successful: {settings.DATABASE_HOST}")

            return cls._instance

connection = MongoDatabaseConnector()

