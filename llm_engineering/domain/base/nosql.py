import uuid 
from abc import ABC
from typing import Generic, Type, TypeVar

from loguru import logger
from pydantic import UUID4, BaseModel, Field
from pymongo import errors

from llm_engineering.domain.exceptions import ImproperlyConfigured
from llm_engineering.infrastructure.db.mongo import connection # pulls from the mongo.py file to pull out the mongodb connection
from llm_engineering.settings import settings

_database = connection.get_database(settings.DATABASE_NAME) # makes the database set to the database name stored in the settings.py file

T = TypeVar("T", bound = "NoSQLBaseDocument") # "T" is the name for the typevar while the bound clause specifies that T must be a subtype of the NoSQLBaseDocument class.

class NoSQLBaseDocument(BaseModel, Generic[T], ABC):
    id: UUID4 = Field(default_factory = uuid.uuid4)

    # equality method to define how objects should be compared
    def __eq__(self, value:object)-> bool:
        # return False if not value in the NoSQLBaseDocument class
        if not isinstance(value, self.__class__):
            return False
        # two entities are equal if both are instances of the same class and their id attributes are equal. 
        # this means that documents have to be of the same type and id, ensures consistency
        return self.id == value.id #
    
    # defining the hash method
    # here we generate a hash value based on the objects id attribute which in this case is the UUID4 
    def __hash__(self)->int:
        return hash(self.id)
    
    @classmethod
    # pulling from mongo and setting the classes
    def from_mongo(cls: Type[T], data:dict)->T:
        """ Convert "_id" (str object) into "id" (UUID object)"""
        
        # Checks to see if data is present
        if not data:
            raise ValueError("Data is empty.")
        # removes the _id field in the dict and resets it to id
        id = data.pop("_id")

        # returns an instance of the class from the data with the id field converted 
        return cls(**dict(data, id = id))

    def to_mongo(self:T, **kwargs) -> dict:
        """ Convert the "id" (UUID object) to "_id" (str object)"""

        # includes all fields including the unset fields
        exclude_unset = kwargs.pop("exclude_unset", False)

        # setting the alias names from the fields
        by_alias = kwargs.pop("by_alias", True)

        parsed = self.model_dump(exclude_unset = exclude_unset, by_alias = by_alias, **kwargs)
        
        # checking to see if the "_id" is in parsed, if it isn't and "id" is,
        # we set the "_id" to be equal to the str version of the removed "id"
        if "_id" not in parsed and "id" in parsed:
            parsed["_id"] = str(parsed.pop("id"))
        # loops through the parsed items and if the value is the UUID we set the parsed key to be the str version of the value            
        for key, value in parsed.items():
            if isinstance(value, uuid.UUID):
                parsed[key] = str(value)
        
        # returning the parsed object 
        return parsed

    def model_dump(self: T, **kwargs) -> dict:
        dict_ = super().model_dump(**kwargs)

        for key, value in dict_.items():
            if isinstance(value, uuid.UUID):
                dict_[key] = str(value)
            
            return dict_
    def save(self:T, **kwargs) -> T | None:
        # setting the collection as the name of the current collection
        collection = _database[self.get_collection_name()]
        
        # attempting to send to mongo via the to_mongo function
        try:
            collection.insert_one(self.to_mongo(**kwargs))

            return self
        # if theres a WriteError output an error message to the logger
        except errors.WriteError:
            logger.exception("Failed to insert document.")

            return None
    
    # multiple classmethod instances allow that each method is accesible without requiring the instance of a class
    @classmethod
    # this function serves to either retrieve an existing NoSQL document or create a new one if no mathing doc exists
    def get_or_create(cls: Type[T], **filter_options)-> T:
        collection = _database[cls.get_collection_name()] #cls here used to pull from the class instead of self from the prior method
        try:
            instance = collection.find_one(filter_options) #executing a query to find the document
            if instance:
                return cls.from_mongo(instance) # returning the instance using the from_mongo function

            new_instance = cls(**filter_options) # getting a new instance from the class
            new_instance = new_instance.save() # saving the new instance

            return new_instance # returning the new_instance
        except errors.OperationFailure: # error logging
            logger.exception(f"Failed to retrieve document with filter options {filter_options}")

            raise
    
    @classmethod
    # function to insert multiple documents into the NoSQL mongodb
    def bulk_insert(cls: Type[T], documents:list[T], **kwargs) -> bool:
        collection = _database[cls.get_collection_name()] # pulling the collection form the class or subclass
        try:
            collection.insert_many(doc.to_mongo(**kwargs) for doc in documents)

            return True
        except (errors.WriteError, errors.BulkWriteError):
            logger.error(f"Failed to insert documents of type: {cls.__name__}")

            return False
    
    @classmethod
    # method to find documents of the given class type
    def find(cls: Type[T], **filter_options) -> T | None:
        collection = _database[cls.get_collection_name()]
        try:
            instance = collection.find_one(filter_options)
            if instance:
                return cls.from_mongo(instance) # if found we return the instance via the from_mongo function

        except errors.OperationFailure:
            logger.error("Failed to retrieve document.")

            return None
    
    @classmethod 
    # method to find documents of one type in bulk
    def bulk_find(cls:Type[T], **filter_options)-> list[T]:
        collection = _database[cls.get_collection_name()]
        try:
            instances = collection.find(filter_options)
            return [document for instance in instances if (document := cls.from_mongo(instance)) is not None] # returns each document if the returned mongo instance is not None
        
        except errors.OperationFailure:
            logger.error("Failed to retrieve documents.")

            return [] # returning an empty list is documents cannot be retrieved.

    @classmethod
    # method to get the collection name from the class
    def get_collection_name(cls: Type[T])-> str:
        if not hasattr(cls, "Settings") or not hasattr(cls.Settings, "name"):
            raise ImproperlyConfigured(
                "Document should define a Settings configuration class with the name of the collection."
            )
        return cls.Settings.name



