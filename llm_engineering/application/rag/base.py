from abc import ABC, abstractmethod
from typing import Any

from langchain.prompts import PromptTemplate
from pydantic import BaseModel

from llm_engineering.domain.queries import Query

# Setting up the prompt template factory to handle the creation of prompts
# This class will inherit properties and methods from both the abstract base class and the BaseModel class from pydantic.
class PromptTemplateFactory(ABC, BaseModel):
    # subclasses must instatiate their own version of this method
    @abstractmethod
    def create_template(self) -> PromptTemplate:
        pass
# Creating the RAGStep class
class RAGStep(ABC):
    def __init__(self, mock: bool = False) -> None:
        self._mock = mock
    
    @abstractmethod
    # Generate method for the class
    def generate(self, query: Query, *args, **kwargs) -> Any:
        pass