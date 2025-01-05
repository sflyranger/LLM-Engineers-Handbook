from llm_engineering.domain.base import VectorBaseDocument
from llm_engineering.domain.cleaned_documents import CleanedDocument
from llm_engineering.domain.types import DataCategory 

# Creating the prompt class that inherits from the VectorBaseDocument class
class Prompt(VectorBaseDocument):
    template: str
    input_variables: dict 
    content: str 
    num_tokens: int | None = None 

    class Config:
        category = DataCategory.PROMPT

# Subclass to generate samples
class GenerateDatasetSamplesPrompt(Prompt):
    data_category: DataCategory 
    document: CleanedDocument 