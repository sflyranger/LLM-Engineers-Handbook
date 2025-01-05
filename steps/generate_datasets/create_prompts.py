from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application.dataset import generation
from llm_engineering.domain.dataset import DatasetType
from llm_engineering.domain.prompt import GenerateDatasetSamplesPrompt
from llm_engineering.domain.types import DataCategory

# Defining the zenml step to create the neccessary prompts.
@step
def create_prompts(
    documents: Annotated[list, "queried_cleaned_documents"], 
    dataset_type: Annotated[DatasetType, "dataset_type"], 
)-> Annotated[dict[DataCategory, list[GenerateDatasetSamplesPrompt]], "prompts"]:
    
    # Intitialize the generator based on the type.
    dataset_generator = generation.get_dataset_generator(dataset_type)
    # Group prompts by the document type.
    grouped_prompts = dataset_generator.get_prompts(documents)

    # Initialize the step context.
    step_context = get_step_context()

    # Add the metadata to the step context for the groups.
    step_context.add_output_metadata(output_name="prompts", metadata=_get_metadata(grouped_prompts))
    
    return grouped_prompts

def _get_metadata(grouped_prompts: dict[DataCategory, list[GenerateDatasetSamplesPrompt]]) -> dict:
    
    prompt_categories = list[grouped_prompts.keys()] # Pull the category key from the grouped prompts
    prompt_num_samples = {category: len(prompts) for category, prompts in grouped_prompts.items()} # Count the prompts for each category.

    return {"data_categories": prompt_categories, "data_categories_num_samples": prompt_num_samples} # Output in dictionary format.

