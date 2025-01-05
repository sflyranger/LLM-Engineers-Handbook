from typing import Any

from typing_extensions import Annotated
from zenml import get_step_context, step, ArtifactConfig

from llm_engineering.application.dataset import generation
from llm_engineering.domain.dataset import DatasetType, InstructTrainTestSplit
from llm_engineering.domain.prompt import GenerateDatasetSamplesPrompt
from llm_engineering.domain.types import DataCategory

# Create the zenml step to generate the instruction datasets.
@step
def generate_instruction_dataset(
    prompts: Annotated[dict[DataCategory, list[GenerateDatasetSamplesPrompt]], "prompts"], 
    test_split_size: Annotated[float, "test_split_size"],
    mock: Annotated[bool, "mock_generation"] = False, 
)-> Annotated[
    InstructTrainTestSplit, 
    ArtifactConfig(
        name="instruct_datasets", 
        tags=["dataset", "instruct", "cleaned"]
    ),
]:
    # Define the generator based on the type.
    dataset_generator = generation.get_dataset_generator(DatasetType.INSTRUCTION)
    # Generate the datasets
    datasets = dataset_generator.generate(prompts, test_size=test_split_size, mock=mock)

    # Initialize the zenml step context.
    step_context = get_step_context()

    # Add the metadata to the step context.
    step_context.add_output_metadata(output_name="instruct_datasets" ,metadata=_get_metadata_instruct_dataset(datasets))

    return datasets 

def _get_metadata_instruct_dataset(datasets: InstructTrainTestSplit) -> dict[str, Any]:

    # Get the categories from the training set of keys.
    instruct_dataset_categories = list(datasets.train.keys())
    # Get the number of samples for each category in each dataset from the training set..
    train_num_samples = {
        category: instruct_dataset.num_samples for category, instruct_dataset in datasets.train.items()
    }
    # Get the number of samples for each category in each dataset for the testing set.
    test_num_samples = {
        category: instruct_dataset.num_samples for category, instruct_dataset in datasets.test.items()
    }

    return {
        "data_categories": instruct_dataset_categories, 
        "test_split_size": datasets.test_split_size,
        "train_num_samples_per_category": train_num_samples,
        "test_num_samples_per_category": test_num_samples
    }
