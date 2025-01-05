from loguru import logger 
from typing_extensions import Annotated
from zenml import step

from llm_engineering.domain.dataset import InstructTrainTestSplit, PreferenceTrainTestSplit
from llm_engineering.settings import settings 

# Create the zenml step to push the generated datasets to huggingface.
@step
def push_to_huggingface(
    dataset: Annotated[InstructTrainTestSplit | PreferenceTrainTestSplit, "dataset_split"],
    dataset_id: Annotated[str, "dataset_id"], 
)-> None:
    assert dataset_id is not None, "Dataset id must be provided to push to Huggingface"
    assert (
        settings.HUGGINGFACE_ACCESS_TOKEN is not None
    ), "Huggingface access token must be provided to push to Huggingface."

    logger.info(f"Pushing dataset {dataset_id} to Huggingface.")

    # Push the dataset.
    huggingface_dataset = dataset.to_huggingface(flatten=True)
    # Ensure token access and push to the hub.
    huggingface_dataset.push_to_hub(dataset_id, token=settings.HUGGINGFACE_ACCESS_TOKEN)
    