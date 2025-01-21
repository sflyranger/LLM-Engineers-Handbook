from zenml import pipeline

from .digital_data_etl import digital_data_etl
from .feature_engineering import feature_engineering
from .generate_datasets import generate_datasets

# The pipeline below is not optimal.
# The reason why is that it creates a single monolithic sequence of pipelines.
# It serves as an example of the possible usage of zenml to run multiple pipelines in sequence.
# In my project I will not do this, I will instead manually trigger each of the pipelines manually through their individual poetry commands.

# Defining the zenml pipeline step that sequentially runs all of the data pipelines
# 1st we run the digital_data_etl pipeline to pull in data into mongodb.
# 2nd we run the feature_engineering pipeline to clean, chunk and embed the documents from mongo db, then store them in Qdrant.
# Lastly we run the generate_datasets pipeline to create prompts, and generate instruction and preference datasets.
@pipeline
def end_to_end_data(
    author_links: list[dict[str, str | list[str]]],
    test_split_size: float = 0.1, 
    push_to_huggingface: bool = False,
    dataset_id: str | None = None, 
    mock: bool = False,
)-> None:
    # Empty list to store the pipeline to wait for.
    wait_for_ids = []

    # Loop through each link perform etl to mongodb.
    for author_data in author_links:
        last_step_invocation_id = digital_data_etl(
            user_full_name=author_data["user_full_name"], links=author_data["links"]
        )

        # Append the digital_data_etl pipeline to the list to wait for.
        wait_for_ids.append(last_step_invocation_id)

    # Get the author_full_names from each link
    author_full_names = [author_data["user_full_name"] for author_data in author_links]
    # Run the feature engineering pipeline waiting on the etl pipeline to finish.
    wait_for_ids = feature_engineering(author_full_names=author_full_names, wait_for=wait_for_ids)

    # Run the generate_datasets pipeline waiting for the feature engineering pipeline to finish.
    generate_datasets(
        test_split_size=test_split_size, 
        push_to_huggingface=push_to_huggingface, 
        dataset_id=dataset_id, 
        mock=mock, 
        wait_for=wait_for_ids,
    )

