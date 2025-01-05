from sklearn.model_selection import train_test_split 

from llm_engineering.application.preprocessing.operations.chunking import chunk_document
from llm_engineering.domain.cleaned_documents import CleanedDocument
from llm_engineering.domain.dataset import (
    InstructDataset, 
    InstructDatasetSample, 
    InstructTrainTestSplit, 
    PreferenceDataset,
    PreferenceDatasetSample, 
    PreferenceTrainTestSplit
)

from llm_engineering.domain.types import DataCategory 

def create_instruct_train_test_split(
    data: dict[DataCategory, InstructDataset], test_size = 0.2, random_state=42
) -> InstructTrainTestSplit:
     # Initialize empty dictionaries for the train and test data.
    train_data = {}
    test_data = {}

    for category, dataset in data.items():
        samples = dataset.samples
        samples_dicts = [sample.model_dump() for sample in samples]

        # If samples are present perform the split.
        if len(samples_dicts) > 0:
            train_samples_dicts, test_samples_dicts = train_test_split(
                samples_dicts, test_size=test_size, random_state=random_state
            )

            # Map each sample to the correct sample set.
            train_samples = [InstructDatasetSample(**sample_dict) for sample_dict in train_samples_dicts]
            test_samples = [InstructDatasetSample(**sample_dict) for sample_dict in test_samples_dicts]
        
        # If no samples are present create the instruction dataset without a split, leave empty
        else:
            train_samples = []
            test_samples = []

            train_samples = InstructDataset(category=category, samples=train_samples)
            test_samples = InstructDataset(category=category, samples=test_samples)

    # Store in huggingface after split    
    return InstructTrainTestSplit(train=train_data, test=test_data, test_split_size=test_size)


def create_preference_train_test_split(
    data: dict[DataCategory, PreferenceDataset], test_size=0.2, 
random_state=42
) -> PreferenceTrainTestSplit:
    # Initialize empty dictionaries for the train and test data.
    train_data = {}
    test_data = {}

    for category, dataset in data.items():
        samples = dataset.samples
        samples_dicts = [sample.model_dump() for sample in samples]

        # If samples are present perform the split.
        if len(samples_dicts) > 0:
            train_samples_dicts, test_samples_dicts = train_test_split(
                samples_dicts, test_size=test_size, random_state=random_state
            )

            # Map each sample to the correct sample set.
            train_samples = [PreferenceDatasetSample(**sample_dict) for sample_dict in train_samples_dicts]
            test_samples = [PreferenceDatasetSample(**sample_dict) for sample_dict in test_samples_dicts]
        
        # If no samples are present create the preference dataset without a split, leave empty
        else:
            train_samples = []
            test_samples = []

        train_data[category] = PreferenceDataset(category=category, samples=train_samples)
        test_data[category] = PreferenceDataset(category=category, samples=test_samples)

    # Store in huggingface after split    
    return PreferenceTrainTestSplit(train=train_data, test=test_data, test_split_size=test_size)

def filter_short_answers(
        data: dict[DataCategory, PreferenceDataset], min_length: int = 100
) -> dict[DataCategory, PreferenceDataset]:
    # Check to see if the chosen example meets the minimum length requirement.
    def is_long_enough(example: PreferenceDatasetSample) -> bool:
        return len(example.chosen) >= min_length
    
    # Empty dictionary for the filtered preferences.
    filtered_data = {}
    for category, dataset in data.items():
        # Only select those samples that meet the minimum length.
        filtered_dataset_samples = list(filter(is_long_enough, dataset.samples))

        # Make the preference dataset.
        filtered_dataset = PreferenceDataset(category=category, samples=filtered_dataset_samples)
        
        # Store the filtered data for the given category.
        filtered_data[category] = filtered_dataset
    
    return filtered_data


def filter_answer_format(data: dict[DataCategory, PreferenceDataset]) -> dict[DataCategory, PreferenceDataset]:
    # Function to check if the format is valid.
    def is_valid_format(example: PreferenceDatasetSample) -> bool:
        chosen = example.chosen

        # Chosen examples need to be present, have an initial character that is uppercase and end with proper punctuation.
        return len(chosen) > 0 and chosen[0].isupper() and chosen[-1] in (".", "!", "?")
    # Empty dict for storage.
    filtered_data = {}
    for category, dataset in data.items():
        # Only select those samples that meet the format requirements.
        filtered_dataset_samples = list(filter(is_valid_format, dataset.samples))

        # Make the preference dataset.
        filtered_dataset = PreferenceDataset(category=category, samples=filtered_dataset_samples)
        
        # Store the filtered data for the given category.
        filtered_data[category] = filtered_dataset
    
    return filtered_data

# Function to extract the substrings from a list of cleaned documents. The min_length and max_length of the character list can be experimented with to get the best results.
def extract_substrings(
        documents: list[CleanedDocument], min_length: int = 1000, max_length: int = 2000
) -> list[CleanedDocument]:
    # Empty list for storage
    extracts = []

    # Create the extracts and make substrings of smaller chunks of text
    for document in documents:
        document_extracts = chunk_document(document.content, min_length, max_length)
        for extract in document_extracts:
            subdocument = document.model_copy() # Make a copy of the Class
            subdocument.content = extract # Store the extracted content in the copy
            
            # add the substrings to the extracts list
            extracts.append(subdocument)

    return extracts




