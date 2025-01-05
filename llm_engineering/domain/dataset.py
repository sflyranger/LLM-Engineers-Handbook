from enum import Enum

from loguru import logger 

try:
    from datasets import Dataset, DatasetDict, concatenate_datasets
except ImportError:
    logger.warning("Huggingface datasets not installed. Install with `pip install datasets`")

from llm_engineering.domain.base import VectorBaseDocument
from llm_engineering.domain.types import DataCategory

# Setting the DatasetType class, inheriting from Enum
class DatasetType(Enum):
    INSTRUCTION = "instruction"
    PREFERENCE = "preference"


class InstructDatasetSample(VectorBaseDocument):
    instruction: str 
    answer: str 

    class Config:
        category = DataCategory.INSTRUCT_DATASET_SAMPLES


class PreferenceDatasetSample(VectorBaseDocument):
    instruction: str 
    rejected: str 
    chosen: str 

    class Config:
        category = DataCategory.PREFERENCE_DATASET_SAMPLES

class InstructDataset(VectorBaseDocument):
    category : DataCategory 
    samples: list[InstructDatasetSample]

    class Config:
        category = DataCategory.INSTRUCT_DATASET

    @property
    def num_samples(self) -> int:
        return len(self.samples)
    
    def to_huggingface(self) -> "Dataset":
        data = [sample.model_dump() for sample in self.samples]

        # Set the format of the dataset object in huggingface.
        return Dataset.from_dict(
            {"instruction": [d["instruction"] for d in data], "output": [d["answer"] for d in data]}
        )

# Class to handle the train test split of the datasets in huggingface.
class TrainTestSplit(VectorBaseDocument):
    train: dict
    test: dict
    test_split_size: dict 

    def to_huggingface(self, flatten: bool = False) -> "DatasetDict":
        # Pull the training items, for the given categories in each dataset, push the values into hugging face via the to_huggingface function.
        train_datasets = {category.value: dataset.to_huggingface() for category, dataset in self.train.items()}
        # Do the same for the test items.
        test_datasets = {category.value: dataset.to_huggingface() for category, dataset in self.test.items()}
        
        # Functionality if the flatten parameter is true
        if flatten:
            # Just list the values from the datasets concatenated together in a list.
            train_datasets = concatenate_datasets(list(train_datasets.values()))
            test_datasets = concatenate_datasets(list(test_datasets.values()))

        # If flatten is false (default) we create the dataset from the given dictionaries for the datasets
        else:
            train_datasets = Dataset.from_dict(train_datasets)
            test_datasets = Dataset.from_dict(test_datasets)
        
        # Stitch the two dataset objects together into one DatasetDict object
        return DatasetDict({"train": train_datasets, "test": test_datasets})

# Subclass for instruction datasets based on the TrainTestSplit functionality.
class InstructTrainTestSplit(TrainTestSplit):
    train: dict[DataCategory, InstructDataset]
    test: dict[DataCategory, InstructDataset]
    test_split_size: float

    class Config:
        category = DataCategory.INSTRUCT_DATASET
    

class PreferenceDataset(VectorBaseDocument):
    category: DataCategory
    samples: list[PreferenceDatasetSample]

    class Config:
        DataCategory.PREFERENCE_DATASET
    
    @property
    def num_samples(self) -> int:
        return len(self.samples)
    
    def to_huggingface(self) -> "Dataset":
        data = [sample.model_dump() for sample in self.samples]

        # Return the huggingface dataset from the dictionary for each item based on the class 
        return Dataset.from_dict(
            {
                "prompt": [d["instruction"] for d in data], 
                "rejected": [d["rejected"] for d in data], 
                "chosen": [d["chosen"] for d in data],
            }
        )

# Subclass for preference datasets with TrainTestSplit class functionality.
class PreferenceTrainTestSplit(TrainTestSplit):
    train: dict[DataCategory, PreferenceDataset]
    test: dict[DataCategory, PreferenceDataset]
    test_split_size: float

    class Config:
        category = DataCategory.PREFERENCE_DATASET

# Final function call the building of each class based on the input datasets.
def build_dataset(dataset_type, *args, **kwargs) -> InstructDataset | PreferenceDataset:
    if dataset_type == DatasetType.INSTRUCTION:
        return InstructDataset(*args, **kwargs) # Create the InstructDataset based on the class settings.
    elif dataset_type == DatasetType.PREFERENCE:
        return PreferenceDataset(*args, **kwargs) # Create the PreferenceDataset based on the class settings.
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")


