from llm_engineering.domain.dataset import DatasetType

# The Mocked responses here serve as formats for JSON-based responses
# This serves to integrate properly with the structure used to generate responses and store them properly for my class structure.

MOCKED_RESPONSE_INSTRUCT = """
[
    {"instruction": "<mocked generated instruction> 1", "answer": "mocked generated answer> 1"}, 
    {"instruction": "<mocked generated instruction> 2", "answer": "mocked generated answer> 2"},
    {"instruction": "<mocked generated instruction> 3", "answer": "mocked generated answer> 3"}
]"""

MOCKED_RESPONSE_PREFERENCE = """
[
    {"instruction": "<mocked generated instruction> 1", "rejected": "mocked generated rejected> 1"}, 
    {"instruction": "<mocked generated instruction> 2", "rejected": "mocked generated rejected> 2"},
    {"instruction": "<mocked generated instruction> 3", "rejected": "mocked generated rejected> 3"}
]"""

# Function to get the mocked response based on the dataset type
def get_mocked_response(dataset_type: DatasetType) -> str:
    if dataset_type == DatasetType.INSTRUCTION:
        return MOCKED_RESPONSE_INSTRUCT
    elif dataset_type == DatasetType.PREFERENCE:
        return MOCKED_RESPONSE_PREFERENCE
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")