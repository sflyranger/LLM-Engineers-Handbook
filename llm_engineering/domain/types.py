from enum import StrEnum

# setting up the DataCategory class that inherits from StrEnum class
# each of the members below will be recognized as both a string and an enumeration
# useful for applications where categories need to be referenced consistently by name but also used as strings in dbs, APIs etc.
class DataCategory(StrEnum): 
    PROMPT  = "prompt" 
    QUERIES = "queries"

    INSTRUCT_DATASET_SAMPLES = "instruct_dataset_samples"
    INSTRUCT_DATASET = "instruct_dataset"
    PREFERENCE_DATASET_SAMPLES = "preference_dataset_samples"
    PREFERENCE_DATASET = "preference_dataset"

    POSTS = "posts"
    ARTICLES = "articles"
    REPOSITORIES = "repositories"
    