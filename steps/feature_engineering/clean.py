from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application.preprocessing import CleaningDispatcher
from llm_engineering.domain.cleaned_documents import CleanedDocument 

# Zenml step to clean all documents.
@step 
def clean_documents(
    documents = Annotated[list, "raw_documents"], 
)-> Annotated[list, "cleaned_documents"]:

# Initialize empty list.
cleaned_documents = []

# Iterate over documents, clean each and append to the cleaned_documents list.
for document in documents:
    cleaned_document = CleaningDispatcher.dispatch(document)
    cleaned_documents.append(cleaned_document)

# Intitialize the step context.
step_context = get_step_context()

# Add the metadata for the output to the step context.
step_context.add_output_metadata(output_name="cleaned_documents", metadata=_get_metadata(cleaned_documents))

return cleaned_documents 


def _get_metadata(cleaned_documents: list[CleanedDocument]) -> dict:
    """
    Returns the metadata as a dictionary for the cleaned documents.
    
    """

    metadata = {"num_documents": len(cleaned_documents)}

    for document in cleaned_documents:
        category = document.get_category()

        if category not in metadata:
            metadata[category] = {} # If no metadata keep the dictionary empty.
        if "authors" not in metadata[category]:
            metadata[category]["authors"] = list() # Place an empty list at "authors" location in the metadata for the category.
        
        metadata[category]["num_documents"] = metadata[category].get("num_documents", 0) + 1
        metadata[category]["authors"].append(document.author_full_name)

    for value in metadata.values():
        if isinstance(value, dict) and "authors" in value:
            value["authors"] = list(set(value["authors"])) # Make "authors" a list of a set of the "authors" (Doesn't let the authors repeat).
        
    return metadata