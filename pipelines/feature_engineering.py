from zenml import pipeline

from steps import feature_engineering as fe_steps

# Creating the pipeline for zenml to execute the full feature engineering process.
@pipeline
def feature_engineering(author_full_names: list[str], wait_for: str | list[str] | None = None) -> list[str]:
    raw_documents = fe_steps.query_data_warehouse(author_full_names, after=wait_for) # Exucute the initial query of the data warehouse in Mongo DB.

    cleaned_documents = fe_steps.clean_documents(raw_documents) # Execute the cleaning of all returned documents from Mongo DB.
    last_step_1 = fe_steps.load_to_vector_db(cleaned_documents) # Execute the loading of the cleaned documents to Qdrant.

    embedded_documents = fe_steps.chunk_and_embed(cleaned_documents) # Executing the chunking and embedding phase of the cleaned documents.
    last_step_2 = fe_steps.load_to_vector_db(embedded_documents) # Executing the loading of the chunked and embedded documents to Qdrant.

    return [last_step_1.invocation_id, last_step_2.invocation_id]