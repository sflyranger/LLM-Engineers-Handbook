from typing import Any

from pydantic import BaseModel
from typing_extensions import Annotated
from zenml import get_step_context, step

#Defining the zenml step to serialize an artifact
@step
def serialize_artifact(artifact: Any, artifact_name:str) -> Annotated[dict, "serialized_artifact"]:
    serialize_artifact = _serialize_artifact(artifact)

    # If None type raise an error
    if serialize_artifact is None: 
        raise ValueError("Artifact is None")
    
    # Create the dictionary object from the artifact data.
    elif not isinstance(serialize_artifact, dict):
        serialized_artifact = {"artifact_data": serialize_artifact}

    # Initialize the step context.
    step_context = get_step_context()
    # Adding the metadata to the context
    step_context.add_output_metadata(output_name="serialized_artifact", metadata={"artifact_name": artifact_name})

    return serialize_artifact

# Internal function to serialize the artifact
def _serialize_artifact(artifact: list | dict | BaseModel | str | int | float | bool | None):

    if isinstance(artifact, list):
        return [_serialize_artifact(item) for item in artifact]
    elif isinstance(artifact, dict):
        return {key: _serialize_artifact(value) for key, value in artifact.items()}
    if isinstance(artifact, BaseModel):
        return artifact.model_dump()
    else:
        return artifact

                                     



    

