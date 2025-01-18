from pathlib import Path

from typing_extensions import Annotated
from zenml import step

from llm_engineering.infrastructure.files_io import JsonFileManager

# Defining the zenml step to export the serialized artifact to json
@step
def to_json(
    data: Annotated[dict, "serialized_artifact"], 
    to_file: Annotated[Path, "to_file"], 
)-> Annotated[Path, "exported_file_path"]:
    absolute_file_path = JsonFileManager.write(
        filename=to_file, 
        data=data,
    )

    return absolute_file_path