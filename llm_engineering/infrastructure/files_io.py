import json
from pathlib import Path


# Creating the JsonFileManager class to load json based documents
class JsonFileManager:
    # Class method to attempt to read the json formatted file
    @classmethod
    def read(cls, filename:str | Path)-> list:
        file_path: Path = Path(filename)

        try:
            # Try to read the file
            with file_path.open("r") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_path=}' does not exist.") from None
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                msg=f"File '{file_path}' is not properly formatted as JSON.",
                doc=e.doc,
                pos=e.pos,
            ) from None
    # Class method to attempt to write the json formatted file.
    @classmethod
    def write(cls, filename: str | Path, data: list | dict) -> Path:
        # Get the path using the Path function
        file_path: Path = Path(filename)

        # Resolve the abolute path.
        file_path = file_path.resolve().absolute()

        # Make the parent directories.
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the json file.
        with file_path.open("w") as file:
            json.dump(data, file, indent=4)
        
        return file_path
    