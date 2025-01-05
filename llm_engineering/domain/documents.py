from abc import ABC 
from typing import Optional

from pydantic import UUID4, Field
from .base import NoSQLBaseDocument
from .types import DataCategory

# Setting up the UserDocument Class, inheriting from the NoSQLBaseDocument class 
class UserDocument(NoSQLBaseDocument):
    first_name : str
    last_name : str

    class Settings:
        name = "users"
    
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

class Document(NoSQLBaseDocument, ABC):
    content : dict # Setting the interior content as a dictionary
    platform: str # Platform dtype is a str
    author_id: UUID4 = Field(alias = "author_id") # Sets up the author_id as a Field aliased as "author_id" in UUID4 format
    author_full_name: str  = Field(alias = "author_full_name") # Sets up the author_full_name as a Field aliased as "author_full_name" in str format


# Setting up the repository class inheriting from the Document class
class RepositoryDocument(Document):
    name: str
    link: str

    class Settings:
        name = DataCategory.REPOSITORIES  # making a setting that points to the REPOSITORIES type from the types.DataCategory variable

# Setting up the post Document class that inherits from the Document class
class PostDocument(Document):
    image: Optional[str] = None # Image is optional within the class
    link: str | None = None # the link format can be a string or None type

    class Settings:
        name  = DataCategory.POSTS # making a setting that points to the POSTS type from .types.DataCategory variable

class ArticleDocument(Document):
    link : str # link in string format

    class Settings:
        name = DataCategory.ARTICLES # making a setting that points to the ARTICLES type from .types.DataCategory