import os
import shutil
import subprocess
import tempfile

from loguru import logger

from llm_engineering.domain.documents import RepositoryDocument

from .base import BaseCrawler

# Setting the GithubCrawler class, inheriting from the BaseCrawler class
class GithubCrawler(BaseCrawler):
    model = RepositoryDocument # setting the model attribute to be the RepositoryDocument class

    # setting the initialization with settings to ignore specific file types from the repo
    def __init__(self, ignore=(".git", ".toml", ".lock", ".png")) -> None:
        super().__init__() # inheriting properties from BaseCrawler
        self._ignore = ignore # ignoring files
    
    def extract(self, link:str, **kwargs) -> None:
        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"Repository already exists in the database: {link}")

            return
        
        logger.info(f" Starting scrapping of Github repository: {link}")

        # exteracting the repo name from the URL, stripping the intenal "/" and ending "/" to just output the repo name
        repo_name = link.rstrip("/").split("/")[-1]

        # making a temporary directory to clone the repo
        local_temp = tempfile.mkdtemp()

        try:
            os.chdir(local_temp) # changing the working directory to the temp directory
            subprocess.run(["git", "clone", link]) # cloning the repo to the temp directory

            # getting the path to the cloned repo
            repo_path = os.path.join(local_temp, os.listdir(local_temp)[0]) #noqa: PTH118
            # noqa hash lets the linter know that we can ignore specific quality warnings for that line
            
            # setting an empty dict for the repo tree
            tree = {}

            for root, _, files in os.walk(repo_path): # walking the repo directory
                dir = root.replace(repo_path, "").lstrip("/") # Normalize the directory path
                if dir.startswith(self._ignore): # skip ignored directories
                    continue
                for file in files: 
                    if file.endswith(self._ignore): # skip ignored file types
                        continue
                    file_path = os.path.join(dir, file) # noqa: PTH118
                    with open(os.path.join(root, file), "r", errors = "ignore") as f: # noqa: PTH118
                        tree[file_path] = f.read().replace(" ", "") # replacing white spaces with no space and storing

            user = kwargs["user"] # pulling in the user from the kwargs

            # creating a new instance for the repo in the database
            instance = self.model(
                content=tree, 
                name=repo_name, 
                link=link, 
                platform="github",
                author_id=user.id,
                author_full_name=user.full_name,
            )

            # saving the instance in the db
            instance.save()
        except Exception:
            raise
        finally:
            shutil.rmtree(local_temp) # cleaning up the temporary directory

        logger.info(f"Finished scrapping GitHub repository: {link}"