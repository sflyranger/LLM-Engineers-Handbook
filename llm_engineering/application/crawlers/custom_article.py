from urllib.parse import urlparse

from langchain.community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers.html2text import Html2TextTransformer
from loguru import logger

from llm_engineering.domain.documents import ArticleDocument

from .base import BaseCrawler

# Creating the CustomArticleCrawler class that inherits from the BaseCrawler class
# this crawler doesn't implement any custom logic because it is not one of the primary methods.
# AsynHtmlLoader and Html2TextTransformer are difficult to use and customize in production environments so thats why this is a fallback.
# Most if not all of our documents may not even use this logic to begin with. In my case, none of my documents do.
# This class is coded for learning purposes only.
class CustomArticleCrawler(BaseCrawler):
    model = ArticleDocument # sets the model attribute to be an ArticleDocument

    def __init__(self)-> None:
        super().__init__() # inheriting all properties from the Basecrawler initialization
    
    def extract(self, link:str, **kwargs) -> None:
        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"Article already exists in the database: {link}") # sending logger info if the article already is present in the db

            return

        logger.info(f"Starting scrapping of article: {link}")

        loader = AsyncHtmlLoader([link]) # using the Html loader to load the link
        docs = loader.load() # loading the docs
        html2text = Html2TextTransformer() # setting the document transformer
        docs_transformed = html2text.transform_documents(docs) # transforming the docs and setting as the docs_transformed var
        doc_transformed = docs_transformed[0]
        
        # setting the content of the doc up by pulling the content and metadata from the transformed doc
        content = {
            "Title": doc_transformed.metadata.get("title"),
            "Subtitle": doc_transformed.metadata.get("description"),
            "Content": doc_transformed.page_content,
            "language": doc_transformed.metadata.get("language")
        }

        parsed_url = urlparse(link) # parsing the link
        platform = parsed_url.netloc # pulling the network location of the parsed url and storing as a platform var

        user = kwargs["user"]
        # getting an instance of the model and setting the different internal variables to the ones configured above/user input vars
        instance = self.model(
            content=content, 
            link=link, 
            platform=platform, 
            author_id=user.id,
            author_full_name=user.full_name,
        )
        instance.save() # saving the instance

        logger.info(f"Finished scrapping the custom article: {link}")
        