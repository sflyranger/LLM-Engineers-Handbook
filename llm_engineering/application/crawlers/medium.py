from bs4 import BeautifulSoup
from loguru import logger

from llm_engineering.domain.documents import ArticleDocument

from .base import BaseSeleniumCrawler

# creating the MediumCrawler class that inherits from the BaseSeleniumCrawler class
class MediumCrawler(BaseSeleniumCrawler):
    model = ArticleDocument # setting the model atrribute to be an ArticleDocument

    def set_extra_driver_options(self, options)-> None:
        options.add_argument(r"--profile-directory=Profile 2")
    
    def extract(self, link:str, **kwargs) -> None:
        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"Article already exists in the database: {link}")

            return
        logger.info(f"Starting scrapping Medium article: {link}")

        # Navigating to the link
        self.driver.get(link)
        self.scroll_page() # scrolling the page

        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        title = soup.find_all("h1", class_="pw-post-title") # finding the title of the article
        subtitle = soup.find_all("h2", class_="pw-subtitle-paragraph") # finding the subtitle
        
        # setting the data dict to be stored in the db
        data = {
            "Title": title[0].string if title else None, 
            "Subtitle": subtitle[0].string if subtitle else None, 
            "Content": soup.get_text(),
        }

        # closing the page
        self.driver.close()

        user = kwargs["user"]
        instance = self.model(
            platform="medium", 
            content=data, 
            link=link, 
            author_id=user.id,
            author_full_name=user.full_name,
        )
        instance.save()

        logger.info(f"Finished scrapping and saving article: {link}")
        