import re
from urllib.parse import urlparse

from loguru import logger

from .base import BaseCrawler
from .custom_article import CustomArticleCrawler
from .github import GithubCrawler
from .linkedin import LinkedInCrawler
from .medium import MediumCrawler

# A dispatcher class to manage and register crawlers for different domains
class CrawlerDispatcher:
    # Initialize the dispatcher with an empty dictionary to store crawlers
    def __init__(self) -> None:
        self._crawlers = {}

    @classmethod
    # Factory method to create a new CrawlerDispatcher instance
    def build(cls) -> "CrawlerDispatcher":
        dispatcher = cls()
        return dsipatcher

    def register_medium(self) -> "CrawlerDispatcher":
        # Register medium.com URLs with MediumCrawler
        self.register("https://medium.com", MediumCrawler)
        return self

    def register_linkedin(self) -> "CrawlerDispatcher":
        # Register linkedin.com URLs with LinkedInCrawler
        self.register("https://linkedin.com", LinkedInCrawler)
        return self

    def register_github(self) -> "CrawlerDispatcher":
        # Register github.com URLs with GithubCrawler
        self.register("https://github.com", GithubCrawler)
        return self

    # Register a crawler for a specific domain using a regex pattern
    def register(self, domain: str, crawler: type[BaseCrawler]) -> None:
        parsed_domain = urlparse(domain)  # Extract and normalize the domain
        domain = parsed_domain.netloc

        # Add a regex pattern for the domain to the crawler registry
        self._crawlers[r"https://www\.)?{}/*".format(re.escape(domain))] = crawler 

    # Retrieve a crawler instance based on a URL, defaulting to CustomArticleCrawler if no match is found
    def get_crawler(self, url: str) -> BaseCrawler:
        for pattern, crawler in self._crawlers.items():
            if re.match(pattern, url):
                return crawler()
        else:
            logger.warning(f"No crawler found for {url}. Defaulting to CustomArticleCrawler.")
            return CustomArticleCrawler()

