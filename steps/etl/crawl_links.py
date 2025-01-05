from urllib.parse import urlparse

from loguru import logger
from tqdm import tqdm
from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application.crawlers.dispatcher import CrawlerDispatcher
from llm_engineering.domain.documents import UserDocument

# defining a step to fully crawl links using the CrawlerDispatcher for each link type.
@step
def crawl_links(user:UserDocument, links: list[str]) -> Annotated[list[str], "crawled_links"]:
    
    dispatcher = CrawlerDispatcher.build().register_linkedin().register_medium().register_github()

    logger.info(f"Starting to crawl {len(links)} links.")

    # empty dict to store the metadata
    metadata = {}
    # intialize the number of successful crawls to 0
    successful_crawls = 0
    
    # loop through each link using tqdm to read
    for link in tqdm(links):
        successful_crawl, crawled_domain = _crawl_link(dispatcher, link, user)
        successful_crawls += successful_crawl

        # adding the metadata, domain, and successful_crawl data to the  metadata var
        metadata = _add_to_metadata(metadata, crawled_domain, successful_crawl)
    
    # setting up the step context
    step_context = get_step_context()
    step_context.add_output_metadata(output_name="crawled_links", metadata=metadata)

    # log to track the number of successful links that were crawled
    logger.info(f"Successfully crawled {successful_crawls}/{len(links)} links.")

    return links
# function to crawl individual links, outputting the result as a tuple with the outcome and the domain name.
def _crawl_link(dispatcher: CrawlerDispatcher, link:str, user:UserDocument) -> tuple[bool, str]:
    # getting the crawler for the specific link
    crawler = dispatcher.get_crawler(link)
    # Parsing to get the domain based on the network location
    crawler_domain = urlparse(link).netloc

    try:
        # run the extract function
        crawler.extract(link=link, user=user)

        return (True, crawler_domain)
    # if it doesn't work output and error message
    except Exception as e:
        logger.error(f"An error occured while crawling: {e!s}")

        return (False, crawler_domain)

# function to add individual metadata for each crawl
def _add_to_metadata(metadata:dict, domain:str, successful_crawl: bool) -> dict:
    # if new domain add an empty dict for that domain
    if domain not in metadata:
        metadata[domain] = {}
    # increment the number of successful crawls for the given domain
    metadata[domain]["successful"] = metadata.get(domain, {}).get("successful", 0) + successful_crawl
    # increment the total number of attempts for the domain
    metadata[domain]["total"] = metadata.get(domain, {}).get("total", 0) + 1

    # return the updated metadata dict
    return metadata