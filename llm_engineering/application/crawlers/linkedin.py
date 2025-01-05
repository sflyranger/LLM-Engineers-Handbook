import time
from typing import Dict, List

from bs4 import BeautifulSoup
from bs4.element import Tag
from loguru import logger
from selenium.webdriver.common.by import By 

from llm_engineering.domain.documents import PostDocument
from llm_engineering.domain.exceptions import ImproperlyConfigured
from llm_engineering.settings import settings

from .base import BaseSeleniumCrawler

# setting up the LinkedInCrawler class that inherits from the BaseSeleniumCrawler Class
class LinkedInCrawler(BaseSeleniumCrawler):
    model = PostDocument # setting the model attribute to be a PostDocument

    def __init__(self, scroll_limit: int = 5, is_deprecated:bool=True) -> None:
        super().__init__(scroll_limit) # inheriting the class settings while storing the scroll_limit

        self._is_deprecated = is_deprecated # setting the _is_deprecated attribute to be True

    # adding an experimental option to detach
    def set_extra_driver_options(self, options) -> None:
        options.add_experimental_option("detach", True)

    def login(self) -> None:
        if self._is_deprecated:
            raise DeprecationWarning(
                "As LinkedIn has updated its security features, the login() method is no longer supported."
            )
        
        # Naviagting to LinkedIn login page
        self.driver.get("https://www.linkedin.com/login")

        # error message if username and password are not configured in the settings
        if not settings.LINKEDIN_USERNAME or not settings.LINKEDIN_PASSWORD:
            raise ImproperlyConfigured(
                "LinkedIn scraper requires the {LINKEDIN_USERNAME} and {LINKEDIN_PASSWORD} settings."
            )
        
        # searching through the driver to enter the username into the login form
        self.driver.find_element(By.ID, "username").send_keys(settings.LINKEDIN_USERNAME)
        # enterin the password through the login form
        self.driver.find_element(By.ID, "password").send_keys(settings.LINKEDIN_PASSWORD)
        # clicking the login button to submit the form
        self.driver.find_element(By.CSS_SELECTOR, ".login__form_action_container button").click()

    def extract(self, link:str, **kwargs) -> None:
        if self._is_deprecated:
            raise DeprecationWarning(
                "As LinkedIn has updated its feed structure, the extract() method is no longer supported."
            )
        if self.model.link is not None:
            old_model = self.model.find(link=link)
            if old_model is not None:
                logger.info(f"Post already exists in the database: {link}")

                return
        
        logger.info(f"Starting scrapping of post: {link}")

        # initiate login function
        self.login()

        soup = self._get_page_content(link)

        # pulling in the data from the linked in post using the functions further in the class.
        data = {
            "Name": self._scrape_section(soup, "h1", class_="text-heading-xlarge"), 
            "About": self._scrape_section(soup, "div", class_="display-flex ph5 pv3"), 
            "Main Page": self._scrape_section(soup, "div", {"id":"main-content"}), 
            "Experience": self._scrape_experience(link), 
            "Education": self._scrape_education(link)
        }
        
        # navigate to the LinkedIn URL
        self.driver.get(link)
        time.sleep(5) # wait for the page to load
        button = self.driver.find_element(
            By.CSS_SELECTOR, ".app-aware-link.profile-creator-shared-content-view__footer-action"
        )
        # click the button to go to the shared content by the creator
        button.click()

        self.scroll_page() # scroll the page
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        post_elements = soup.find_all(
            "div", 
            class_="update-components-text relative update-components-update-v2__commentary", 

        )
        buttons = soup.find_all("button", class_="update-components-image__image-link")
        # extract the images from each button
        post_images = self._extract_image_urls(buttons)
        
        posts = self._extract_posts(post_elements, post_images) # extracting the post elements and the images and storing in the posts var

        logger.info(f"Found {len(posts)} posts for profile: {link}")
        
        # closing the navigation window
        self.driver.close()

        user = kwargs["user"]

        # bulk inserting all posts into the db
        self.model.bulk_insert(
            [
                PostDocument(platform="linkedin",
                 content=post, 
                 author_id = user.id,
                 author_full_name = user.full_name) for post in posts
            ]
        )

        logger.info(f"Finished scrapping data for profile: {link}")


    def _scrape_section(self, soup: BeautifulSoup, *args, **kwargs) -> str:
        """ Scrape a specific section of the LinkedIn profile."""
        # Example: Scrape the 'About' section.

        # finding the args and kwargs via soup based on input for the function
        parent_div = soup.find(*args, **kwargs)

        return parent_div.get_text(strip=True) if parent_div else ""
    
    def _extract_image_urls(self, buttons: List[Tag]) -> Dict[str, str]: 
        """ Extracts image URLs from button elements.

        Args:
            buttons (List[Tag]): A list of BeautifulSoup Tag objects representing buttons.
        
        Returns:
            Dict[str, str]: A dictionary mapping post indexes to image URLs.
        """

        # setting an empty dict
        post_images = {}
        # looping through each index, button pair in the buttons
        for i, button in enumerate(buttons):
            img_tag = button.find("img")
            # if the tag and the "src" is in img_tag attributes output a log that the image is not found
            if img_tag and "src" in img_tag.attrs:
                logger.warning("No image found in this button.")
            return post_images


    # function to get page content through the url and transform to BeautifulSoup Object    
    def _get_page_content(self, url:str) -> BeautifulSoup:
        """ Retrieve page content for a given URL."""

        # Navigate to the url
        self.driver.get(url)
        time.sleep(5) # wait for the url to load

        return BeautifulSoup(self.driver.page_source, "html_parser")
    
    def _extract_posts(self, post_elements:List[Tag], post_images: Dict[str, str]) -> Dict[str,Dict[str, str]]:
        """
        Extracts post texts and combines them with their respective images.

        Args:
            post_elements (List[Tag]): A list of BeautifulSoup Tag objects representing post elements.
            post_images (Dict[str,str]): A dictionary containing image URLs mapped by post index.

        Returns
            Dict[str, Dict[str, str]]: A dictionary containing post data with text and optional image URL.
        """
        
        # empty dict for the post data
        posts_data = {}
        for i, post_element in enumerate(post_elements):
            # pulling the text from the post element
            post_text = post_element.get_text(strip=True, seperator="\n")
            # settting the post data to be a key value with "text" as the ket and the post_text as the value
            post_data = {"text": post_text}

            # Check if the given post has an associated image
            if f"Post_{i}" in post_images:
                # set the "image" key in post_data equal to the post index
                post_data["image"] = post_images[f"Post_{i}"]
            # set the data for the given post equal to the extracted post text and image for each post
            posts_data[f"Post_{i}"] = post_data
        
        # return the full dict from the posts data
        return posts_data
    
    def _scrape_experience(self, profile_url:str) -> str:
        """ Scrapes the Experience section of the LinkedIn profile."""

        # navigate to the experience url
        self.driver.get(profile_url + "/details/experience/")
        time.sleep(5) # wait for the url to load

        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        
        # use soup to parse through the page to find the experience section 
        experience_content = soup.find("section", {"id": "experience-section"})

        return experience_content.get_text(strip=True) if experience_content else ""

    def _scrape_education(self, profile_url: str) -> str:
        """ Scrapes the education section of the LinkedIn profile."""

        # Navigate to the education section
        self.driver.get(profile_url + "/details/education/")
        time.sleep(5) # wait for the page to load

        soup = BeautifulSoup(self.driver.page_source, "html.parser")

        education_content = soup.find("section", {"id": "education-section"})

        return education_content.get_text(strip=True) if education_content else ""