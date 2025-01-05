import time
from abc import ABC, abstractmethod
from tempfile import mkdtemp

import chromedriver_autoinstaller
from selenium import webdriver # tool used for automating web browsers
from selenium.webdriver.chrome.options import options

from llm_engineering.domain.documents import NoSQLBaseDocument

# Check if the current version of the chromedriver exists
# and if it doesn't exist, download it automatically, 
# then add chromdriver path

chromedriver_autoinstaller.install()

# Creating the BaseCrawler class that inherits from the ABC (Abstract Base Class).
class BaseCrawler(ABC):
    model : type[NoSQLBaseDocument] # model is a class attribute pulled from NoSQLBaseDocument

    # setting an abstract method, meaning all subclasses of BaseCrawler must implement this method, if not it cannot be instatiated
    @abstractmethod
    def extract(self, link:str, **kwargs)-> None: ...

# setting up the BaseMediumCrawler class that inherits from ABC and BaseCrawler
class BaseSeleniumCrawler(BaseCrawler, ABC):
    def __init__(self, scroll_limit:int = 5) -> None:
        options = webdriver.ChromeOptions()

        options.add_argument("--no-sandbox") # disables Chrome snadbox for compatibility in restricted envrionments
        options.add_argument("--headless=new") # Runs Chrome in headless mode (no UI), with updated performance features
        options.add_argument("--disable-dev-shm-usage") # avoids shared memory issues in environments with limited `/dev/shm`
        options.add_argument("--log-level-3") # Supresses logs except for critical errors
        options.add_argument("--disable-popup-blocking") # Allows popups to ensure uninteruppted crawling 
        options.add_argument("--disable-notifications") # Blocks website notifications
        options.add_argument("--disable-extensions") # Disables Chrome extensions for a clean browsing environment
        options.add_argument("--disable-background-networking") # Prevents unnecessary background network activity
        options.add_argument("--ignore-certificate-errors") # Ignores SSL/TLK errors to access pages with invalid certificates
        options.add_argument(f"--user-data-dir={mkdtemp()}") # Uses a temporary directory for user data
        options.add_argument(f"--data-path={mkdtemp()}") # uses a temporary directory for Chrome's internal data
        options.add_argument(f"--disk-cache-dir={mkdtemp()}") # Uses a temporary directory for disk cache
        options.add_argument("--remote-debugging-port=9226") # Enables remote debugging on port 9226, preventing collisions

        self.set_extra_driver_options(options) # sets the extra driver options to include the above options

        self.scroll_limit = scroll_limit
        self.driver = webdriver.Chrome(
            # setting the webdriver options to be the set options
            options = options,
        )
    def set_extra_driver_options(self, options: Options) -> None:
        pass
    
    def login(self) -> None:
        pass
    
    def scroll_page(self) -> None:
        """ Scroll through the LinkedIn page based on the scroll limit."""
        current_scroll = 0
        # pulling the initial height of the page
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        while True:
            # scroll to the bottom of the page
            self.driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
            time.sleep(5) # wait for the page to load new content
            # get the new page height after scrolling
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            # stop scrolling if the page height hasn't changed or the scroll limit has been reached
            if new_height == last_height or (self.scroll_limit and current_scroll >= self.scroll_limit):
                break
            last_height = new_height # update the last height for the next iteration
            current_scroll+=1 # increment the scroll counter