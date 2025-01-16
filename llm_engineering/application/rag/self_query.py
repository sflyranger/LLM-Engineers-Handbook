import opik
from langchain_openai import ChatOpenAI
from loguru import logger

from llm_engineering.application import utils
from llm_engineering.domain.documents import UserDocument
from llm_engineering.domain.queries import Query
from llm_engineering.settings import settings

from .base import RAGStep
from .prompt_templates import SelfQueryTemplate

# Creating the SelfQuery class to generate queries for the vector DB
class SelfQuery(RAGStep):
    # Tracking the SelfQuery generation process.
    @opik.track(name="SelfQuery.generate")
    def generate(self, query: Query) -> Query:
        # Return mock queries if in mock mode.
        if self._mock:
            return query
        
        # Creating the prompt from the SelfQueryTemplate file.
        prompt = SelfQueryTemplate().create_template()
        model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, api_key=settings.OPENAI_API_KEY, temperature=0)

        chain = prompt | model # Run the prompt then the model.

        response = chain.invoke({"question": query})
        user_full_name = response.content.strip("\n") # Strip the 

        # If theres not a user_full_name present return only the query without the user name.
        if user_full_name == "none":
            return query
        
        # Get the first_name and last_name using the split_user_full_name function
        first_name, last_name = utils.split_user_full_name(user_full_name)
        # Get or create the user
        user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)

        query.author_id = user.id
        query.author_full_name = user.full_name

        return query
    

if __name__ == "__main__":
    ##################EDIT IF NEEDED#######################
    query = Query.from_str("I am Steven Evans. Write and article about the best types of Machine Learning Models.")
    self_query = SelfQuery()
    query = self_query.generate(query)
    logger.info(f"Extracted author_id: {query.author_id}")
    logger.info(f"Extracted author_full_name: {query.author_full_name}")



