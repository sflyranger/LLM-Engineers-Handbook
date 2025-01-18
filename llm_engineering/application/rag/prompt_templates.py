from langchain.prompts import PromptTemplate

from .base import PromptTemplateFactory

# QueryExpansionTemplate inheriting from the PromptTemplateFactory base class

class QueryExpansionTemplate(PromptTemplateFactory):
    # Base prompt to enhance similarity search
    prompt: str = """You are an AI language model assistant. Your task is to generate {expand_to_n}
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions seperated by '{seperator}'.
    Original question: {question}"""

    # Creating the seperator property.,
    @property
    def seperator(self) -> str:
        return "#next-question"
    
    # Create template function.
    def create_template(self, expand_to_n: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt, 
            # Question will be changed everytime the chain is called from query expansion.
            input_variables=["question"], 
            # Providing the following as partial vairbles to make them immutable at runtime.
            partial_variables={
                "seperator": self.seperator, # providing a unique string to split the generated queries.
                "expand_to_n": self.expand_to_n, # Defining how many queries to generate.
            },
        )
    

# Creating the SelfQueryTemplate class to extract the relevant information from each query, such as metadata like te ID, comments, shares etc.
class SelfQueryTemplate(PromptTemplateFactory):
    ##############MAY NEED TO CHANGE THE ID NUMBER##################
    prompt: str = """You are an AI mode assistant. Your task is to extact information from a user question.
    The required information that needs to be extracted is the user name of user id. 
    Your response should consist of only the extracted user name (e.g., John Doe) or id (e.g., 1345256), nothing else.
    If the user question does not contain any user name or id, you should return the following token: none.
    
    For example:
    QUESTION 1:
    My name is Steven Evans and I want to post about...
    RESPONSE 1:
    Steven Evans
    
    QUESTION 2:
    I want to write a post about...
    RESPONSE 2:
    1345256

    QUESTION 3:
    My user id is 1345256 and I want to write a post about...
    RESPONSE 3:
    1345256
    
    User Question: {question}"""

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(template=self.prompt, input_variables=["question"])
    