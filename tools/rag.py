from langchain.globals import set_verbose
from loguru import logger

from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.infrastructure.opik_utils import configure_opik

if __name__ == "__main__":
    configure_opik()
    set_verbose(True)

    query = """
        My name is Steven Evans.
        
        Could you draft a LinkedIn post discussing text classification?
        I'm particularly interested in:
            - How embeddings are formed
            - How they are trained to create context to predict the next token.
        """
    
    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query, k=9)

    logger.info("Retrieved documents:")
    for rank, document in enumerate(documents):
        logger.info(f"{rank + 1}: {document}")
    