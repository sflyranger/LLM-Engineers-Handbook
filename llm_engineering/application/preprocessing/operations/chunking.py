import re 

from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

from llm_engineering.application.networks import EmbeddingModelSingleton 

embedding_model = EmbeddingModelSingleton()

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Splits a large chunk of text into smaller chunks for better processing and storage for Vector DB.
    The function has two stages of splitting.
    1. Character-based splitting (`RecursiveTCharacterTextSplitter`)
    2. Token-based splitting (`SentenceTransformersTokenSplitter`)

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of the character-based count.
        chunk_overlap (int): The number of tokens to overlap between consecutive chunks.

    Returns:
        list[str]: A list of text chunks suitable for embedding or further processing.
    
    """

    # Step 1: First, we split the text inot smaller sections via character-based splitting.
    # This uses the logical double newline seperators ("\n\n") to ensure cohesion.
    # Note: No chunk overlap at this stage (chunk_overlap=0)

    # set the splitter
    character_splitter = RecursiveCharacterTextSplitter(seperators = ["\n\n"], chunk_size=chunk_size, chunk_overlap=0)
    
    # perform the split
    text_split_by_characters = character_splitter.split_text(text)

    # Step 2: Use a token-based splitter to further refine chunks.
    # This is more precise, ensuring chunks fit within a token limit.
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap, 
        tokens_per_chunk=embedding_model.max_input_length, 
        model_name=embedding_model.model_id, 
    )
    
    # Set an empty dict to store chunks
    chunks_by_tokens = []

    # iterate through each section of the split by character chunks and split based on the token
    for section in text_split_by_characters:
        chunks_by_tokens.extend(token_splitter.split_text(section))

    # return the chunks
    return chunks_by_tokens

    
    def chunk_document(text: str, min_length: int, max_length: int)-> list[str]:
        """ Alias for chunk_article()."""

        return chunk_article(text, min_length, max_length)


    def chunk_article(text:str, min_length: int, max_length: int) -> list[str]:
        # regex split that handles abbreviations and initials within sentences over the text.
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)

        # empty list for extracted chunks
        extracts = []

        # Initialize a temporary variable to accumulate sentences for the current chunk.
        current_chunk = ""
        
        # Iterate over each sentence to group them into chunks based on the specified length constraints.
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # if the current chunk plus the next sentence is less than the max length then add the sentence and white space
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + " "
            
            else:
                # If the current chunk meets the minimum length, append it to the results list. 
                if len(current_chunk) >= min_length:
                    extracts.append(current_chunk.strip())
                # Start a new chunk with the current sentence.
                current_chunk = sentence + " "

        # If long enough append the chunk to the extract list    
        if len(current_chunk) >= min_length:
            extracts.append(current_chunk.strip())
        
        # return the list of chunks for the article
        return extracts
