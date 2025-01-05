from llm_engineering.domain.exceptions import ImproperlyConfigured


def split_user_full_name(user:str | None) -> tuple[str, str]: 
    """ Takes the user full name and returns a tuple with the first name and the last name."""
    if user is None:
        raise ImproperlyConfigured("User name is empty.")
    
    # getting the tokens based on there being a space between the names
    name_tokens = user.split(" ")

    if len(name_tokens) == 0:
        raise ImproperlyConfigured("User name is empty.")
    # if only one token store it as both the first and the last name.
    elif len(name_tokens) == 1:
        first_name, last_name = name_tokens[0], name_tokens[0]
    # If two or more tokens, combine all but the last as the first name and use the last token as the last name
    else:
        first_name, last_name = " ".join(name_tokens[:-1]), name_tokens[-1]
    
    return first_name, last_name