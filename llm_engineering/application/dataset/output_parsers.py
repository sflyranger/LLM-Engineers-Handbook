from langchain.output_parsers import PydanticOutputParser

# Class to list all pydantic output parsers
class ListPydanticOutputParser(PydanticOutputParser):
    def _parse_obj(self, obj: dict | list):
        if isinstance(obj, list):
            return [super(ListPydanticOutputParser, self)._parse_obj(obj_) for obj_ in obj] # If a list of objects parse them all.
        else:
            return super(ListPydanticOutputParser, self)._parse_obj(obj) # If just one object parse it as a single object.
        