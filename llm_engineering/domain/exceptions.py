# The two classes in this file allow for customizable error calling in logs
class LLMTwinException(Exception):
    pass

class ImproperlyConfigured(LLMTwinException):
    pass