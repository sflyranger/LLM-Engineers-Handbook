from abc import ABC, abstractmethod

class DeploymentStrategy(ABC):
    # Abstract method to deploy, meant to be overridden in any instance.
    @abstractmethod
    def deploy(self, model, endpoint_name: str, endpoint_config_name: str) -> None:
        pass


class Inference(ABC):
    """An abstract class for performing inference."""

    def __init__(self):
        self.model=None
    
    # Methods meant to be overridden.
    @abstractmethod 
    def set_payload(self, inputs, parameters=None):
        pass

    @abstractmethod
    def inference(self):
        pass