import json
from typing import Any, Dict, Optional

from loguru import logger

try:
    import boto3
except ModuleNotFoundError:
    logger.warning("Couldn't load AWS or SageMaker imports. Run 'poetry install --with aws' to support AWS.")

# Import necessary modules from the project
from llm_engineering.domain.inference import Inference
from llm_engineering.settings import settings

class LLMInferenceSagemakerEndpoint(Inference):
    """
    Class for performing inference using a SageMaker endpoint for LLM schemas.
    """

    def __init__(
        self,
        endpoint_name: str,
        default_payload: Optional[Dict[str, Any]] = None,
        inference_component_name: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Initialize the SageMaker client with AWS credentials and region
        self.client = boto3.client(
            "sagemaker-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
        )
        self.endpoint_name = endpoint_name
        self.payload = default_payload if default_payload else self._default_payload()
        self.inference_component_name = inference_component_name

    def _default_payload(self) -> Dict[str, Any]:
        """
        Generates the default payload for the inference request.

        Returns:
            dict: The default payload containing input text and parameters.
        """

        return {
            "inputs": "How is the weather?",  # Default input text
            "parameters": {
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,  # Maximum number of tokens to generate
                "top_p": settings.TOP_P_INFERENCE,  # Top-p sampling parameter
                "temperature": settings.TEMPERATURE_INFERENCE,  # Sampling temperature
                "return_full_text": False,  # Whether to return the full text or not
            },
        }

    def set_payload(self, inputs: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Sets the payload for the inference request.

        Args:
            inputs (str): The input text for the inference.
            parameters (dict, optional): Additional parameters for the inference. Defaults to None.
        """

        self.payload["inputs"] = inputs  # Set the input text
        if parameters:
            self.payload["parameters"].update(parameters)  # Update parameters if provided

    def inference(self) -> Dict[str, Any]:
        """
        Performs the inference request using the SageMaker endpoint.

        Returns:
            dict: The response from the inference request.
        Raises:
            Exception: If an error occurs during the inference request.
        """

        try:
            logger.info("Inference request sent.")
            invoke_args = {
                "EndpointName": self.endpoint_name,  # SageMaker endpoint name
                "ContentType": "application/json",  # Content type of the request
                "Body": json.dumps(self.payload),  # Payload for the inference request
            }
            if self.inference_component_name not in ["None", None]:
                invoke_args["InferenceComponentName"] = self.inference_component_name  # Add component name if provided
            response = self.client.invoke_endpoint(**invoke_args)  # Invoke the endpoint
            response_body = response["Body"].read().decode("utf8")  # Read and decode the response

            return json.loads(response_body)  # Parse and return the response as a dictionary

        except Exception:
            logger.exception("SageMaker inference failed.")  # Log the exception

            raise  # Re-raise the exception
