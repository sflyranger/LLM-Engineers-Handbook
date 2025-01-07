from __future__ import annotations

from llm_engineering.domain.inference import Inference
from llm_engineering.settings import settings


class InferenceExecutor:
    def __init__(
        self,
        llm: Inference,
        query: str,
        context: str | None = None,
        prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.query = query
        self.context = context if context else ""

        # Set the default prompt if none is provided
        if prompt is None:
            self.prompt = """
You are a content creator. Write what the user asked you to while using the provided context as the primary source of information for the content.
User query: {query}
Context: {context}
            """
        else:
            self.prompt = prompt

    def execute(self) -> str:
        # Format the prompt with the query and context
        self.llm.set_payload(
            inputs=self.prompt.format(query=self.query, context=self.context),
            parameters={
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,  # Maximum number of tokens to generate
                "repetition_penalty": 1.1,  # Penalty for repetition
                "temperature": settings.TEMPERATURE_INFERENCE,  # Sampling temperature
            },
        )
        # Perform inference and get the generated text
        answer = self.llm.inference()[0]["generated_text"]

        return answer
