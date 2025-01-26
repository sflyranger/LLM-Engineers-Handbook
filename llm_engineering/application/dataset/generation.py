from abc import ABC, abstractmethod 

import tiktoken 
from langchain_core.exceptions import OutputParserException 
from langchain_core.language_models.fake import FakeListLLM 
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.openai import ChatOpenAI
from loguru import logger 

from llm_engineering import domain
from llm_engineering.application import utils
from llm_engineering.domain.cleaned_documents import CleanedDocument
from llm_engineering.domain.dataset import DatasetType, TrainTestSplit
from llm_engineering.domain.prompt import GenerateDatasetSamplesPrompt, Prompt
from llm_engineering.domain.types import DataCategory
from llm_engineering.settings import settings 

from . import constants
from . import utils as generation_utils
from .output_parsers import ListPydanticOutputParser

# Base class to generate datasets, inherits from the abstract base class.
class DatasetGenerator(ABC):
    tokenizer = tiktoken.encoding_for_model(settings.OPENAI_MODEL_ID)
    dataset_type = DatasetType | None = None

    system_prompt_template = """You are a helpful assistant who generates {dataset_format} based on the given context. \
Provide your response in JSON format.
        """
    prompt_template_str: str | None = None

    @classmethod
    def get_system_prompt(cls) -> Prompt:
        # Ensure there is a dataset type, instruction or preference.
        assert cls.dataset_type is not None, "Dataset type must be set before calling get_system_prompt()"
        
        # Set the format to either instruction-answer pairs or instruction-answer triples
        dataset_format = (
            "instruction-answer pairs" if cls.dataset_type == DatasetType.INSTRUCTION else "instruction-answer triples"
        )
        # Set the input variables to be the dataset format.
        input_variables = {
            "dataset_format": dataset_format
        }
        # Set the system prompt to be based on the template.
        system_prompt = cls.system_prompt_template.format(**input_variables)
        
        # Return the full prompt.
        return Prompt(
            template = cls.system_prompt_template, 
            input_variables=input_variables, 
            content=system_prompt
        )

    # Method to get all prompts for the samples
    @classmethod 
    def get_prompts(cls, documents: list[CleanedDocument]) -> dict[DataCategory, list[GenerateDatasetSamplesPrompt]]:
        # Extract the substrings.
        documents = generation_utils.extract_substrings(documents)

        # Empty dictionary for stored prompts
        grouped_prompts = {}
        # Group the documents by their data category
        grouped_cleaned_documents = CleanedDocument.group_by_category(documents)

        # Store the prompts based on their category in the grouped_prompts dictionary.
        for category, category_documents in grouped_cleaned_documents.items():
            category_prompts = [cls.get_prompt(document) for document in category_documents]
            grouped_prompts[category] = category_prompts

        return grouped_prompts
    
    # Method to get a single prompt for a given sample
    @classmethod
    def get_prompt(cls, document: CleanedDocument) -> GenerateDatasetSamplesPrompt:
        # Ensure the prompt_template is present.
        assert cls.prompt_template_str is not None, "Prompt template must be set before calling get_prompt()"

        # Pull the data category.
        data_category = document.get_category()

        prompt_template = PromptTemplate.from_template(
            template=cls.prompt_template_str, 
            template_format="jinja2",
        )

        # Set the extraction as the documents content.
        input_variables = {
            "extract": document.content
        }
        # Format based on content
        prompt = prompt_template.format(**input_variables)
        # Tokenize the content.
        prompt_tokens = cls.tokenizer.encode(prompt)
        # If we go past the max token length then we only pull tokens up to the maximum length.
        if len(prompt_tokens) > settings.OPENAI_MAX_TOKEN_WINDOW:
            prompt_tokens = prompt_tokens[: settings.OPENAI_MAX_TOKEN_WINDOW]
            prompt = cls.tokenizer.decode(prompt_tokens)
        
        # Finalize the prompt 
        prompt = GenerateDatasetSamplesPrompt(
            template=prompt_template.template, 
            input_variables=input_variables, 
            content=prompt, 
            num_tokens=len(prompt_tokens),
            data_category=data_category, 
            document=document,
        )

        return prompt
    
    # Method to generate prompts
    @classmethod
    def generate(
        cls, 
        prompts: dict[DataCategory, list[GenerateDatasetSamplesPrompt]], 
        test_size: float = 0.2,
        mock: bool = False,
    ) -> TrainTestSplit:
        assert cls.dataset_type is not None, "Dataset type must be set before calling generate()"

        # Internal function to push samples into langchain to get the system and human messages.
        def _to_langchain(
            prompt: GenerateDatasetSamplesPrompt
        )-> list[BaseMessage]:
            messages = [
                SystemMessage(content=cls.get_system_prompt().content),
                HumanMessage(content=prompt.content),
            ]

            return messages

        if mock:
            llm = FakeListLLM(responses=[constants.get_mocked_response(cls.dataset_type)])
        # If not a mock prompt we need to generate the prompt using OPENAI.    
        else:
            assert settings.OPENAI_API_KEY is not None, "OpenAI API key must be set to generate objects."

            llm = ChatOpenAI(
                model=settings.OPENAI_MODEL_ID,
                api_key=settings.OPENAI_API_KEY, 
                max_tokens=2000 if cls.dataset_type == DatasetType.PREFERENCE else 1200, # Setting the max_tokens based on the model
                temperature= 0.7 # Increasing the variability of responses.
            )

        # Used to parse the llm output into a structured format (pydantic objects) based on the type.
        parser = ListPydanticOutputParser(pydantic_object=cls._get_dataset_sample_type())
        
        # The llm is obtained using the parser, either mock or FakeList.
        chain = llm | parser 
        
        # Empty dictionary for storage.
        datasets = {}
        for category, category_prompts in prompts.items():
            langchain_category_prompts = [_to_langchain(prompt) for prompt in category_prompts]
            batches = utils.misc.batch(langchain_category_prompts, size=24) # Creating batches of size 24.

            # Empty list for flattened samples.
            flattened_instruct_dataset_samples = []
            for batch in batches:
                try:
                    batched_dataset_samples = chain.batch(batch, stop=None)

                    for instruct_dataset_sample_batch in batched_dataset_samples:
                        flattened_instruct_dataset_samples.extend(instruct_dataset_sample_batch) # Adding the batches to the empty list.

                except OutputParserException:
                    logger.exception(f"Failed to parse the output JSON for a batch of category: {category}")

            # Build the dataset object.
            dataset = domain.dataset.build_dataset(
                dataset_type = cls.dataset_type, category=category, samples=flattened_instruct_dataset_samples
            )

            # Store the dataset in the dictionary
            datasets[category] = dataset
            logger.info(f"Generated {len(dataset.samples)} samples for category '{category}'.")

            # Storing the processed datasets post split.
        processed_datasets = cls.post_process_datasets(datasets, test_size=test_size)

        return processed_datasets

    # Internal method to get the sample type (Instruction of Preference).    
    @classmethod
    def _get_dataset_sample_type(
        cls,
    ) -> type[domain.dataset.InstructDatasetSample] | type[domain.dataset.PreferenceDatasetSample]:
        return (
            domain.dataset.InstructionDatasetSample
            if cls.dataset_type == DatasetType.INSTRUCTION
            else domain.dataset.PreferenceDatasetSample
        )

    # Abstract method enforced to run the train test split.
    @classmethod
    @abstractmethod
    def post_process_datasets(
        cls, datasets: dict[DataCategory, domain.dataset.InstructDataset], test_size: float
    )-> TrainTestSplit:
        pass

# Subclass inheriting from the DatasetGenerator class to generate Instruct datasets.
class InstructionDatasetGenerator(DatasetGenerator):
    dataset_type = DatasetType.INSTRUCTION

    prompt_template_str = """Based on the following extract, generate five instruction-answer pairs. Each instruction \
must ask to write about a specific topic contained in the context. Each answer \
must provide a relevant paragraph based on the information found in the \
context. Only use concepts from the context to generate the instructions. \
Instructions must never explicitly mention a context, a system, a course, or an extract. \
Instructions must be self-contained and general. \
Answers must imitate the writing style of the context. \
    
Example instruction: Explain the concept of an LLM Twin. \
Example answer: An LLM Twin is essentially an AI character that mimics your writing style, personality, and voice. \
It's designed to write just like you by incorporating these elements into a language model. \
The idea is to create a digital replica of your writing habits using advanced AI techniques. \

Structure the answer in JSON format, ready to be loaded in Python by json.loads(), as a list of objects.
Do not add any extra characters and provide your response in JSON format with the following structure:
[
    {"instruction": "...", "answer": "..."},
    ...
]

Extract:
{extract}
"""

    @classmethod
    def post_process_datasets(
        cls, datasets: dict[DataCategory, domain.dataset.InstructDataset], test_size: float
    ) -> TrainTestSplit:
        train_test_split = generation_utils.create_instruct_train_test_split(
            datasets, test_size=test_size, random_state=42
        )

        return train_test_split

# Subclass inheriting from DatasetGenerator class to generate Preference datasets.s 
class PreferenceDatasetGenerator(DatasetGenerator):
    dataset_type = DatasetType.PREFERENCE

    prompt_template_str = """Based on the following extract, generate five instruction-answer triples. Each triple should consist of:
1. An instruction asking about a specific topic in the context.
2. A generated answer that attempts to answer the instruction based on the context, named as 'rejected'.
3. An extracted answer that is a relevant excerpt directly from the given context, named as 'chosen'.

Instructions must be self-contained and general, without explicitly mentioning a context, system, course, or extract.

Important:
- Ensure that the extracted answer, the chosen one, is a verbatim copy from the context, including all punctuation and apostrophes.
- Do not add any ellipsis (...) or [...]  to indicate skipped text in the extracted answer.
- If the relevant text is not continuous, use two separate sentences from the context instead of skipping text.

Structure the answer in JSON format, ready to be loaded in Python by json.loads(), as a list of objects.
Do not add any extra characters and provide your response in JSON format with the following structure:
[
    {
        "instruction": "...",
        "rejected": "...",
        "chosen": "..."
    },
    ...
]

Extract:
{extract}
"""

    @classmethod
    def post_process_datasets(
        cls, datasets: dict[DataCategory, domain.dataset.PreferenceDataset], test_size: float
    )-> TrainTestSplit:
        # Filter based on length.
        datasets = utils.filter_short_answers(datasets)
        # Filter based on format.
        datasets = utils.filter_answer_format(datasets)

        # Find the number of samples after filtering.
        remaining_samples = sum([dataset.num_samples for dataset in datasets.values()])
        logger.info(
            f"Filtered out short answers and answers with the incorrect format. Remaining samples: {remaining_samples}"
        )

        # Split the filtered datasets.
        train_test_split = utils.create_preference_train_test_split(
            datasets, test_size=test_size, random_state=42
        )

        return train_test_split

# Final function to generate datasets.
def get_dataset_generator(dataset_type: DatasetType) -> type[DatasetGenerator]:
    if dataset_type == DatasetType.INSTRUCTION:
        return InstructionDatasetGenerator
    elif dataset_type == DatasetType.PREFERENCE:
        return PreferenceDatasetGenerator
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")