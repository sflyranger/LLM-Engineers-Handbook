from pathlib import Path

from huggingface_hub import HfApi
from loguru import logger

try:
    from sagemaker.huggingface import HuggingFaceProcessor
except ModuleNotFoundError:
    logger.warning("Couldn't load SageMaker imports. Run 'poetry install --with aws' to support AWS.")

from llm_engineering import settings

# Define paths for evaluation directory and requirements file
evaluation_dir = Path(__file__).resolve().parent
evaluation_requirements_path = evaluation_dir / "requirements.txt"

# Function to run evaluation on SageMaker
def run_evaluation_on_sagemaker(is_dummy: bool = True) -> None:
    # Ensure required settings are available
    assert settings.HUGGINGFACE_ACCESS_TOKEN, "Hugging Face access token is required."
    assert settings.OPENAI_API_KEY, "OpenAI API key is required."
    assert settings.AWS_ARN_ROLE, "AWS ARN role is required."

    # Check if evaluation directory and requirements file exist
    if not evaluation_dir.exists():
        raise FileNotFoundError(f"The directory {evaluation_dir} does not exist.")
    if not evaluation_requirements_path.exists():
        raise FileNotFoundError(f"The file {evaluation_requirements_path} does not exist.")

    # Get Hugging Face user information
    api = HfApi()
    user_info = api.whoami(token=settings.HUGGINGFACE_ACCESS_TOKEN)
    huggingface_user = user_info["name"]
    logger.info(f"Current Hugging Face user: {huggingface_user}")

    # Set environment variables for the SageMaker job
    env = {
        "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
        "OPENAI_API_KEY": settings.OPENAI_API_KEY,
        "DATASET_HUGGINGFACE_WORKSPACE": huggingface_user,
        "MODEL_HUGGINGFACE_WORKSPACE": huggingface_user,
    }
    if is_dummy:
        env["IS_DUMMY"] = "True"

    # Initialize the HuggingFaceProcessor
    hfp = HuggingFaceProcessor(
        role=settings.AWS_ARN_ROLE,
        instance_count=1,
        instance_type="ml.g5.2xlarge",
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        base_job_name="evaluate-llm-twin",
        env=env,
    )

    # Run the processing job
    hfp.run(
        code="evaluate.py",
        source_dir=str(evaluation_dir),
    )

# Run the evaluation if the script is executed directly
if __name__ == "__main__":
    run_evaluation_on_sagemaker()
