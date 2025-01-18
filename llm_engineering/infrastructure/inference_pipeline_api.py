import opik 
from fastapi import FastAPI, HTTPException
from opik import opik_context
from pydantic import BaseModel

from llm_engineering import settings
from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.application.utils import misc
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.infrastructure.opik_utils import configure_opik
from llm_engineering.model.inference import InferenceExecutor, LLMInferenceSagemakerEndpoint

# Intitialize the opik configuration..
configure_opik()

# Initialize the app as a FastAPI endpoint.
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str


# Track the llm service call for inference.
@opik.track
def call_llm_service(query: str, context: str | None) -> str:
    # Call in the Sagemaker endpoint for the llm.
    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None)
    # Get the answer from the InferenceExecutor based on the llm, query and given context
    answer = InferenceExecutor(llm, query, context).execute()

    return answer

# Track the rag document retrieval pipeline
@opik.track
def rag(query: str):
    # Initialize the context retriever in non mock mode.
    retriever = ContextRetriever(mock=False)
    # Search and find the top three best queries.
    documents = retriever.search(query, k=3)
    # Get the context as embedded versions of the queries
    context = EmbeddedChunk.to_context(documents)

    # Get the answer based on the given query and context
    answer = call_llm_service(query, context)

    opik_context.update_current_trace(
        tags=["rag"], 
        metadata={
            "model_id": settings.HF_MODEL_ID, 
            "embedding_model_id": settings.TEXT_EMBEDDING_MODEL_ID, 
            "temperature": settings.TEMPERATURE_INFERENCE,
            "query_tokens": misc.compute_num_tokens(query), 
            "context_tokens": misc.compute_num_tokens(context), 
            "answer_tokens": misc.compute_num_tokens(answer)
        },
    )

    return answer 

# Establish the connection via the API and get the response.
@app.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest):
    try:
        answer = rag(query=request.query)

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e 