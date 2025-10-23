"""Routes for the API."""

from litestar import Request, get, post
from litestar.datastructures import State
from litestar.di import Provide
from litestar.exceptions import NotAuthorizedException
from sentence_transformers import SentenceTransformer

from .config import (
    API_KEY,
    DEFAULT_BATCH_SIZE,
    DEVICE,
    MAX_BATCH_SIZE,
    MODEL_IDLE_TIMEOUT_SECONDS,
    MODEL_NAME,
)
from .embedding_service import generate_embeddings_batch, prepare_texts
from .model_manager import get_model
from .schemas import EmbeddingRequest, EmbeddingResponse, HealthResponse


# --- Security ---
async def api_key_guard(request: Request) -> None:
    """Guard function to validate API keys."""
    if not API_KEY:
        return  # Skip if no API key is configured
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise NotAuthorizedException("Missing or invalid Authorization header.")
    if auth_header[7:] != API_KEY:
        raise NotAuthorizedException("Invalid API key")


# --- API Routes ---
@get("/", summary="API Information", tags=["System"])
async def root() -> dict:
    """API information endpoint"""
    return {
        "name": "Nomic Embed Text v1.5 API",
        "version": "1.0.0",
        "model": MODEL_NAME,
        "device": DEVICE,
        "max_batch_size": MAX_BATCH_SIZE,
        "default_batch_size": DEFAULT_BATCH_SIZE,
        "authentication": "required" if API_KEY else "not required",
        "auto_unload_timeout_seconds": MODEL_IDLE_TIMEOUT_SECONDS
        if MODEL_IDLE_TIMEOUT_SECONDS > 0
        else "disabled",
    }


@get("/health", summary="Health Check", tags=["System"])
async def health_check(state: State) -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=hasattr(state, "model") and state.model is not None,
        device=DEVICE,
        max_batch_size=MAX_BATCH_SIZE,
    )


@post(
    "/embed",
    summary="Generate Embeddings",
    description="Generate embeddings for a list of texts.",
    dependencies={"model": Provide(get_model)},
    tags=["Embeddings"],
)
async def embed_texts(
    data: EmbeddingRequest, request: Request, model: SentenceTransformer
) -> EmbeddingResponse:
    """Generate embeddings endpoint with batching support."""
    await api_key_guard(request)

    prepared_texts = prepare_texts(data.texts, data.task_type)
    batch_size = data.batch_size or DEFAULT_BATCH_SIZE

    embeddings_tensor = generate_embeddings_batch(
        model, prepared_texts, data.dimensionality, data.normalize, batch_size
    )

    return EmbeddingResponse(
        embeddings=embeddings_tensor.cpu().tolist(),
        model=MODEL_NAME,
        task_type=data.task_type,
        dimensionality=data.dimensionality,
        num_texts=len(data.texts),
    )
