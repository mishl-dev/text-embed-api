"""
Schemas for the API
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .config import DEFAULT_BATCH_SIZE, MAX_BATCH_SIZE, MODEL_NAME


class EmbeddingRequest(BaseModel):
    """Request model for generating embeddings"""

    texts: List[str] = Field(
        ...,
        description="List of texts to embed",
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        examples=[["Hello world", "How are you?"]],
    )
    task_type: Literal[
        "search_document", "search_query", "clustering", "classification"
    ] = Field(
        default="search_document", description="Task type for embedding generation"
    )
    dimensionality: int = Field(
        default=768,
        description="Output embedding dimension using Matryoshka representation",
        ge=64,
        le=768,
        examples=[768, 512, 256],
    )
    normalize: bool = Field(
        default=True, description="Whether to L2 normalize the embeddings"
    )
    batch_size: Optional[int] = Field(
        default=None,
        description=f"Internal batch size for processing (default: {DEFAULT_BATCH_SIZE})",
        ge=1,
        le=MAX_BATCH_SIZE,
    )

    @field_validator("dimensionality")
    @classmethod
    def validate_dimensionality(cls, v: int) -> int:
        valid_dims = [64, 128, 256, 512, 768]
        if v not in valid_dims:
            raise ValueError(f"Dimensionality must be one of {valid_dims}")
        return v


class EmbeddingResponse(BaseModel):
    """Response model for embeddings"""

    embeddings: List[List[float]]
    model: str = Field(..., examples=[MODEL_NAME])
    task_type: str
    dimensionality: int
    num_texts: int


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_loaded: bool
    device: str
    max_batch_size: int


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str
    detail: Optional[str] = None
