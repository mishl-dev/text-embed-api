"""Embedding service module."""

from typing import List

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from .config import DEFAULT_BATCH_SIZE, DEVICE


def prepare_texts(texts: List[str], task_type: str) -> List[str]:
    """Add task prefix to texts."""
    return [f"{task_type}: {text}" for text in texts]


def generate_embeddings_batch(
    model: SentenceTransformer,
    texts: List[str],
    dimensionality: int,
    normalize: bool,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> torch.Tensor:
    """Generate embeddings with specified dimensionality using batching."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = model.encode(
            batch_texts, convert_to_tensor=True, device=DEVICE, show_progress_bar=False
        )
        batch_embeddings = F.layer_norm(
            batch_embeddings, normalized_shape=(batch_embeddings.shape[1],)
        )
        if dimensionality < 768:
            batch_embeddings = batch_embeddings[:, :dimensionality]
        if normalize:
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        all_embeddings.append(batch_embeddings)

    return torch.cat(all_embeddings, dim=0)
