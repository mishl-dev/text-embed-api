"""Model manager module."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import torch
from litestar import Litestar
from litestar.datastructures import State
from sentence_transformers import SentenceTransformer

from .config import (
    DEVICE,
    MAX_BATCH_SIZE,
    MODEL_CHECK_INTERVAL_SECONDS,
    MODEL_IDLE_TIMEOUT_SECONDS,
    MODEL_NAME,
)

logger = logging.getLogger(__name__)


def _load_model() -> SentenceTransformer:
    """Loads the sentence transformer model."""
    logger.info(f"Loading model {MODEL_NAME} on {DEVICE}...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=DEVICE)
    model.eval()
    logger.info(f"Model loaded successfully on {DEVICE}!")
    return model


async def model_manager_task(state: State) -> None:
    """A background task that unloads the model after a period of inactivity."""
    while True:
        await asyncio.sleep(MODEL_CHECK_INTERVAL_SECONDS)
        async with state.model_lock:
            if state.model is not None:
                idle_time = time.time() - state.last_used_time
                if idle_time > MODEL_IDLE_TIMEOUT_SECONDS:
                    logger.info(f"Model idle for {idle_time:.0f}s. Unloading...")
                    state.model = None
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                    logger.info("Model unloaded successfully.")


@asynccontextmanager
async def lifespan(app: Litestar):
    """Load model on startup, manage background task, and cleanup on shutdown."""
    app.state.model = _load_model()
    app.state.last_used_time = time.time()
    app.state.model_lock = asyncio.Lock()

    if MODEL_IDLE_TIMEOUT_SECONDS > 0:
        manager_task = asyncio.create_task(model_manager_task(app.state))
        logger.info(
            f"Model auto-unload enabled. Timeout: {MODEL_IDLE_TIMEOUT_SECONDS}s"
        )
    else:
        manager_task = None
        logger.info("Model auto-unload disabled.")

    logger.info(f"Max batch size: {MAX_BATCH_SIZE}")
    yield

    if manager_task:
        manager_task.cancel()
        try:
            await manager_task
        except asyncio.CancelledError:
            logger.info("Model manager task cancelled.")
    logger.info("Shutting down...")


async def get_model(state: State) -> SentenceTransformer:
    """Dependency to get or reload the model and update its last used time."""
    async with state.model_lock:
        if state.model is None:
            logger.info("Model is not loaded. Reloading for incoming request...")
            state.model = _load_model()
        state.last_used_time = time.time()
        return state.model
