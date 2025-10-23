"""
Configuration for the API
"""

import os

import torch

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
API_KEY = os.getenv(key="API_KEY", default="")  # Empty string means no auth required
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
SERVER_PORT = int(os.getenv("PORT", "8000"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "100"))
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "32"))
MODEL_IDLE_TIMEOUT_SECONDS = int(
    os.getenv("MODEL_IDLE_TIMEOUT_SECONDS", "3600")
)  # Set to 0 to disable auto-unloading
MODEL_CHECK_INTERVAL_SECONDS = int(os.getenv("MODEL_CHECK_INTERVAL_SECONDS", "60"))
