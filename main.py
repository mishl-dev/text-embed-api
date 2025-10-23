"""Main entry point for the API."""

import logging

from litestar import Litestar
from litestar.openapi.config import OpenAPIConfig
from litestar.openapi.spec import Contact, License

from embed_api.config import DEBUG_MODE, SERVER_PORT
from embed_api.model_manager import lifespan
from embed_api.routes import embed_texts, health_check, root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name=__name__)

openapi_config = OpenAPIConfig(
    title="Nomic Embed Text v1.5 API",
    version="1.0.0",
    description="""
    Production-ready embedding API using Nomic Embed Text v1.5 model.
    
    ## Features
    - Multiple task types
    - Matryoshka representation learning (64-768 dimensions)
    - Automatic batching for large inputs
    - Optional API key authentication
    - Model unloads from memory after inactivity
    """,
    contact=Contact(name="API Support", email="meow@mishl.dev"),
    license=License(name="MIT", identifier="MIT"),
    use_handler_docstrings=True,
)

app = Litestar(
    route_handlers=[health_check, embed_texts, root],
    lifespan=[lifespan],
    openapi_config=openapi_config,
    debug=DEBUG_MODE,
)

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on http://0.0.0.0:{SERVER_PORT}")
    logger.info(f"OpenAPI docs available at http://0.0.0.0:{SERVER_PORT}/schema")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
