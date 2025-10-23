# Text Embedding API

<p align="center">
<a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.9+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-brightgreen.svg?style=for-the-badge"></a>
<a href="https://litestar.dev/"><img alt="Framework" src="https://img.shields.io/badge/Built%20with-Litestar-8C43F6.svg?style=for-the-badge&logo=litestar"></a>
</p>

A high-performance REST API for the `nomic-ai/nomic-embed-text-v1.5` model. Features dynamic dimensionality and an auto-unload mechanism to conserve GPU memory.

## Setup

These instructions assume you have [uv](https://github.com/astral-sh/uv) installed (`pip install uv`).

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mishl-dev/text-embed-api
    cd text-embed-api
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    uv sync
    ```

3.  **Run the server:**
    ```bash
    uv run main.py
    ```
    The API is now running at `http://localhost:8000`.
    Interactive documentation is available at `http://localhost:8000/schema`.

## Configuration

Settings are managed in the `.env` file:

```ini
# Leave empty to disable authentication
API_KEY=your-secret-key

# Unload model after 3600s (1hr) of inactivity. Set to 0 to disable.
MODEL_IDLE_TIMEOUT_SECONDS=3600
```

## Usage

Use `curl` or any HTTP client to make a `POST` request to the `/embed` endpoint:

```bash
curl -X POST "http://localhost:8000/embed" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer your-secret-key" \
-d '{
  "texts": ["Hello world!"],
  "dimensionality": 256
}'
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.