# One image, three roles: the gateway API, the retrieval MCP server, the tickets MCP server.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    HF_HOME=/cache/hf FASTEMBED_CACHE_PATH=/cache/fastembed
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY mcp_servers ./mcp_servers
COPY ingest ./ingest
COPY evals ./evals
COPY configs ./configs
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 app && mkdir -p /app/data /cache && chown -R app /app/data /cache
USER app
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["python", "-m", "gateway.main"]
