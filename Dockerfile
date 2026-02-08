FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set environment variables
ENV UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# --- Dependency Layer ---
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project --no-dev

# --- Application Layer ---
COPY README.md .
COPY alembic ./alembic
COPY alembic.ini .
COPY src ./src

RUN uv sync --frozen --no-dev


EXPOSE 1337


ENTRYPOINT ["python", "-m"]
