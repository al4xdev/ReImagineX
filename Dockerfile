FROM ghcr.io/astral-sh/uv:python3.13-alpine AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

FROM python:3.13-alpine

RUN addgroup -S app && adduser -S app -G app
WORKDIR /app

COPY --from=builder --chown=app:app /app /app
RUN mkdir -p /app/gallery_data && chown app:app /app/gallery_data

ENV PATH="/app/.venv/bin:$PATH" \
    DATA_DIR="/app/gallery_data" \
    REIMAGINEX_HOST="0.0.0.0"

VOLUME ["/app/gallery_data"]
EXPOSE 8888
USER app

CMD ["python", "-m", "src.server"]
