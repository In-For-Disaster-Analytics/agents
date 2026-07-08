FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CKAN_AGENT_API_HOST=0.0.0.0 \
    CKAN_AGENT_API_PORT=8787

WORKDIR /app

COPY ckan-agent-api/pyproject.toml ckan-agent-api/README.md ckan-agent-api/langgraph.json ./
COPY ckan-agent-api/app ./app
COPY ckan-registration ./ckan-registration

ENV CKAN_AGENT_LEGACY_DIR=/app/ckan-registration

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 8787

CMD ["sh", "-c", "uvicorn app.main:app --host ${CKAN_AGENT_API_HOST} --port ${CKAN_AGENT_API_PORT}"]
