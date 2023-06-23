FROM python:3.10 as poetry

WORKDIR /app

RUN apt-get update && \
    pip install pipx && PIPX_BIN_DIR=/usr/local/bin pipx install poetry

COPY pyproject.toml poetry.lock .
RUN poetry export -f requirements.txt --without-hashes > requirements.txt

FROM python:3.10-slim as prod
WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY --from=poetry /app/requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD data_paths='{"base_path": "/mnt"}' bokeh serve examples/pilot