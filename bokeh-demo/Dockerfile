FROM python:3.10-slim-buster

WORKDIR /app

COPY matfact-0.1.0-py3-none-any.whl .

COPY requirements_base.txt requirements.txt

RUN --mount=type=cache,target=/root/.cache pip install --no-deps -r requirements.txt

COPY . .

RUN pip install .

CMD bokeh serve --allow-websocket-origin='*' examples/pilot

EXPOSE 5006
