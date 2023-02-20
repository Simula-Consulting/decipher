FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements_base.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

RUN pip install .

CMD bokeh serve --allow-websocket-origin='*' examples/pilot

EXPOSE 5006
