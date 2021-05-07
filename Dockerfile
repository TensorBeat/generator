# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

COPY . .

RUN apt-get update && apt-get -y install gcc curl
RUN pip3 install poetry
RUN poetry install

CMD ["python3", "-m", "Generator.py"]
