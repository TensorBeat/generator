# syntax=docker/dockerfile:1

FROM tensorflow/tensorflow:2.4.1-gpu

WORKDIR /app

RUN apt-get update && apt-get -y install gcc curl
RUN pip3 install poetry
COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false
RUN poetry install

COPY . .

CMD ["python3", "Generator.py"]
