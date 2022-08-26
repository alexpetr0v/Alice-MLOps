FROM python:3.9

WORKDIR /code

RUN pip install --upgrade pip
RUN pip install poetry
COPY pyproject.toml poetry.lock /code/
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

COPY src/ /code/src
COPY cfg/ /code/cfg

CMD ["uvicorn", "src.api.inference:app", "--host", "0.0.0.0", "--port", "80"]
#CMD ls -d