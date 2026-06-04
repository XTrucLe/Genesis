FROM python:3.12

WORKDIR /app

COPY pyproject.toml ./

COPY src/ ./src/

RUN pip install --no-cache-dir -e .

CMD ["genesis-train"]