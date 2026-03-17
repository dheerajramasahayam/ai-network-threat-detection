FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dataset/ ./dataset/
COPY demo/ ./demo/
COPY evaluation/ ./evaluation/
COPY models/ ./models/
COPY training/ ./training/
COPY run_training.sh README.md CITATION.cff ./

RUN mkdir -p results && chmod +x run_training.sh

CMD ["bash", "run_training.sh", "--max-train-rows", "25000", "--max-test-rows", "10000", "--epochs", "5"]
