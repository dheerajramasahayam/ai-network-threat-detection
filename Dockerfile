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
COPY src/ ./src/
COPY training/ ./training/
COPY run_training.sh README.md CITATION.cff ./

RUN mkdir -p results && chmod +x run_training.sh

CMD ["bash", "run_training.sh", "--epochs", "2", "--batch-size", "256", "--rf-trees", "150", "--cicids-sample-size", "50000"]
