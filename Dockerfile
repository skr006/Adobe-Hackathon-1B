FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev poppler-utils \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
    torch==2.1.2 \
    sentence-transformers==2.2.2 \
    huggingface-hub==0.19.4 \
    PyMuPDF==1.23.7

CMD ["python", "app.py"]
