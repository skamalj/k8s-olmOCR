FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install system deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip ghostscript tesseract-ocr libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy app
COPY run_ocr.py .

# Run uvicorn
CMD ["uvicorn", "run_ocr:app", "--host", "0.0.0.0", "--port", "8080"]
