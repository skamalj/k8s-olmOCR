# Dockerfile (GPU)
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

WORKDIR /app

# System dependencies for PyMuPDF, Pillow, etc.
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy requirements
COPY requirements.txt /app/

# Install Python deps
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy app
COPY app /app

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
