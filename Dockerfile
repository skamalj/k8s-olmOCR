FROM alleninstituteforai/olmocr:latest

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY run_ocr.py .

# Run uvicorn
CMD ["uvicorn", "run_ocr:app", "--host", "0.0.0.0", "--port", "8080"]
