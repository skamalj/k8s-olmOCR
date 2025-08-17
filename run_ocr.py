from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import base64
import fitz  # PyMuPDF
import tempfile
import os
import ocrmypdf

app = FastAPI()

@app.post("/process-pdf/")
async def process_pdf(
    file_base64: str = Form(...),
    start_page: Optional[int] = Form(None),
    end_page: Optional[int] = Form(None)
):
    try:
        # Decode base64
        pdf_bytes = base64.b64decode(file_base64)
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_input.write(pdf_bytes)
        temp_input.close()
        temp_output.close()

        # Apply page range if provided
        if start_page is not None or end_page is not None:
            doc = fitz.open(temp_input.name)
            start = start_page - 1 if start_page else 0
            end = end_page if end_page else doc.page_count
            subset = fitz.open()
            for i in range(start, end):
                subset.insert_pdf(doc, from_page=i, to_page=i)
            subset.save(temp_input.name)
            subset.close()

        # Run OCR (using GPU if ocrmypdf[ocrmypdf-gpu] installed)
        ocrmypdf.ocr(temp_input.name, temp_output.name, use_threads=True)

        # Return OCR-ed file as base64
        with open(temp_output.name, "rb") as f:
            result_base64 = base64.b64encode(f.read()).decode("utf-8")

        os.unlink(temp_input.name)
        os.unlink(temp_output.name)

        return {"status": "success", "ocr_pdf_base64": result_base64}

    except Exception as e:
        return {"status": "error", "message": str(e)}
