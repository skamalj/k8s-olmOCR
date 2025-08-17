# app/main.py
import base64
import tempfile
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

app = FastAPI(title="OLM-OCR API")

# --- vLLM Server Configuration ---
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm-api:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", None)  # can be None
VLLM_MODEL = os.getenv("VLLM_MODEL", "allenai/olmOCR-7B-0225-preview")


# Input schema
class OCRRequest(BaseModel):
    file_base64: str
    start_page: int = 1
    end_page: int = 1


@app.post("/ocr")
async def run_ocr(req: OCRRequest):
    try:
        # Decode base64 -> temp PDF
        pdf_bytes = base64.b64decode(req.file_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            pdf_path = tmp.name

        # Initialize vLLM client
        llm = ChatOpenAI(
            model=VLLM_MODEL,
            openai_api_key=VLLM_API_KEY,
            openai_api_base=VLLM_BASE_URL,
            max_tokens=1024,
            temperature=0.8,
        )

        results = {}

        # Iterate over page range
        for page_number in range(req.start_page, req.end_page + 1):
            try:
                # Render PDF page to base64 PNG
                image_base64 = render_pdf_to_base64png(
                    pdf_path, page_number, target_longest_image_dim=1024
                )

                # Build OCR prompt
                anchor_text = get_anchor_text(
                    pdf_path, page_number, pdf_engine="pdfreport", target_length=4000
                )
                prompt_text = build_finetuning_prompt(anchor_text)

                # Construct multimodal message
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        },
                    ]
                )

                # Call vLLM server
                response = llm.invoke([message])
                results[page_number] = response.content
            except Exception as inner_e:
                results[page_number] = f"Error processing page {page_number}: {inner_e}"

        return {"ocr_results": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
