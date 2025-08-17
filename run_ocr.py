# app/main.py
import base64
import tempfile
import os
import asyncio
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
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "10000"))  # default 512


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
            openai_api_base=VLLM_BASE_URL,
            max_tokens=MAX_TOKENS,
            temperature=0.1,
        )

        async def process_page(page_number: int):
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

                # Async call to vLLM server
                response = await llm.ainvoke([message])
                return page_number, response.content
            except Exception as inner_e:
                return page_number, f"Error processing page {page_number}: {inner_e}"

        # Run all pages concurrently
        tasks = [process_page(page) for page in range(req.start_page, req.end_page + 1)]
        results_list = await asyncio.gather(*tasks)

        # Convert to dict
        results = {page: content for page, content in results_list}

        return {"ocr_results": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
