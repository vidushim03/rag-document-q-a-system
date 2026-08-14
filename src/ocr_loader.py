import base64
import io
import os
import re
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from groq import Groq
from langchain_core.documents import Document
from PIL import Image

load_dotenv()

MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "10"))
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
POPPLER_PATH = os.getenv("POPPLER_PATH", "").strip()
GROQ_OCR_MODEL = os.getenv("GROQ_OCR_MODEL", "qwen/qwen3.6-27b").strip()


def _configure_local_ocr():
    try:
        import pytesseract
    except ImportError:
        return None

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    return pytesseract


def _get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Add it to your .env file before using Groq OCR."
        )

    return Groq(api_key=api_key)


def _local_ocr_available() -> bool:
    return _configure_local_ocr() is not None


def _enough_text(text: str, minimum_chars: int = 40) -> bool:
    return len((text or "").strip()) >= minimum_chars


def _clean_response(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned or text.strip()


def pdf_page_to_base64(page):
    """Render a PDF page to a base64 PNG."""
    matrix = fitz.Matrix(2, 2)
    pixmap = page.get_pixmap(matrix=matrix)
    return base64.b64encode(pixmap.tobytes("png")).decode("utf-8")


def extract_text_with_groq_vision(base64_image):
    """Extract text from an image using Groq vision."""
    try:
        response = _get_client().chat.completions.create(
            model=GROQ_OCR_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Extract all the text exactly as it appears. Only return text.",
                        },
                    ],
                }
            ],
            max_tokens=1500,
        )
        return _clean_response(response.choices[0].message.content)
    except Exception as exc:
        print(f"Groq OCR Error: {exc}")
        return ""


def _image_bytes_to_text_local(image_bytes: bytes) -> str:
    pytesseract = _configure_local_ocr()
    if pytesseract is None:
        return ""

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        return pytesseract.image_to_string(image).strip()
    except Exception as exc:
        print(f"Local OCR image error: {exc}")
        return ""


def _load_pdf_with_local_ocr(path):
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("pdf2image is not installed, skipping local PDF OCR.")
        return []

    if not _local_ocr_available():
        print("pytesseract is not installed, skipping local PDF OCR.")
        return []

    docs = []

    try:
        images = convert_from_path(
            path,
            dpi=220,
            first_page=1,
            last_page=MAX_OCR_PAGES,
            poppler_path=POPPLER_PATH or None,
        )

        print(f"Local OCR processing started: {len(images)} pages loaded")

        for page_index, image in enumerate(images, start=1):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            text = _image_bytes_to_text_local(buffer.getvalue())

            if _enough_text(text):
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": path, "page": page_index},
                    )
                )

        print(f"Local OCR completed: {len(docs)} pages extracted")
        return docs
    except Exception as exc:
        print(f"Local OCR failed for {path}: {exc}")
        return []


def _load_pdf_with_groq_ocr(path):
    docs = []

    try:
        pdf = fitz.open(path)
        total_pages = len(pdf)
        pages_to_process = min(total_pages, MAX_OCR_PAGES)

        print(f"Groq OCR processing started: {total_pages} pages found")

        for page_num in range(pages_to_process):
            page = pdf[page_num]
            print(f"Processing page {page_num + 1}/{pages_to_process}")

            text = extract_text_with_groq_vision(pdf_page_to_base64(page))
            if _enough_text(text):
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": path, "page": page_num + 1},
                    )
                )

        pdf.close()
        print(
            f"Groq OCR completed: {len(docs)} pages extracted "
            f"(limited to {pages_to_process})"
        )
    except Exception as exc:
        print(f"Groq OCR failed for {path}: {exc}")

    return docs


def load_pdf_with_ocr(path):
    local_docs = _load_pdf_with_local_ocr(path)
    if local_docs:
        return local_docs

    return _load_pdf_with_groq_ocr(path)


def load_image_text(path):
    try:
        with open(path, "rb") as file:
            image_bytes = file.read()
    except OSError as exc:
        print(f"Image read error: {exc}")
        return ""

    local_text = _image_bytes_to_text_local(image_bytes)
    if _enough_text(local_text):
        return local_text

    try:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        extension = Path(path).suffix.lower().lstrip(".")
        mime_type = "image/png" if extension == "png" else "image/jpeg"

        response = _get_client().chat.completions.create(
            model=GROQ_OCR_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Extract all the text exactly as it appears. Only return text.",
                        },
                    ],
                }
            ],
            max_tokens=1500,
        )
        return _clean_response(response.choices[0].message.content.strip())
    except Exception as exc:
        print(f"Groq OCR image error: {exc}")
        return ""
