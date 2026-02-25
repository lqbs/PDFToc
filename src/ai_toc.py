# -*- coding: utf-8 -*-

"""AI-based table-of-contents extraction service."""

import json
import logging
import os
import re
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from base64 import b64encode
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from pypdf import PdfReader

logger = logging.getLogger(__name__)
DEBUG_LOG_PATH = "/Users/liqingbin/Code/Github/PDFToc/.cursor/debug-08fe99.log"
DEBUG_SESSION_ID = "08fe99"
OCR_LOW_CONCURRENCY = 3
OCR_REQUEST_TIMEOUT = 90
PNG_FAST_COMPRESSION_LEVEL = 1


def _debug_log(
    run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]
) -> None:
    # region agent log
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # endregion


class AITocError(Exception):
    """Base exception for AI TOC service."""


class AIConfigError(AITocError):
    """Raised when AI config is missing or invalid."""


class PDFTextExtractionError(AITocError):
    """Raised when text extraction from PDF fails."""


class OCRExtractionError(AITocError):
    """Raised when OCR extraction fails."""


class AIResponseError(AITocError):
    """Raised when AI API call fails or returns invalid output."""


def extract_toc_text(
    pdf_path: str,
    max_pages: int = 20,
    min_text_chars: int = 300,
    request_timeout: int = 60,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> str:
    """
    Extract TOC text from PDF and normalize it via an OpenAI-compatible model.

    Flow:
    1) Extract selectable text from selected page range.
    2) Fallback to OCR when extracted text quality is not enough.
    3) Send merged candidate text to OpenAI-compatible chat completions API.
    4) Normalize response to plain TOC lines.
    """
    run_id = "pre-fix"
    # region agent log
    _debug_log(
        run_id=run_id,
        hypothesis_id="H0",
        location="src/ai_toc.py:extract_toc_text",
        message="开始提取目录",
        data={
            "pdf_path_exists": os.path.isfile(pdf_path),
            "max_pages": max_pages,
            "min_text_chars": min_text_chars,
            "start_page": start_page,
            "end_page": end_page,
        },
    )
    # endregion
    if not pdf_path:
        raise ValueError("pdf_path is required.")
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError("PDF file not found: %s" % pdf_path)

    reader = _open_pdf_reader(pdf_path)
    total_pages = _get_pdf_page_count(pdf_path, reader=reader)
    page_start_idx, page_end_idx = _resolve_page_range(
        total_pages=total_pages,
        max_pages=max_pages,
        start_page=start_page,
        end_page=end_page,
    )

    selectable_text = _extract_text_from_pdf(
        pdf_path,
        start_idx=page_start_idx,
        end_idx=page_end_idx,
        reader=reader,
    )
    candidate_text = selectable_text
    selectable_lines = [line for line in selectable_text.split("\n") if line.strip()]
    selectable_usable = _is_text_usable(selectable_text, min_text_chars=min_text_chars)
    selectable_garbled, garbled_metrics = _is_text_likely_garbled(selectable_text)
    # region agent log
    _debug_log(
        run_id=run_id,
        hypothesis_id="H1",
        location="src/ai_toc.py:extract_toc_text",
        message="PDF可选文本质量",
        data={
            "char_count": len(selectable_text),
            "line_count": len(selectable_lines),
            "usable": selectable_usable,
            "sample_head": selectable_lines[:3],
            "likely_garbled": selectable_garbled,
            "garbled_metrics": garbled_metrics,
        },
    )
    # endregion

    logger.info("Trying OCR extraction by default.")
    try:
        ocr_text = _extract_text_by_ocr(
            pdf_path,
            start_idx=page_start_idx,
            end_idx=page_end_idx,
        )
    except OCRExtractionError as exc:
        logger.warning("OCR extraction failed: %s", exc)
        # region agent log
        _debug_log(
            run_id=run_id,
            hypothesis_id="H2",
            location="src/ai_toc.py:extract_toc_text",
            message="OCR默认提取失败，回退可选文本",
            data={
                "error": str(exc),
                "fallback_to_selectable": bool(selectable_text.strip()),
            },
        )
        # endregion
        if not selectable_text.strip():
            raise
    else:
        candidate_text = ocr_text
        ocr_lines = [line for line in ocr_text.split("\n") if line.strip()]
        # region agent log
        _debug_log(
            run_id=run_id,
            hypothesis_id="H2",
            location="src/ai_toc.py:extract_toc_text",
            message="OCR默认提取成功并作为候选文本",
            data={
                "ocr_char_count": len(ocr_text),
                "ocr_line_count": len(ocr_lines),
                "ocr_sample_head": ocr_lines[:3],
            },
        )
        # endregion

    if not candidate_text.strip():
        raise PDFTextExtractionError("Cannot extract useful text from the PDF.")

    candidate_lines = [line for line in candidate_text.split("\n") if line.strip()]
    # region agent log
    _debug_log(
        run_id=run_id,
        hypothesis_id="H3",
        location="src/ai_toc.py:extract_toc_text",
        message="送入LLM前候选文本",
        data={
            "candidate_char_count": len(candidate_text),
            "candidate_line_count": len(candidate_lines),
            "candidate_sample_head": candidate_lines[:5],
        },
    )
    # endregion
    formatted_text = _format_toc_with_llm(
        candidate_text, timeout=request_timeout
    )
    # region agent log
    _debug_log(
        run_id=run_id,
        hypothesis_id="H4",
        location="src/ai_toc.py:extract_toc_text",
        message="LLM原始返回",
        data={"raw_output_head": formatted_text.splitlines()[:8], "raw_output_len": len(formatted_text)},
    )
    # endregion
    normalized_text = normalize_toc_text(formatted_text)
    # region agent log
    _debug_log(
        run_id=run_id,
        hypothesis_id="H5",
        location="src/ai_toc.py:extract_toc_text",
        message="归一化后目录文本",
        data={
            "normalized_output_head": normalized_text.splitlines()[:8],
            "normalized_output_len": len(normalized_text),
        },
    )
    # endregion
    if not normalized_text:
        raise AIResponseError("AI returned empty TOC content.")
    return normalized_text


def normalize_toc_text(text: str) -> str:
    """Normalize model output into one TOC item per line."""
    if not text:
        return ""

    clean_text = _strip_code_fence(text).strip()
    json_lines = _try_parse_json_toc(clean_text)
    if json_lines:
        clean_text = "\n".join(json_lines)

    lines = []
    for raw_line in clean_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        # Drop common list markers added by LLM.
        line = re.sub(r"^[\-\*\u2022]\s+", "", line)
        line = re.sub(r"^\d+[\.\)]\s+", "", line)
        # Normalize dotted leaders and internal spaces.
        line = re.sub(r"\.{2,}", " ", line)
        line = re.sub(r"\s+", " ", line).strip()

        if line:
            lines.append(line)

    return "\n".join(_deduplicate_keep_order(lines))


def _extract_text_from_pdf(
    pdf_path: str, start_idx: int, end_idx: int, reader: Optional[Any] = None
) -> str:
    """Extract text directly from PDF pages."""
    if reader is None:
        reader = _open_pdf_reader(pdf_path)

    fragments = []
    for i in range(start_idx, end_idx):
        page = reader.pages[i]
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Text extraction failed on page %s: %s", i, exc)
            continue
        if text.strip():
            fragments.append(text)

    return _compact_multiline_text("\n".join(fragments))


def _extract_text_by_ocr(pdf_path: str, start_idx: int, end_idx: int) -> str:
    """Extract text via cloud multimodal OCR by rendering PDF pages to images."""
    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError as exc:
        raise OCRExtractionError(
            "OCR dependency is missing. Please install pypdfium2."
        ) from exc

    try:
        pdf_doc = pdfium.PdfDocument(pdf_path)
    except Exception as exc:
        raise OCRExtractionError("Failed to initialize OCR renderer: %s" % exc) from exc

    rendered_pages: List[Tuple[int, str]] = []
    try:
        for i in range(start_idx, end_idx):
            page = pdf_doc[i]
            bitmap = None
            try:
                bitmap = page.render(scale=2)
                image_data_url = _render_bitmap_to_data_url(bitmap)
                rendered_pages.append((i, image_data_url))
            except Exception as exc:
                logger.warning("OCR failed on page %s: %s", i, exc)
            finally:
                if bitmap is not None and hasattr(bitmap, "close"):
                    bitmap.close()
                if hasattr(page, "close"):
                    page.close()
    finally:
        if hasattr(pdf_doc, "close"):
            pdf_doc.close()

    page_text_by_index: Dict[int, str] = {}
    if rendered_pages:
        ai_config = _read_ai_config()
        worker_count = max(1, min(OCR_LOW_CONCURRENCY, len(rendered_pages)))
        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_page = {
                    executor.submit(
                        _request_vision_ocr_text,
                        image_data_url=image_data_url,
                        timeout=OCR_REQUEST_TIMEOUT,
                        session=session,
                        ai_config=ai_config,
                    ): page_idx
                    for page_idx, image_data_url in rendered_pages
                }
                for future in as_completed(future_to_page):
                    page_idx = future_to_page[future]
                    try:
                        page_text = future.result()
                        normalized_page_text = _normalize_vision_ocr_text(page_text)
                        if normalized_page_text:
                            page_text_by_index[page_idx] = normalized_page_text
                    except Exception as exc:
                        logger.warning("OCR failed on page %s: %s", page_idx, exc)

    lines: List[str] = []
    for page_idx, _ in rendered_pages:
        page_text = page_text_by_index.get(page_idx, "")
        if page_text:
            lines.extend(page_text.split("\n"))

    text = _compact_multiline_text("\n".join(lines))
    if not text:
        raise OCRExtractionError("OCR produced no readable text.")
    return text


def _get_pdf_page_count(pdf_path: str, reader: Optional[Any] = None) -> int:
    """Read total page count from PDF."""
    if reader is None:
        reader = _open_pdf_reader(pdf_path)

    total_pages = len(reader.pages)
    if total_pages <= 0:
        raise PDFTextExtractionError("PDF has no pages.")
    return total_pages


def _open_pdf_reader(pdf_path: str) -> Any:
    """Open PDF reader with unified error translation."""
    try:
        return PdfReader(pdf_path)
    except Exception as exc:
        raise PDFTextExtractionError("Failed to open PDF: %s" % exc) from exc


def _resolve_page_range(
    total_pages: int,
    max_pages: int,
    start_page: Optional[int],
    end_page: Optional[int],
) -> Tuple[int, int]:
    """
    Validate user-facing 1-based range and return 0-based half-open indexes.

    Returns:
        (start_idx, end_idx), where page indexes are [start_idx, end_idx).
    """
    if not isinstance(max_pages, int) or isinstance(max_pages, bool):
        raise ValueError("max_pages must be an integer.")
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1.")
    if not isinstance(total_pages, int) or isinstance(total_pages, bool):
        raise ValueError("total_pages must be an integer.")
    if total_pages < 1:
        raise ValueError("total_pages must be >= 1.")

    resolved_start = 1 if start_page is None else start_page
    if end_page is None:
        # Keep backward compatibility for default call: process first N pages.
        if start_page is None:
            resolved_end = min(total_pages, max_pages)
        else:
            resolved_end = total_pages
    else:
        resolved_end = end_page

    if not isinstance(resolved_start, int) or isinstance(resolved_start, bool):
        raise ValueError("start_page must be an integer.")
    if not isinstance(resolved_end, int) or isinstance(resolved_end, bool):
        raise ValueError("end_page must be an integer.")

    if resolved_start < 1:
        raise ValueError("start_page must be >= 1.")
    if resolved_end < 1:
        raise ValueError("end_page must be >= 1.")
    if resolved_end < resolved_start:
        raise ValueError("end_page must be greater than or equal to start_page.")
    if resolved_start > total_pages:
        raise ValueError(
            "start_page exceeds total PDF pages (%s)." % total_pages
        )
    if resolved_end > total_pages:
        raise ValueError("end_page exceeds total PDF pages (%s)." % total_pages)

    start_idx = resolved_start - 1
    end_idx = resolved_end
    return start_idx, end_idx


def _format_toc_with_llm(source_text: str, timeout: int) -> str:
    """Call OpenAI-compatible chat completions API and return plain text content."""
    endpoint, api_key, model = _read_ai_config()
    payload = _build_chat_payload(source_text=source_text, model=model)
    headers = {
        "Authorization": "Bearer %s" % api_key,
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            endpoint, headers=headers, json=payload, timeout=timeout
        )
    except requests.RequestException as exc:
        raise AIResponseError("AI request failed: %s" % exc) from exc

    if response.status_code >= 400:
        body = response.text[:500]
        raise AIResponseError(
            "AI request failed: HTTP %s, body=%s" % (response.status_code, body)
        )

    try:
        data = response.json()
    except ValueError as exc:
        content_type = response.headers.get("content-type", "")
        body = response.text[:200]
        raise AIResponseError(
            "AI response is not valid JSON. content-type=%s, body=%s"
            % (content_type, body)
        ) from exc

    content = _extract_message_content(data)
    if not content.strip():
        raise AIResponseError("AI response content is empty.")
    return content


def _extract_message_content(data: Dict[str, Any]) -> str:
    """Extract textual content from OpenAI-compatible response body."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AIResponseError("AI response has no choices.")

    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    content = ""
    if isinstance(message, dict):
        content = message.get("content", "")

    # Some providers return content as blocks.
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, dict) and isinstance(block.get("text"), dict):
                value = block["text"].get("value")
                if isinstance(value, str):
                    parts.append(value)
            elif isinstance(block, dict) and isinstance(block.get("content"), str):
                parts.append(block["content"])
            elif isinstance(block, str):
                parts.append(block)
        text = "\n".join(parts).strip()
        if text:
            return text
    elif isinstance(content, str) and content.strip():
        return content

    # Fallback for non-standard OpenAI-compatible providers.
    text = first.get("text") if isinstance(first, dict) else ""
    if isinstance(text, str) and text.strip():
        return text

    delta = first.get("delta") if isinstance(first, dict) else {}
    if isinstance(delta, dict):
        delta_content = delta.get("content", "")
        if isinstance(delta_content, str) and delta_content.strip():
            return delta_content

    top_keys = sorted(list(data.keys()))
    first_keys = sorted(list(first.keys())) if isinstance(first, dict) else []
    raise AIResponseError(
        "AI response content format is unsupported. top_keys=%s, choice_keys=%s"
        % (top_keys[:20], first_keys[:20])
    )


def _read_ai_config() -> Tuple[str, str, str]:
    """Read OpenAI-compatible endpoint configuration from env vars."""
    endpoint_env = os.getenv("PDFDIR_AI_CHAT_ENDPOINT", "").strip()
    base_url = os.getenv("PDFDIR_AI_BASE_URL", "").strip()
    api_key = os.getenv("PDFDIR_AI_API_KEY", "").strip()
    model = os.getenv("PDFDIR_AI_MODEL", "").strip()

    if not endpoint_env and not base_url:
        raise AIConfigError("Missing env PDFDIR_AI_BASE_URL.")
    if not api_key:
        raise AIConfigError("Missing env PDFDIR_AI_API_KEY.")
    if not model:
        raise AIConfigError("Missing env PDFDIR_AI_MODEL.")

    endpoint = endpoint_env if endpoint_env else _build_chat_endpoint(base_url)
    return endpoint, api_key, model


def _build_chat_endpoint(base_url: str) -> str:
    """Build /chat/completions endpoint from a base URL."""
    endpoint = base_url.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    return endpoint + "/chat/completions"


def _build_chat_payload(source_text: str, model: str) -> Dict[str, Any]:
    """Build API payload for OpenAI-compatible chat completions."""
    trimmed = source_text[:30000]
    system_prompt = (
        "You are a PDF table-of-contents formatter. "
        "Output plain text only. "
        "Each line must be one TOC item in the format 'title pageNumber'. "
        "Do not output markdown, code fences, explanations, or JSON."
    )
    user_prompt = (
        "Extract and normalize TOC lines from the following raw text. "
        "Keep meaningful hierarchy markers in titles. "
        "If a line has no page number but is clearly a TOC title, keep it.\n\n"
        "RAW TEXT:\n%s" % trimmed
    )
    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }


def _render_bitmap_to_data_url(bitmap: Any) -> str:
    """Convert pypdfium2 bitmap to PNG data URL without Pillow."""
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise OCRExtractionError("numpy is required for bitmap conversion.") from exc

    try:
        array = np.asarray(bitmap.to_numpy(), dtype=np.uint8)
    except Exception as exc:
        raise OCRExtractionError("Failed to convert rendered bitmap to array: %s" % exc) from exc

    if array.ndim == 2:
        array = np.stack((array, array, array), axis=-1)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 3:
        # pdfium bitmap is typically BGR.
        array = array[:, :, ::-1]
    elif array.ndim == 3 and array.shape[2] == 4:
        # Drop alpha to avoid BGRx/BGRA ambiguity across providers.
        array = array[:, :, [2, 1, 0]]
    else:
        raise OCRExtractionError("Unsupported bitmap shape: %s" % (array.shape,))

    height, width = int(array.shape[0]), int(array.shape[1])
    channels = int(array.shape[2])
    return _encode_image_bytes_to_png_data_url(
        width=width,
        height=height,
        image_bytes=array.tobytes(),
        channels=channels,
    )


def _encode_image_bytes_to_png_data_url(
    width: int, height: int, image_bytes: bytes, channels: int
) -> str:
    """
    Encode raw RGB/RGBA image bytes to a PNG data URL without Pillow.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        image_bytes: Raw pixel bytes in row-major order.
        channels: 3 for RGB, 4 for RGBA.
    """
    png_bytes = _encode_image_bytes_to_png_bytes(
        width=width, height=height, image_bytes=image_bytes, channels=channels
    )
    return "data:image/png;base64,%s" % b64encode(png_bytes).decode("ascii")


def _encode_image_bytes_to_png_bytes(
    width: int, height: int, image_bytes: bytes, channels: int
) -> bytes:
    """Encode raw RGB/RGBA image bytes into PNG bytes."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers.")
    if channels not in (3, 4):
        raise ValueError("channels must be 3 (RGB) or 4 (RGBA).")

    row_stride = width * channels
    expected_length = row_stride * height
    if len(image_bytes) != expected_length:
        raise ValueError(
            "image_bytes length mismatch, expected=%s, actual=%s"
            % (expected_length, len(image_bytes))
        )

    # PNG scanlines use a leading filter byte per row; 0 means no filter.
    raw_scanlines = bytearray()
    for row_idx in range(height):
        start = row_idx * row_stride
        end = start + row_stride
        raw_scanlines.append(0)
        raw_scanlines.extend(image_bytes[start:end])

    color_type = 2 if channels == 3 else 6
    ihdr = struct.pack("!IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    compressed = zlib.compress(bytes(raw_scanlines), level=PNG_FAST_COMPRESSION_LEVEL)

    return (
        b"\x89PNG\r\n\x1a\n"
        + _build_png_chunk(b"IHDR", ihdr)
        + _build_png_chunk(b"IDAT", compressed)
        + _build_png_chunk(b"IEND", b"")
    )


def _build_png_chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
    """Build a PNG chunk with CRC."""
    payload = chunk_type + chunk_data
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return (
        struct.pack("!I", len(chunk_data))
        + payload
        + struct.pack("!I", crc)
    )


def _build_vision_ocr_payload(image_data_url: str, model: str) -> Dict[str, Any]:
    """Build multimodal chat payload for OCR extraction."""
    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an OCR assistant. "
                    "Extract visible text from the image faithfully. "
                    "Return plain text only without markdown or explanations."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all readable text from this page image."},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    }


def _request_vision_ocr_text(
    image_data_url: str,
    timeout: int,
    session: Optional[requests.Session] = None,
    ai_config: Optional[Tuple[str, str, str]] = None,
) -> str:
    """Send one image to OpenAI-compatible multimodal endpoint and return OCR text."""
    endpoint, api_key, model = ai_config if ai_config is not None else _read_ai_config()
    payload = _build_vision_ocr_payload(image_data_url=image_data_url, model=model)
    headers = {
        "Authorization": "Bearer %s" % api_key,
        "Content-Type": "application/json",
    }
    requester = session if session is not None else requests

    try:
        response = requester.post(
            endpoint, headers=headers, json=payload, timeout=timeout
        )
    except requests.RequestException as exc:
        raise AIResponseError("Vision OCR request failed: %s" % exc) from exc

    if response.status_code >= 400:
        body = response.text[:500]
        raise AIResponseError(
            "Vision OCR request failed: HTTP %s, body=%s"
            % (response.status_code, body)
        )

    try:
        data = response.json()
    except ValueError as exc:
        content_type = response.headers.get("content-type", "")
        body = response.text[:200]
        raise AIResponseError(
            "Vision OCR response is not valid JSON. content-type=%s, body=%s"
            % (content_type, body)
        ) from exc

    return _extract_message_content(data)


def _normalize_vision_ocr_text(text: str) -> str:
    """Normalize OCR output by removing empty and duplicate lines."""
    if not text:
        return ""
    lines = []
    seen = set()
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return "\n".join(lines)


def _compact_multiline_text(text: str) -> str:
    """Normalize multiline text while keeping line boundaries."""
    if not text:
        return ""
    lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _is_text_usable(text: str, min_text_chars: int) -> bool:
    """
    Heuristic to decide whether direct text extraction is sufficient.

    Conditions:
    - enough characters
    - enough lines
    - at least two lines likely ending with a page number
    """
    if not text:
        return False

    lines = [line for line in text.split("\n") if line.strip()]
    if len(text) < min_text_chars or len(lines) < 6:
        return False

    page_like_count = 0
    for line in lines:
        if re.search(r"[\]\)\}\>》】）]?\s*\d+\s*$", line):
            page_like_count += 1
        if page_like_count >= 2:
            return True
    return False


def _is_text_likely_garbled(text: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Detect common PDF extraction mojibake patterns conservatively.

    Heuristics:
    - no CJK chars
    - enough ASCII letters/tokens to be meaningful
    - mostly short alpha tokens and very low lowercase ratio
    """
    if not text:
        return False, {
            "ascii_letter_count": 0,
            "cjk_count": 0,
            "lower_ratio": 0.0,
            "short_token_ratio": 0.0,
            "token_count": 0,
        }

    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    ascii_letters = re.findall(r"[A-Za-z]", text)
    ascii_letter_count = len(ascii_letters)
    lower_count = sum(1 for c in ascii_letters if c.islower())
    lower_ratio = (float(lower_count) / ascii_letter_count) if ascii_letter_count else 0.0

    alpha_tokens = re.findall(r"[A-Za-z]+", text)
    token_count = len(alpha_tokens)
    short_token_count = sum(1 for token in alpha_tokens if len(token) <= 2)
    short_token_ratio = (
        float(short_token_count) / token_count if token_count else 0.0
    )

    likely_garbled = (
        cjk_count == 0
        and ascii_letter_count >= 120
        and token_count >= 50
        and short_token_ratio >= 0.40
        and lower_ratio <= 0.20
    )
    return likely_garbled, {
        "ascii_letter_count": ascii_letter_count,
        "cjk_count": cjk_count,
        "lower_ratio": round(lower_ratio, 3),
        "short_token_ratio": round(short_token_ratio, 3),
        "token_count": token_count,
    }


def _merge_sources(primary: str, secondary: str) -> str:
    """Merge text sources while preserving unique lines."""
    merged_lines = []
    seen = set()
    for line in (primary + "\n" + secondary).split("\n"):
        normalized = re.sub(r"\s+", " ", line).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged_lines.append(normalized)
    return "\n".join(merged_lines)


def _strip_code_fence(text: str) -> str:
    """Remove surrounding markdown code fences."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        inner = re.sub(r"^```[\w\-]*\n?", "", stripped)
        inner = re.sub(r"\n?```$", "", inner)
        return inner.strip()
    return stripped


def _try_parse_json_toc(text: str) -> List[str]:
    """
    Parse JSON-like TOC output from model and convert to lines.

    Supported forms:
    - [{"title":"...", "page": 10}, ...]
    - {"toc":[...]}
    """
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return []

    try:
        data = json.loads(stripped)
    except ValueError:
        return []

    records: Optional[Sequence[Any]] = None
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and isinstance(data.get("toc"), list):
        records = data["toc"]
    if not records:
        return []

    lines = []
    for item in records:
        if isinstance(item, str):
            text_line = item.strip()
            if text_line:
                lines.append(text_line)
            continue
        if not isinstance(item, dict):
            continue

        title = item.get("title") or item.get("name") or item.get("text")
        page = _first_present_value(item, ("page", "page_num", "pagenum"))
        if isinstance(title, str):
            title = re.sub(r"\s+", " ", title).strip()
            if not title:
                continue
            if page is None or page == "":
                lines.append(title)
            else:
                lines.append("%s %s" % (title, page))
    return lines


def _deduplicate_keep_order(lines: Iterable[str]) -> List[str]:
    """Deduplicate lines with order preserved."""
    seen = set()
    result = []
    for line in lines:
        key = line.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _first_present_value(data: Dict[str, Any], keys: Sequence[str]) -> Any:
    """Return first non-None value by key order."""
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None

