from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import fitz


LOGGER = logging.getLogger(__name__)
BOLD_FLAG = getattr(fitz, "TEXT_FONT_BOLD", 16)
DEFAULT_BODY_SIZE = 11
MAX_FONT_SAMPLE_PAGES = 30
MAX_MARGIN_SAMPLE_PAGES = 50


class PermanentAnalysisError(RuntimeError):
    """An input problem that will not be fixed by retrying the same analysis job."""

    retryable = False


class UnsupportedPdfError(PermanentAnalysisError):
    pass


@dataclass(frozen=True)
class TextLine:
    text: str
    size: float
    is_bold: bool
    bbox: tuple[float, float, float, float]


def sanitize_author(value: str | None) -> str:
    if not value:
        return "Unknown Author"
    lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    first = lines[0] if lines else str(value).strip()
    markers = ("This version is considered", "Project Gutenberg", "www.gutenberg.org")
    for marker in markers:
        position = first.find(marker)
        if position > 0:
            first = first[:position].strip()
            break
    first = " ".join(first.split()).strip()
    return first or "Unknown Author"


def has_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))


def is_likely_toc_page(text: str) -> bool:
    score = 0
    lower_text = text.lower()
    if "contents" in lower_text or "index" in lower_text:
        score += 2
    if "목차" in text or "차례" in text:
        score += 5
    if len(re.findall(r"\.{3,}\s*\d+", text)) > 2:
        score += 5
    chapter_count = len(
        re.findall(
            r"(?i)^\s*(chapter|part|제|section)\s*[\dIVXOne]+",
            text,
            re.MULTILINE,
        )
    )
    if chapter_count > 4:
        score += 5
    return score >= 3


def find_start_page_and_author(doc: Any) -> tuple[str, int]:
    detected_author = "Unknown Author"
    start_page_index = 0
    metadata = getattr(doc, "metadata", None) or {}
    metadata_author = metadata.get("author")
    if metadata_author and metadata_author.strip():
        detected_author = metadata_author

    for page_index in range(min(MAX_FONT_SAMPLE_PAGES, len(doc))):
        text = doc[page_index].get_text()
        if detected_author == "Unknown Author":
            match = re.search(
                r"(?i)(?:By|Author[:\s]+)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                text,
            )
            if match:
                detected_author = match.group(1)
            match_kr = re.search(r"(?:지은이|저자|글)[:\s]+([가-힣]{2,4})", text)
            if match_kr:
                detected_author = match_kr.group(1)
        if is_likely_toc_page(text):
            continue
        start_pattern = (
            r"(?i)^\s*((Chapter|Part)\s+(one|1|I)|제?\s*\d+\s*장|"
            r"프롤로그|서문|머리말|Prologue)\b"
        )
        if re.search(start_pattern, text, re.MULTILINE):
            start_page_index = page_index
            break
    return sanitize_author(detected_author), start_page_index


def _page_dictionary(page: Any) -> dict[str, Any]:
    try:
        return page.get_text("dict", sort=True)
    except TypeError:
        # Older/mocked Page objects may not expose the sort keyword.
        return page.get_text("dict")


def _join_span_text(spans: Iterable[dict[str, Any]]) -> str:
    result = ""
    for span in spans:
        value = str(span.get("text") or "").strip()
        if not value:
            continue
        if not result:
            result = value
        elif result.endswith("-") and not has_korean(result):
            result += value
        else:
            result += " " + value
    return " ".join(result.split())


def iter_text_lines(page: Any) -> Iterable[TextLine]:
    page_data = _page_dictionary(page)
    blocks = [block for block in page_data.get("blocks", []) if block.get("lines")]
    blocks.sort(key=lambda block: (block.get("bbox", (0, 0, 0, 0))[1], block.get("bbox", (0, 0, 0, 0))[0]))

    for block in blocks:
        lines = sorted(
            block.get("lines", []),
            key=lambda line: (line.get("bbox", (0, 0, 0, 0))[1], line.get("bbox", (0, 0, 0, 0))[0]),
        )
        for line in lines:
            spans = [span for span in line.get("spans", []) if str(span.get("text") or "").strip()]
            text = _join_span_text(spans)
            if not text:
                continue
            character_count = sum(len(str(span.get("text") or "").strip()) for span in spans)
            bold_count = sum(
                len(str(span.get("text") or "").strip())
                for span in spans
                if int(span.get("flags") or 0) & BOLD_FLAG
            )
            size = max((float(span.get("size") or 0) for span in spans), default=0.0)
            bbox = tuple(line.get("bbox") or block.get("bbox") or (0, 0, 0, 0))
            yield TextLine(
                text=text,
                size=size,
                is_bold=character_count > 0 and bold_count * 2 >= character_count,
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            )


def normalize_margin_text(text: str) -> str:
    return " ".join(text.lower().split())


def is_page_marker(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"\s*[-–—]?\s*(?:(?:page|쪽)\s*)?\d+\s*[-–—]?\s*",
            text,
            re.IGNORECASE,
        )
    )


def page_height(page: Any) -> float:
    return float(getattr(getattr(page, "rect", None), "height", 0) or 0)


def is_repeated_margin_line(
    line: TextLine,
    page: Any,
    repeated_margin_texts: set[str],
) -> bool:
    if normalize_margin_text(line.text) not in repeated_margin_texts:
        return False
    height = page_height(page)
    if height <= 0:
        return False
    return line.bbox[1] <= height * 0.12 or line.bbox[3] >= height * 0.88


def find_repeated_margin_texts(doc: Any, start_page: int = 0) -> set[str]:
    page_indexes = list(range(start_page, min(len(doc), start_page + MAX_MARGIN_SAMPLE_PAGES)))
    if len(page_indexes) < 2:
        return set()

    occurrences: Counter[str] = Counter()
    for page_index in page_indexes:
        page = doc[page_index]
        height = float(getattr(getattr(page, "rect", None), "height", 0) or 0)
        seen_on_page: set[str] = set()
        try:
            for line in iter_text_lines(page):
                if is_page_marker(line.text) or len(line.text) > 120:
                    continue
                near_top = height > 0 and line.bbox[1] <= height * 0.12
                near_bottom = height > 0 and line.bbox[3] >= height * 0.88
                if near_top or near_bottom:
                    seen_on_page.add(normalize_margin_text(line.text))
        except (KeyError, TypeError, ValueError) as error:
            LOGGER.warning("Failed to inspect page %s margins: %s", page_index, error)
        occurrences.update(seen_on_page)

    threshold = max(2, math.ceil(len(page_indexes) * 0.3))
    return {text for text, count in occurrences.items() if count >= threshold}


def analyze_font_characteristics(
    doc: Any,
    start_page: int = 0,
    repeated_margin_texts: set[str] | None = None,
) -> dict[str, Any]:
    repeated_margin_texts = repeated_margin_texts or set()
    font_data: Counter[tuple[int, bool]] = Counter()
    final_page = min(len(doc), start_page + MAX_FONT_SAMPLE_PAGES)

    for page_index in range(start_page, final_page):
        page = doc[page_index]
        try:
            for line in iter_text_lines(page):
                if is_repeated_margin_line(line, page, repeated_margin_texts):
                    continue
                size = round(line.size)
                if size < 9:
                    continue
                font_data[(size, line.is_bold)] += len(line.text)
        except (KeyError, TypeError, ValueError) as error:
            LOGGER.warning("Failed to inspect page %s fonts: %s", page_index, error)

    if not font_data:
        return {"body_size": DEFAULT_BODY_SIZE, "body_is_bold": False}
    body_size, body_is_bold = max(font_data, key=font_data.get)
    return {"body_size": body_size, "body_is_bold": body_is_bold}


def _looks_like_explicit_heading(text: str) -> bool:
    return bool(re.match(
        r"(?i)^\s*(?:chapter|part|section|book|제?\s*\d+\s*장|"
        r"프롤로그|에필로그|서문|머리말|prologue|epilogue|"
        r"\d+(?:[.-]\d+)*[.)]?)\b",
        text,
    ))


def _is_heading_candidate(line: TextLine, body_size: float, body_is_bold: bool) -> bool:
    text = line.text.strip()
    if not text or len(text) > 80:
        return False
    explicit = _looks_like_explicit_heading(text)
    if text.endswith((".", "?", "!", "。", "？", "！")) and not explicit:
        return False
    if not has_korean(text) and text[0].islower() and not explicit:
        return False

    size_delta = line.size - body_size
    if size_delta > 1.5:
        return True
    short_title = len(text) <= 60 and len(text.split()) <= 12
    if not body_is_bold and line.is_bold and line.size >= body_size - 0.5:
        return explicit or short_title
    if body_is_bold and line.is_bold and size_delta > 0.5:
        return explicit or short_title
    return explicit and line.size >= body_size - 0.5


def extract_chapters_by_font_size(
    doc: Any,
    font_info: dict[str, Any],
    start_page: int = 0,
    repeated_margin_texts: set[str] | None = None,
) -> list[dict[str, str]]:
    chapters: list[dict[str, str]] = []
    current_title = "Intro"
    current_content: list[str] = []
    body_size = float(font_info["body_size"])
    body_is_bold = bool(font_info["body_is_bold"])
    repeated_margin_texts = repeated_margin_texts or set()
    extracted_text = False

    def flush_chapter() -> None:
        nonlocal current_content
        full_text = " ".join(current_content).strip()
        if full_text:
            chapters.append({"title": current_title, "text": full_text})
        current_content = []

    for page_index in range(start_page, len(doc)):
        page = doc[page_index]
        try:
            lines = list(iter_text_lines(page))
        except (KeyError, TypeError, ValueError) as error:
            LOGGER.warning("Failed to extract page %s: %s", page_index, error)
            continue
        for line in lines:
            text = line.text.strip()
            if not text:
                continue
            extracted_text = True
            if is_page_marker(text):
                continue
            if is_repeated_margin_line(line, page, repeated_margin_texts):
                continue

            if _is_heading_candidate(line, body_size, body_is_bold):
                if current_content:
                    flush_chapter()
                    current_title = text
                elif current_title == "Intro":
                    current_title = text
                elif len(current_title) + len(text) + 1 <= 160:
                    # Titles are frequently split across multiple styled lines.
                    current_title = f"{current_title} {text}"
                else:
                    current_title = text
            else:
                current_content.append(text)

    flush_chapter()
    if not extracted_text:
        raise UnsupportedPdfError(
            "The PDF contains no extractable text. OCR is required for scanned/image-only PDFs."
        )
    return chapters
