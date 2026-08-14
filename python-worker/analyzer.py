from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import fitz

from music_mapper import MusicMapper
from pdf_structure import (
    PermanentAnalysisError,
    UnsupportedPdfError,
    analyze_font_characteristics,
    extract_chapters_by_font_size,
    find_repeated_margin_texts,
    find_start_page_and_author,
    has_korean,
    is_likely_toc_page,
    sanitize_author,
)
from text_segmenter import group_pages, split_into_reading_pages


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
STORAGE_ROOT = Path(os.environ.get("EBOOK_STORAGE_ROOT", PROJECT_ROOT)).resolve()
DEFAULTS_DIR = STORAGE_ROOT / "defaults"
BUNDLED_DEFAULTS_DIR = BASE_DIR / "defaults"

# Backward-compatible name for callers that imported the former helper.
split_into_full_pages = split_into_reading_pages


def _save_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(data, output, ensure_ascii=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _extract_cover_atomic(doc: Any, destination: Path) -> None:
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    try:
        pixmap = doc[0].get_pixmap()
        pixmap.save(str(temporary))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_fallback_cover(destination: Path) -> None:
    candidates = (DEFAULTS_DIR / "default.png", BUNDLED_DEFAULTS_DIR / "default.png")
    fallback = next((path for path in candidates if path.is_file()), None)
    if fallback is None:
        raise RuntimeError("Unable to create a cover and no fallback cover exists")
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    try:
        shutil.copy2(fallback, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _build_segment(
    mapper: MusicMapper,
    pages: list[dict[str, Any]],
    segment_index: int,
    music_folder: Path,
) -> dict[str, Any]:
    combined_text = " ".join(page["text"] for page in pages)
    analysis = mapper.analyze_segment(combined_text)
    vibe = analysis["emotion"]
    selection_key = analysis["content_signature"]
    music_info = mapper.get_music(vibe, selection_key)
    if not music_info:
        music_info = mapper.get_music("neutral", selection_key)

    if music_info:
        filename = str(music_info["filename"])
        bpm = int(round(float(music_info.get("bpm") or 0)))
        source = "preset" if not music_info.get("is_custom") else "ai_reused"
        music_path = f"music/{filename}"
    elif (music_folder / "default_ambient.wav").is_file():
        filename = "default_ambient.wav"
        bpm = 80
        source = "system_default"
        music_path = f"music/{filename}"
    else:
        filename = ""
        bpm = 0
        source = "pending_generation"
        music_path = ""

    return {
        "segment_index": segment_index,
        "emotion": vibe,
        "music_filename": filename,
        "music_path": music_path,
        "music_source": source,
        "bpm": bpm,
        "generation_hint": {
            "target_emotion": vibe,
            "keywords": analysis["keywords"],
            "content_signature": analysis["content_signature"],
        },
        "pages": pages,
    }


def process_full_book_for_offline(
    pdf_path: str,
    book_root_folder: str,
    music_folder: str,
    web_path_prefix: str,
) -> dict[str, Any]:
    """Analyze a text PDF and atomically publish the offline-reader JSON artifacts."""
    pdf = Path(pdf_path)
    book_root = Path(book_root_folder)
    music_root = Path(music_folder).resolve()
    filename_base = pdf.stem
    book_identity = book_root.name
    book_root.mkdir(parents=True, exist_ok=True)
    mapper = MusicMapper(str(music_root.parent / "music_index.json"), str(music_root))

    try:
        doc = fitz.open(str(pdf))
    except (fitz.FileDataError, fitz.EmptyFileError) as error:
        raise PermanentAnalysisError(f"The uploaded file is not a readable PDF: {error}") from error
    except Exception as error:
        raise RuntimeError(f"PDF could not be opened: {error}") from error

    try:
        if getattr(doc, "needs_pass", False):
            raise PermanentAnalysisError("Password-protected PDFs are not supported")
        if len(doc) == 0:
            raise PermanentAnalysisError("The PDF has no pages")

        author, start_page = find_start_page_and_author(doc)
        repeated_margin_texts = find_repeated_margin_texts(doc, start_page)
        font_info = analyze_font_characteristics(doc, start_page, repeated_margin_texts)
        chapters = extract_chapters_by_font_size(doc, font_info, start_page, repeated_margin_texts)
        if not chapters:
            raise PermanentAnalysisError("No readable chapter content was found in the PDF")

        final_chapters = []
        for chapter_index, chapter in enumerate(chapters):
            grouped_pages = group_pages(split_into_reading_pages(chapter["text"]))
            segments = [
                _build_segment(mapper, pages, segment_index, music_root)
                for segment_index, pages in enumerate(grouped_pages)
            ]
            final_chapters.append({
                "chapter_index": chapter_index + 1,
                "title": chapter["title"],
                "segments": segments,
            })

        cover_filename = f"{book_identity}.png"
        cover_path = book_root / cover_filename
        try:
            _extract_cover_atomic(doc, cover_path)
        except Exception:
            _copy_fallback_cover(cover_path)

        clean_web_prefix = (web_path_prefix or "").rstrip("/")
        book_data = {
            "book_info": {
                "title": filename_base,
                "author": sanitize_author(author),
                "cover_path": f"{clean_web_prefix}/{cover_filename}",
                "total_chapters": len(final_chapters),
            },
            "chapters": final_chapters,
        }
        json_filename = f"{filename_base}_full.json"
        _save_json_atomic(book_root / json_filename, book_data)
        return {
            "success": True,
            "text_file": json_filename,
            "cover_image": cover_filename,
            "real_author": sanitize_author(author),
            "title": filename_base,
        }
    finally:
        doc.close()


__all__ = [
    "PermanentAnalysisError",
    "UnsupportedPdfError",
    "analyze_font_characteristics",
    "extract_chapters_by_font_size",
    "find_start_page_and_author",
    "has_korean",
    "is_likely_toc_page",
    "process_full_book_for_offline",
    "sanitize_author",
    "split_into_full_pages",
]
