import json
from types import SimpleNamespace

import fitz
import pytest

import music_mapper
from analyzer import _save_json_atomic
from music_mapper import MusicMapper
from pdf_structure import (
    BOLD_FLAG,
    UnsupportedPdfError,
    analyze_font_characteristics,
    extract_chapters_by_font_size,
    find_repeated_margin_texts,
    iter_text_lines,
)
from text_segmenter import group_pages, split_into_reading_pages


def span(text, *, size=11, flags=0):
    return {"text": text, "size": size, "flags": flags}


def line(*spans, y=100):
    return {"bbox": (40, y, 560, y + 20), "spans": list(spans)}


class FakePage:
    def __init__(self, lines, height=800):
        self.lines = lines
        self.rect = SimpleNamespace(height=height)

    def get_text(self, kind=None, sort=False):
        if kind == "dict":
            return {"blocks": [{"bbox": (40, 0, 560, 800), "lines": self.lines}]}
        return "\n".join(
            " ".join(str(item.get("text") or "") for item in value["spans"])
            for value in self.lines
        )


class FakeDocument(list):
    metadata = {}


def test_pymupdf_bold_flag_is_used_instead_of_serif_flag():
    assert BOLD_FLAG == fitz.TEXT_FONT_BOLD == 16
    page = FakePage([
        line(span("Serif", flags=fitz.TEXT_FONT_SERIFED)),
        line(span("Bold", flags=fitz.TEXT_FONT_BOLD), y=140),
    ])
    extracted = list(iter_text_lines(page))
    assert extracted[0].is_bold is False
    assert extracted[1].is_bold is True


def test_split_title_spans_are_merged_and_repeated_header_is_removed():
    header = line(span("EBookStudio Sample", size=9), y=10)
    document = FakeDocument([
        FakePage([
            header,
            line(span("Chapter", size=17, flags=BOLD_FLAG), span("One", size=17, flags=BOLD_FLAG), y=80),
            line(span("First chapter body text that is long enough to remain as content."), y=140),
        ]),
        FakePage([
            header,
            line(span("Chapter Two", size=17, flags=BOLD_FLAG), y=80),
            line(span("Second chapter body text."), y=140),
        ]),
        FakePage([header, line(span("More second chapter text."), y=140)]),
    ])
    repeated = find_repeated_margin_texts(document)
    assert "ebookstudio sample" in repeated
    font_info = analyze_font_characteristics(document, repeated_margin_texts=repeated)
    chapters = extract_chapters_by_font_size(document, font_info, repeated_margin_texts=repeated)
    assert [chapter["title"] for chapter in chapters] == ["Chapter One", "Chapter Two"]
    assert all("EBookStudio Sample" not in chapter["text"] for chapter in chapters)


def test_image_only_pdf_is_reported_as_permanent_unsupported_input():
    document = FakeDocument([FakePage([])])
    with pytest.raises(UnsupportedPdfError, match="OCR is required") as error:
        extract_chapters_by_font_size(
            document,
            {"body_size": 11, "body_is_bold": False},
        )
    assert error.value.retryable is False


def test_long_first_sentence_never_creates_an_empty_page():
    text = " ".join(f"word-{index}" for index in range(650))
    pages = split_into_reading_pages(text, words_per_page=300)
    assert [len(page.split()) for page in pages] == [300, 300, 50]
    assert all(page.strip() for page in pages)


def test_page_indexes_remain_unique_across_segments():
    groups = group_pages([f"page-{index}" for index in range(5)], pages_per_segment=3)
    assert [[page["page_index"] for page in group] for group in groups] == [[0, 1, 2], [3, 4]]
    assert [page["is_new_segment"] for group in groups for page in group] == [True, False, False, True, False]


def test_music_selection_is_deterministic_and_ignores_missing_files(tmp_path):
    music_folder = tmp_path / "music"
    music_folder.mkdir()
    for filename in ("bright-a.wav", "bright-b.wav"):
        (music_folder / filename).write_bytes(b"audio")
    index = {
        "1": {"filename": "bright-a.wav", "genre": "joy", "bpm": 110},
        "2": {"filename": "bright-b.wav", "genre": "joy", "bpm": 120},
        "3": {"filename": "missing.wav", "genre": "joy", "bpm": 130},
    }
    index_path = tmp_path / "music_index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    mapper = MusicMapper(
        str(index_path),
        str(music_folder),
        keyword_map={"joy": ["happy"], "neutral": ["calm"]},
    )
    first = mapper.get_music("joy", "same-book-segment")
    second = mapper.get_music("joy", "same-book-segment")
    assert first == second
    assert first["filename"] in {"bright-a.wav", "bright-b.wav"}
    assert all(item["filename"] != "missing.wav" for item in mapper.genre_bucket["joy"])


def test_missing_wordnet_data_does_not_trigger_a_runtime_download(monkeypatch):
    class MissingWordNet:
        @staticmethod
        def synsets(_):
            raise LookupError("not installed")

    monkeypatch.setattr(music_mapper, "wordnet", MissingWordNet())
    result = music_mapper.build_vibe_keywords_automatically()
    assert "joy" in result
    assert "bright" in result["joy"]


def test_json_output_is_replaced_atomically(tmp_path):
    destination = tmp_path / "book.json"
    destination.write_text('{"old": true}', encoding="utf-8")
    _save_json_atomic(destination, {"new": True})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"new": True}
    assert not (tmp_path / "book.json.tmp").exists()
