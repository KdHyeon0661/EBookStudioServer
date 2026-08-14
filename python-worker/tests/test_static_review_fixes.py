import json

import fitz

import spring_worker
from music_mapper import MusicMapper, content_signature
from pdf_structure import (
    BOLD_FLAG,
    PermanentAnalysisError,
    analyze_font_characteristics,
    extract_chapters_by_font_size,
    find_repeated_margin_texts,
)


def span(text, *, size=11, flags=0):
    return {"text": text, "size": size, "flags": flags}


def line(*spans, y=100):
    return {"bbox": (40, y, 560, y + 20), "spans": list(spans)}


class FakePage:
    def __init__(self, lines, height=800):
        self.lines = lines
        self.rect = type("Rect", (), {"height": height})()

    def get_text(self, kind=None, sort=False):
        if kind == "dict":
            return {"blocks": [{"bbox": (40, 0, 560, 800), "lines": self.lines}]}
        return "\n".join(
            " ".join(str(item.get("text") or "") for item in value["spans"])
            for value in self.lines
        )


class FakeDocument(list):
    metadata = {}


def test_repeated_header_text_is_only_removed_from_the_margin():
    document = FakeDocument([
        FakePage([
            line(span("Shared Book Title", size=9), y=10),
            line(span("Chapter One", size=17, flags=BOLD_FLAG), y=80),
            line(span("Shared Book Title"), y=200),
        ]),
        FakePage([
            line(span("Shared Book Title", size=9), y=10),
            line(span("The second body line remains readable."), y=200),
        ]),
        FakePage([
            line(span("Shared Book Title", size=9), y=10),
            line(span("The third body line remains readable."), y=200),
        ]),
    ])
    repeated = find_repeated_margin_texts(document)
    font_info = analyze_font_characteristics(document, repeated_margin_texts=repeated)
    chapters = extract_chapters_by_font_size(document, font_info, repeated_margin_texts=repeated)
    body = " ".join(chapter["text"] for chapter in chapters)
    assert "Shared Book Title" in body
    assert body.count("Shared Book Title") == 1


def test_segment_analysis_uses_content_keywords_and_normalized_signature(tmp_path):
    index_path = tmp_path / "music_index.json"
    index_path.write_text(json.dumps({}), encoding="utf-8")
    mapper = MusicMapper(str(index_path), keyword_map={"joy": ["happy"], "neutral": ["calm"]})
    analysis = mapper.analyze_segment(
        "The lighthouse storm returned. The lighthouse keeper felt happy."
    )
    assert analysis["emotion"] == "joy"
    assert "lighthouse" in analysis["keywords"]
    assert analysis["content_signature"] == content_signature(
        "  THE lighthouse storm returned. The lighthouse keeper felt happy.  "
    )


def test_password_protected_pdf_is_a_permanent_input_error(tmp_path):
    from analyzer import process_full_book_for_offline

    source = fitz.open()
    page = source.new_page()
    page.insert_text((72, 72), "Protected book body")
    pdf_path = tmp_path / "protected.pdf"
    source.save(
        pdf_path,
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="reader-secret",
    )
    source.close()

    try:
        process_full_book_for_offline(
            str(pdf_path),
            str(tmp_path / "book"),
            str(tmp_path / "music"),
            "/files/user/book",
        )
    except PermanentAnalysisError as error:
        assert error.retryable is False
        assert "Password-protected" in str(error)
    else:
        raise AssertionError("Encrypted PDF was accepted without a password")


def test_terminal_analysis_failure_removes_only_the_validated_book_folder(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    book_root = storage / "users" / "user-1" / "book-1"
    book_root.mkdir(parents=True)
    (book_root / "partial.png").write_bytes(b"partial")
    monkeypatch.setattr(spring_worker, "STORAGE_ROOT", storage)
    monkeypatch.setattr(spring_worker, "DB_PATH", storage / "jobs.db")
    spring_worker.ensure_schema()
    with spring_worker.connect() as connection:
        connection.execute(
            "INSERT INTO jobs(id, type, user_uuid, book_id, status, created_at, worker_id, "
            "attempt_count, max_attempts, book_root_folder) "
            "VALUES ('failed-analysis', 'analyze', 'user-1', 'book-1', 'running', 1, "
            "'worker-1', 1, 3, ?)",
            (str(book_root),),
        )

    spring_worker.fail_job(
        {
            "id": "failed-analysis",
            "type": "analyze",
            "attempt_count": 1,
            "max_attempts": 3,
            "book_root_folder": str(book_root),
        },
        "worker-1",
        PermanentAnalysisError("unsupported input"),
    )

    assert not book_root.exists()
    assert (storage / "users" / "user-1").is_dir()
    with spring_worker.connect() as connection:
        assert connection.execute(
            "SELECT status FROM jobs WHERE id='failed-analysis'"
        ).fetchone()[0] == "error"
