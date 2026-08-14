import json

import fitz

from analyzer import process_full_book_for_offline


def test_real_pdf_is_analyzed_into_offline_reader_json(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Chapter One", fontsize=18, fontname="hebo")
    page.insert_text(
        (72, 110),
        "This is the first chapter body. It is calm and quiet. " * 20,
        fontsize=11,
        fontname="helv",
    )
    page.insert_text((72, 360), "Chapter Two", fontsize=18, fontname="hebo")
    page.insert_text(
        (72, 400),
        "This is the second chapter body. It is happy and bright. " * 20,
        fontsize=11,
        fontname="helv",
    )
    document.save(pdf_path)
    document.close()

    book_root = tmp_path / "book-1"
    music_root = tmp_path / "defaults" / "music"
    music_root.mkdir(parents=True)
    (music_root / "default_ambient.wav").write_bytes(b"not-decoded-by-analyzer")
    (music_root.parent / "music_index.json").write_text(
        json.dumps({
            "1": {
                "filename": "default_ambient.wav",
                "genre": "neutral",
                "bpm": 80,
                "is_custom": False,
            }
        }),
        encoding="utf-8",
    )

    result = process_full_book_for_offline(
        str(pdf_path),
        str(book_root),
        str(music_root),
        "/files/user/book-1",
    )

    output = json.loads((book_root / result["text_file"]).read_text(encoding="utf-8"))
    assert output["book_info"]["title"] == "sample"
    assert len(output["chapters"]) == 2
    assert [chapter["title"] for chapter in output["chapters"]] == ["Chapter One", "Chapter Two"]
    assert output["chapters"][0]["segments"][0]["pages"][0]["page_index"] == 0
    assert output["chapters"][0]["segments"][0]["music_source"] == "preset"
    assert (book_root / result["cover_image"]).is_file()
    assert not list(book_root.glob("*.tmp*"))
