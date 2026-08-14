import json

from book_music_metadata import attach_segment_music, save_book_json_atomic


def test_generated_music_replaces_fallback_and_removes_hint(tmp_path):
    data = {
        "chapters": [{"segments": [{
            "music_filename": "default_ambient.wav",
            "music_path": "music/default_ambient.wav",
            "music_source": "system_default",
            "bpm": 80,
            "generation_hint": {"target_emotion": "joy", "keywords": ["bright"]},
        }]}]
    }
    attach_segment_music(data, 0, 0, "generated.wav", 124, "ai_generated")
    path = tmp_path / "book.json"
    save_book_json_atomic(str(path), data)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    segment = loaded["chapters"][0]["segments"][0]
    assert segment["music_filename"] == "generated.wav"
    assert segment["music_path"] == "music/generated.wav"
    assert segment["music_source"] == "ai_generated"
    assert segment["bpm"] == 124
    assert "generation_hint" not in segment
    assert not (tmp_path / "book.json.tmp").exists()