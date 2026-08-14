"""Pure helpers for applying generated music to a book JSON document."""

from __future__ import annotations

import json
import os
from typing import Any


def attach_segment_music(data: dict[str, Any], chapter_index: int, segment_index: int,
                         filename: str, bpm: int, source: str) -> None:
    segment = data["chapters"][chapter_index]["segments"][segment_index]
    segment["music_filename"] = filename
    segment["music_path"] = f"music/{filename}"
    segment["music_source"] = source
    segment["bpm"] = int(bpm)
    segment.pop("generation_hint", None)


def save_book_json_atomic(json_path: str, data: dict[str, Any]) -> None:
    temp_path = json_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2)
    os.replace(temp_path, json_path)