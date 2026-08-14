from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from nltk.corpus import wordnet
except ImportError:
    wordnet = None


LOGGER = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z'-]{2,}|[가-힣]{2,}")
STOP_WORDS = {
    "about", "after", "again", "also", "another", "because", "before", "being",
    "between", "book", "chapter", "could", "does", "each", "from", "have", "into",
    "more", "most", "only", "other", "over", "same", "some", "such", "than", "that",
    "their", "them", "then", "there", "these", "they", "this", "through", "under",
    "very", "what", "when", "where", "which", "while", "with", "would", "your",
    "그리고", "그러나", "그런데", "대한", "때문", "또는", "매우", "에서", "으로",
    "이다", "있는", "있다", "한다", "하는", "했다", "하지만", "위해", "통해",
}

FALLBACK_KEYWORDS = {
    "joy": ["happy", "smile", "fun", "excited", "delight"],
    "sadness": ["sad", "cry", "tear", "grief", "depression"],
    "fear": ["scary", "dark", "ghost", "horror", "afraid"],
    "anger": ["angry", "mad", "rage", "furious"],
    "neutral": ["normal", "calm", "quiet", "silent"],
    "surprise": ["shock", "amazed", "wow", "sudden"],
}

EMOTION_TO_CATEGORIES = {
    "anger": ["industrial", "noise", "battle", "rock", "trap", "hiphop", "techno", "metal"],
    "disgust": ["industrial", "noise", "horror", "drone"],
    "fear": ["horror", "noir", "drone", "ambient", "suspense"],
    "joy": ["pop", "edm", "house", "funk", "samba", "reggae", "trance", "groove", "disco"],
    "neutral": ["minimal", "ambient", "lofi", "jazz", "acoustic", "drone", "calm"],
    "sadness": ["ambient", "lofi", "noir", "acoustic", "romantic", "impressionist", "blues"],
    "surprise": ["experimental", "vaporwave", "noise", "industrial", "fusion", "synthwave"],
}

EMOTION_ALIASES = {
    "peace": "neutral",
    "love": "romantic",
    "hope": "ambient",
    "curiosity": "experimental",
    "confusion": "experimental",
    "courage": "battle",
    "pride": "cinematic",
    "excitement": "joy",
}


def normalize_content(text: str) -> str:
    return " ".join((text or "").lower().split())


def content_signature(text: str) -> str:
    return hashlib.sha256(normalize_content(text).encode("utf-8")).hexdigest()


def build_vibe_keywords_automatically() -> dict[str, list[str]]:
    try:
        from defaults.emotions_20 import EMOTIONS_20
    except (ImportError, AttributeError):
        return {key: list(values) for key, values in FALLBACK_KEYWORDS.items()}

    keyword_map: dict[str, list[str]] = {}
    for key, definition in EMOTIONS_20.items():
        synonyms = {key.lower()}
        if wordnet is not None:
            try:
                for synonym_set in wordnet.synsets(key):
                    for lemma in synonym_set.lemmas():
                        synonyms.add(lemma.name().replace("_", " ").lower())
            except LookupError:
                # Images install WordNet at build time. Offline local runs use style terms.
                pass
        style = str(definition.get("style") or "")
        synonyms.update(value.strip().lower() for value in style.split(",") if value.strip())
        keyword_map[key.lower()] = sorted(synonyms)
    return keyword_map or {"neutral": ["neutral"]}


@lru_cache(maxsize=1)
def get_vibe_keywords() -> dict[str, list[str]]:
    return build_vibe_keywords_automatically()


class MusicMapper:
    def __init__(
        self,
        index_file: str,
        music_folder: str | None = None,
        keyword_map: dict[str, list[str]] | None = None,
    ):
        self.genre_bucket: dict[str, list[dict[str, Any]]] = {}
        self.index_file = index_file
        self.music_folder = Path(music_folder).resolve() if music_folder else None
        self.keyword_map = keyword_map or get_vibe_keywords()
        self._keyword_sets = {vibe: set(keywords) for vibe, keywords in self.keyword_map.items()}
        self._available_filenames = self._scan_available_filenames()
        self._load_index()

    def _scan_available_filenames(self) -> set[str] | None:
        if self.music_folder is None or not self.music_folder.is_dir():
            return None
        return {path.name for path in self.music_folder.rglob("*") if path.is_file()}

    def _load_index(self) -> None:
        if not os.path.isfile(self.index_file):
            return
        try:
            with open(self.index_file, "r", encoding="utf-8") as source:
                library = json.load(source)
        except (OSError, json.JSONDecodeError) as error:
            LOGGER.warning("Music index could not be loaded: %s", error)
            return
        if not isinstance(library, dict):
            LOGGER.warning("Music index root must be an object: %s", self.index_file)
            return
        for info in library.values():
            if not isinstance(info, dict):
                continue
            filename = str(info.get("filename") or "").strip()
            if not filename:
                continue
            if self._available_filenames is not None and filename not in self._available_filenames:
                continue
            vibe = str(info.get("genre") or info.get("vibe") or "unknown").lower()
            self.genre_bucket.setdefault(vibe, []).append(info)
        for values in self.genre_bucket.values():
            values.sort(key=lambda info: str(info.get("filename") or ""))

    def analyze_vibe(self, text: str) -> str:
        normalized = normalize_content(text)
        if not normalized:
            return "neutral"
        token_counts = Counter(token.lower() for token in TOKEN_PATTERN.findall(normalized))
        scores: dict[str, int] = {}
        for vibe, keywords in self._keyword_sets.items():
            score = 0
            for keyword in keywords:
                if " " in keyword:
                    score += normalized.count(keyword)
                else:
                    score += token_counts[keyword]
            scores[vibe] = score
        if not any(scores.values()):
            return "neutral"
        return max(scores, key=lambda vibe: (scores[vibe], vibe))

    def extract_keywords(self, text: str, limit: int = 5) -> list[str]:
        if limit <= 0:
            return []
        tokens = [token.lower() for token in TOKEN_PATTERN.findall(text or "")]
        filtered = [token for token in tokens if token not in STOP_WORDS and not token.isdigit()]
        if not filtered:
            return []
        counts = Counter(filtered)
        first_position: dict[str, int] = {}
        for index, token in enumerate(filtered):
            first_position.setdefault(token, index)
        ranked = sorted(counts, key=lambda token: (-counts[token], first_position[token], token))
        return ranked[:limit]

    def analyze_segment(self, text: str) -> dict[str, Any]:
        vibe = self.analyze_vibe(text)
        keywords = self.extract_keywords(text)
        if not keywords:
            keywords = list(self.keyword_map.get(vibe, []))[:5]
        return {
            "emotion": vibe,
            "keywords": keywords,
            "content_signature": content_signature(text),
        }

    @staticmethod
    def _pick(candidates: list[dict[str, Any]], selection_key: str) -> dict[str, Any] | None:
        if not candidates:
            return None
        digest = hashlib.sha256(selection_key.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % len(candidates)
        return candidates[index]

    def get_music(self, vibe: str, selection_key: str = "") -> dict[str, Any] | None:
        normalized_vibe = (vibe or "neutral").lower()
        key = selection_key or normalized_vibe
        direct = self._pick(self.genre_bucket.get(normalized_vibe, []), f"{key}:{normalized_vibe}")
        if direct:
            return direct
        for category in EMOTION_TO_CATEGORIES.get(normalized_vibe, []):
            selected = self._pick(self.genre_bucket.get(category, []), f"{key}:{category}")
            if selected:
                return selected
        alias = EMOTION_ALIASES.get(normalized_vibe)
        if alias:
            selected = self._pick(self.genre_bucket.get(alias, []), f"{key}:{alias}")
            if selected:
                return selected
            for category in EMOTION_TO_CATEGORIES.get(alias, []):
                selected = self._pick(self.genre_bucket.get(category, []), f"{key}:{category}")
                if selected:
                    return selected
        return None
