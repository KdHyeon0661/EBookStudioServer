"""Persistent prompt-to-music catalog shared by the Spring worker processes."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


DEFAULT_GENERATOR_VERSION = os.environ.get(
    "MUSIC_GENERATOR_VERSION", "facebook/musicgen-small:v1"
)


def prompt_signature(
    prompt: str,
    genre: str,
    bpm: int,
    keywords: list[str],
    target_duration_sec: int,
    segment_duration_sec: int,
    generator_version: str = DEFAULT_GENERATOR_VERSION,
) -> str:
    payload = {
        "bpm": int(bpm),
        "generator_version": generator_version.strip(),
        "genre": " ".join(genre.strip().lower().split()),
        "keywords": [" ".join(value.strip().lower().split()) for value in keywords],
        "prompt": " ".join(prompt.strip().split()),
        "segment_duration_sec": int(segment_duration_sec),
        "target_duration_sec": int(target_duration_sec),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class MusicPromptCatalog:
    def __init__(self, db_path: str | os.PathLike[str], music_root: str | os.PathLike[str]):
        self.db_path = Path(db_path).resolve()
        self.music_root = Path(music_root).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.music_root.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS music_prompt_cache (
                    signature TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    genre TEXT NOT NULL,
                    bpm INTEGER NOT NULL,
                    keywords_json TEXT NOT NULL,
                    target_duration_sec INTEGER NOT NULL,
                    segment_duration_sec INTEGER NOT NULL,
                    generator_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    relative_path TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    generated_at INTEGER,
                    last_used_at INTEGER,
                    reuse_count INTEGER NOT NULL DEFAULT 0,
                    owner_job_id TEXT,
                    error TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_music_prompt_cache_status_updated "
                "ON music_prompt_cache(status, updated_at)"
            )

    def _safe_asset_path(self, relative_path: str | None) -> Path | None:
        if not relative_path:
            return None
        candidate = (self.music_root / relative_path).resolve()
        try:
            candidate.relative_to(self.music_root)
        except ValueError:
            return None
        return candidate

    def find_ready(self, signature: str) -> str | None:
        now = int(time.time())
        with self._connect() as connection:
            row = connection.execute(
                "SELECT relative_path FROM music_prompt_cache "
                "WHERE signature=? AND status='ready'", (signature,)
            ).fetchone()
            path = self._safe_asset_path(row["relative_path"] if row else None)
            if path is not None and path.is_file():
                connection.execute(
                    "UPDATE music_prompt_cache SET reuse_count=reuse_count+1, "
                    "last_used_at=?, updated_at=? WHERE signature=?",
                    (now, now, signature),
                )
                return str(path)
            if row is not None:
                connection.execute(
                    "UPDATE music_prompt_cache SET status='missing', updated_at=?, "
                    "error='Catalog file is missing' WHERE signature=?",
                    (now, signature),
                )
        return None

    def mark_generating(self, signature: str, metadata: dict[str, Any], filename: str,
                        owner_job_id: str | None) -> None:
        now = int(time.time())
        values = self._metadata_values(metadata)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO music_prompt_cache(
                    signature, prompt, genre, bpm, keywords_json, target_duration_sec,
                    segment_duration_sec, generator_version, status, filename,
                    created_at, updated_at, owner_job_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'generating', ?, ?, ?, ?)
                ON CONFLICT(signature) DO UPDATE SET
                    prompt=excluded.prompt, genre=excluded.genre, bpm=excluded.bpm,
                    keywords_json=excluded.keywords_json,
                    target_duration_sec=excluded.target_duration_sec,
                    segment_duration_sec=excluded.segment_duration_sec,
                    generator_version=excluded.generator_version,
                    status='generating', filename=excluded.filename,
                    relative_path=NULL, updated_at=excluded.updated_at,
                    owner_job_id=excluded.owner_job_id, error=NULL
                """,
                (signature, *values, filename, now, now, owner_job_id),
            )

    def mark_ready(self, signature: str, metadata: dict[str, Any], filename: str,
                   asset_path: str | os.PathLike[str], owner_job_id: str | None,
                   reused: bool = False) -> None:
        path = Path(asset_path).resolve()
        relative_path = path.relative_to(self.music_root).as_posix()
        now = int(time.time())
        values = self._metadata_values(metadata)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO music_prompt_cache(
                    signature, prompt, genre, bpm, keywords_json, target_duration_sec,
                    segment_duration_sec, generator_version, status, filename,
                    relative_path, created_at, updated_at, generated_at, last_used_at,
                    reuse_count, owner_job_id, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(signature) DO UPDATE SET
                    prompt=excluded.prompt, genre=excluded.genre, bpm=excluded.bpm,
                    keywords_json=excluded.keywords_json,
                    target_duration_sec=excluded.target_duration_sec,
                    segment_duration_sec=excluded.segment_duration_sec,
                    generator_version=excluded.generator_version, status='ready',
                    filename=excluded.filename, relative_path=excluded.relative_path,
                    updated_at=excluded.updated_at,
                    generated_at=COALESCE(music_prompt_cache.generated_at, excluded.generated_at),
                    last_used_at=excluded.last_used_at,
                    reuse_count=music_prompt_cache.reuse_count + excluded.reuse_count,
                    owner_job_id=excluded.owner_job_id, error=NULL
                """,
                (
                    signature, *values, filename, relative_path, now, now, now, now,
                    1 if reused else 0, owner_job_id,
                ),
            )

    def mark_failed(self, signature: str, error: str, cancelled: bool = False) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE music_prompt_cache SET status=?, updated_at=?, error=?, "
                "owner_job_id=NULL WHERE signature=?",
                ("cancelled" if cancelled else "failed", int(time.time()), error[:2000], signature),
            )

    @staticmethod
    def _metadata_values(metadata: dict[str, Any]) -> tuple[Any, ...]:
        return (
            metadata["prompt"], metadata["genre"], int(metadata["bpm"]),
            json.dumps(metadata["keywords"], ensure_ascii=False, separators=(",", ":")),
            int(metadata["target_duration_sec"]), int(metadata["segment_duration_sec"]),
            metadata.get("generator_version") or DEFAULT_GENERATOR_VERSION,
        )
