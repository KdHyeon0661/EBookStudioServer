"""Migrate the legacy unified SQLite database into PostgreSQL + a queue-only SQLite file."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Iterable


APP_ONLY_TABLES = (
    "users", "user", "token_blocklist", "verification_codes",
    "request_rate_limits", "usage_events", "books", "processing_runs",
    "book_artifacts", "music_assets", "book_music_bindings",
)


def table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def rows(connection: sqlite3.Connection, query: str) -> Iterable[sqlite3.Row]:
    return connection.execute(query).fetchall()


def columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}


def create_queue_copy(source: sqlite3.Connection, destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(f"queue output already exists: {destination}")
    temporary = destination.with_name(destination.name + ".migrating")
    if temporary.exists():
        temporary.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = sqlite3.connect(temporary)
    try:
        source.backup(target)
        for table in APP_ONLY_TABLES:
            if table_exists(target, table):
                target.execute(f'DROP TABLE "{table}"')
        target.commit()
    finally:
        target.close()
    return temporary


def migrate(source: sqlite3.Connection, postgres_url: str) -> dict[str, int]:
    try:
        import psycopg
    except ImportError as error:
        raise RuntimeError(
            "psycopg is required; install scripts/requirements-migration.txt"
        ) from error

    counts: dict[str, int] = {}
    user_table = "users" if table_exists(source, "users") else "user"
    if not table_exists(source, user_table):
        raise RuntimeError("legacy SQLite database has no users/user table")

    with psycopg.connect(postgres_url) as target:
        with target.cursor() as cursor:
            auth_version = "auth_version" if "auth_version" in columns(source, user_table) else "0"
            user_rows = rows(source, f"SELECT public_id, username, email, password_hash, "
                                     f"{auth_version} AS auth_version FROM {user_table}")
            for row in user_rows:
                cursor.execute("""
                    INSERT INTO users(public_id, username, email, password_hash, auth_version)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT(public_id) DO UPDATE SET
                        username=excluded.username, email=excluded.email,
                        password_hash=excluded.password_hash, auth_version=excluded.auth_version
                    """, tuple(row))
            counts["users"] = len(user_rows)

            if table_exists(source, "token_blocklist"):
                token_rows = rows(source, "SELECT jti, created_at, expires_at FROM token_blocklist")
                for row in token_rows:
                    cursor.execute("""
                        INSERT INTO token_blocklist(jti, created_at, expires_at)
                        VALUES (%s, %s, %s) ON CONFLICT(jti) DO NOTHING
                        """, tuple(row))
                counts["token_blocklist"] = len(token_rows)

            if table_exists(source, "verification_codes"):
                verification_rows = rows(source, """
                    SELECT email, code, expires_at, purpose, last_sent_at, failed_attempts
                    FROM verification_codes
                    """)
                for row in verification_rows:
                    cursor.execute("""
                        INSERT INTO verification_codes(
                            email, code, expires_at, purpose, last_sent_at, failed_attempts)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT(email) DO UPDATE SET
                            code=excluded.code, expires_at=excluded.expires_at,
                            purpose=excluded.purpose, last_sent_at=excluded.last_sent_at,
                            failed_attempts=excluded.failed_attempts
                        """, tuple(row))
                counts["verification_codes"] = len(verification_rows)

            if table_exists(source, "request_rate_limits"):
                rate_rows = rows(source, """
                    SELECT key_hash, scope, window_started_at, request_count
                    FROM request_rate_limits
                    """)
                for row in rate_rows:
                    cursor.execute("""
                        INSERT INTO request_rate_limits(
                            key_hash, scope, window_started_at, request_count)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT(key_hash) DO UPDATE SET
                            scope=excluded.scope,
                            window_started_at=excluded.window_started_at,
                            request_count=excluded.request_count
                        """, tuple(row))
                counts["request_rate_limits"] = len(rate_rows)

            if table_exists(source, "usage_events"):
                usage_rows = rows(source, """
                    SELECT user_uuid, event_id, event_type, book_id, occurred_at,
                           duration_seconds, page_turns, progress_percent, created_at
                    FROM usage_events
                    """)
                for row in usage_rows:
                    cursor.execute("""
                        INSERT INTO usage_events(
                            user_uuid, event_id, event_type, book_id, occurred_at,
                            duration_seconds, page_turns, progress_percent, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT(user_uuid, event_id) DO NOTHING
                        """, tuple(row))
                counts["usage_events"] = len(usage_rows)

            if table_exists(source, "jobs"):
                job_rows = rows(source, """
                    SELECT id, user_uuid, book_id, status, created_at,
                           COALESCE(finished_at, created_at), book_title, author,
                           cover_file, output_json, pdf_path
                    FROM jobs
                    WHERE type='analyze' AND book_id IS NOT NULL
                    """)
                for row in job_rows:
                    status = {
                        "done": "READY", "error": "FAILED", "cancelled": "CANCELLED",
                        "cancel_requested": "CANCELLING",
                    }.get(row[3], "PROCESSING")
                    source_pdf = os.path.basename(row[10]) if row[10] else None
                    cursor.execute("""
                        INSERT INTO books(
                            owner_public_id, folder, job_id, status, title, author,
                            cover_file, text_file, source_pdf, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT(owner_public_id, folder) DO UPDATE SET
                            job_id=excluded.job_id, status=excluded.status,
                            title=excluded.title, author=excluded.author,
                            cover_file=excluded.cover_file, text_file=excluded.text_file,
                            source_pdf=excluded.source_pdf, updated_at=excluded.updated_at
                        """, (
                            row[1], row[2], row[0], status, row[6], row[7], row[8], row[9],
                            source_pdf, row[4], row[5],
                        ))
                counts["books"] = len(job_rows)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True, type=Path, help="legacy users.db")
    parser.add_argument("--queue-output", required=True, type=Path, help="new jobs.db")
    parser.add_argument("--postgres-url", default=os.environ.get("DATABASE_URL"))
    arguments = parser.parse_args()
    if not arguments.postgres_url:
        parser.error("--postgres-url or DATABASE_URL is required")
    source_path = arguments.sqlite.resolve()
    output_path = arguments.queue_output.resolve()
    if source_path == output_path:
        parser.error("queue output must differ from the legacy SQLite path")
    if not source_path.is_file():
        parser.error(f"legacy SQLite file does not exist: {source_path}")

    source = sqlite3.connect(f"file:{source_path.as_posix()}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    temporary: Path | None = None
    try:
        temporary = create_queue_copy(source, output_path)
        counts = migrate(source, arguments.postgres_url)
        temporary.replace(output_path)
        temporary = None
    finally:
        source.close()
        if temporary is not None and temporary.exists():
            temporary.unlink()

    print("migration completed")
    for table, count in sorted(counts.items()):
        print(f"{table}: {count}")
    print(f"queue database: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
