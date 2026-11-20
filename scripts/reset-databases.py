"""Safely reset EBookStudio PostgreSQL and SQLite service data.

The command is a dry-run unless --execute and the exact confirmation phrase are
both supplied.  It preserves schemas and Flyway history, creates restorable
backups first, and refuses to run while queue workers are live.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit


CONFIRMATION_PHRASE = "RESET EBOOKSTUDIO DATABASES"
LIVE_WINDOW_SECONDS = 120

POSTGRES_TABLES = (
    "book_music_bindings",
    "book_artifacts",
    "processing_runs",
    "usage_events",
    "books",
    "music_assets",
    "token_blocklist",
    "verification_codes",
    "request_rate_limits",
    "users",
)

QUEUE_TABLES = (
    "jobs",
    "music_prompt_cache",
    "worker_nodes",
    "account_cleanup_tasks",
)


class SafetyError(RuntimeError):
    """Raised when reset preconditions are not safe."""


def sqlite_table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def validate_queue_schema(connection: sqlite3.Connection) -> None:
    missing = [table for table in QUEUE_TABLES if not sqlite_table_exists(connection, table)]
    if missing:
        raise SafetyError(
            "The SQLite file is not an EBookStudio queue database; missing tables: "
            + ", ".join(missing)
        )


def queue_summary(
    connection: sqlite3.Connection, *, now: int | None = None
) -> dict[str, int]:
    validate_queue_schema(connection)
    current_time = int(time.time()) if now is None else now
    live_cutoff = current_time - LIVE_WINDOW_SECONDS
    summary = {
        table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        for table in QUEUE_TABLES
    }
    summary["active_status_jobs"] = int(
        connection.execute(
            "SELECT COUNT(*) FROM jobs WHERE status IN ('running', 'cancel_requested')"
        ).fetchone()[0]
    )
    summary["live_jobs"] = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM jobs
            WHERE status IN ('running', 'cancel_requested')
              AND COALESCE(heartbeat_at, started_at, 0) >= ?
            """,
            (live_cutoff,),
        ).fetchone()[0]
    )
    summary["live_workers"] = int(
        connection.execute(
            "SELECT COUNT(*) FROM worker_nodes WHERE heartbeat_at >= ?", (live_cutoff,)
        ).fetchone()[0]
    )
    return summary


def require_idle_queue(summary: dict[str, int]) -> None:
    if summary["live_jobs"] or summary["live_workers"]:
        raise SafetyError(
            "Queue workers are still active. Stop the API/workers, wait at least "
            f"{LIVE_WINDOW_SECONDS} seconds, and run the command again."
        )


def require_idle_postgres(summary: dict[str, Any]) -> None:
    if summary["other_client_sessions"]:
        raise SafetyError(
            "Other PostgreSQL client sessions are connected to the target database. "
            "Stop the API and other clients before executing the reset."
        )


def validate_queue_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if path.is_symlink():
        raise SafetyError("Refusing a symbolic-link queue database path")
    if not resolved.is_file():
        raise SafetyError(f"Queue database does not exist or is not a file: {resolved}")
    return resolved


def validate_execute_request(execute: bool, confirmation: str | None) -> None:
    if execute and confirmation != CONFIRMATION_PHRASE:
        raise SafetyError(
            "Execution requires --confirm \"" + CONFIRMATION_PHRASE + "\""
        )


def connect_postgres(postgres_url: str):
    try:
        import psycopg
    except ImportError as error:
        raise SafetyError(
            "psycopg is required; install scripts/requirements-migration.txt"
        ) from error
    return psycopg.connect(postgres_url)


def postgres_summary(connection: Any) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute("SELECT current_database(), current_user")
        database_name, database_user = cursor.fetchone()
        cursor.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema='public'
            """
        )
        existing = {row[0] for row in cursor.fetchall()}
        missing = [table for table in POSTGRES_TABLES if table not in existing]
        if missing:
            raise SafetyError(
                "The PostgreSQL target is not an EBookStudio database; missing tables: "
                + ", ".join(missing)
            )

        counts: dict[str, int] = {}
        for table in POSTGRES_TABLES:
            cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
            counts[table] = int(cursor.fetchone()[0])
        cursor.execute(
            """
            SELECT COUNT(*) FROM pg_stat_activity
            WHERE datname=current_database()
              AND pid<>pg_backend_pid()
              AND backend_type='client backend'
            """
        )
        other_client_sessions = int(cursor.fetchone()[0])
    return {
        "database": database_name,
        "user": database_user,
        "tables": counts,
        "other_client_sessions": other_client_sessions,
    }


def backup_sqlite(connection: sqlite3.Connection, destination: Path) -> None:
    if destination.exists():
        raise SafetyError(f"Backup already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    try:
        target = sqlite3.connect(temporary)
        try:
            connection.backup(target)
            result = target.execute("PRAGMA quick_check").fetchone()
            if result is None or result[0] != "ok":
                raise SafetyError("SQLite backup integrity check failed")
            target.commit()
        finally:
            target.close()
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def pg_dump_connection_settings(postgres_url: str) -> tuple[str, dict[str, str]]:
    """Remove credentials from pg_dump's visible command line.

    libpq accepts both PostgreSQL URLs and key/value connection strings. Secrets
    are returned separately so the caller can pass them through the child
    process environment instead of exposing them in the OS process list.
    """

    parsed = urlsplit(postgres_url)
    if parsed.scheme in {"postgres", "postgresql"}:
        environment: dict[str, str] = {}
        if parsed.username:
            environment["PGUSER"] = unquote(parsed.username)
        if parsed.password:
            environment["PGPASSWORD"] = unquote(parsed.password)

        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = hostname + (f":{parsed.port}" if parsed.port else "")
        safe_query: list[tuple[str, str]] = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            lowered = key.lower()
            if lowered == "password":
                environment["PGPASSWORD"] = value
            elif lowered == "user":
                environment["PGUSER"] = value
            else:
                safe_query.append((key, value))
        safe_url = urlunsplit(
            (parsed.scheme, netloc, parsed.path, urlencode(safe_query), parsed.fragment)
        )
        return safe_url, environment

    environment = {}
    safe_parts: list[str] = []
    for part in shlex.split(postgres_url):
        key, separator, value = part.partition("=")
        lowered = key.lower()
        if separator and lowered == "password":
            environment["PGPASSWORD"] = value
        elif separator and lowered == "user":
            environment["PGUSER"] = value
        else:
            safe_parts.append(part)
    return shlex.join(safe_parts), environment


def backup_postgres(
    postgres_url: str,
    destination: Path,
    *,
    runner: Callable[..., Any] = subprocess.run,
    pg_dump_executable: str | None = None,
) -> None:
    if destination.exists():
        raise SafetyError(f"Backup already exists: {destination}")
    executable = pg_dump_executable or shutil.which("pg_dump")
    if not executable:
        raise SafetyError("pg_dump was not found; no data will be deleted")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    safe_connection, credential_environment = pg_dump_connection_settings(postgres_url)
    child_environment = os.environ.copy()
    child_environment.update(credential_environment)
    try:
        runner(
            [
                executable,
                "--format=custom",
                "--no-owner",
                "--no-privileges",
                f"--file={temporary}",
                f"--dbname={safe_connection}",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            env=child_environment,
        )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise SafetyError("pg_dump did not create a valid backup file")
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def clear_queue_tables(connection: sqlite3.Connection) -> None:
    for table in QUEUE_TABLES:
        connection.execute(f'DELETE FROM "{table}"')
    if sqlite_table_exists(connection, "sqlite_sequence"):
        placeholders = ",".join("?" for _ in QUEUE_TABLES)
        connection.execute(
            f"DELETE FROM sqlite_sequence WHERE name IN ({placeholders})", QUEUE_TABLES
        )


def truncate_postgres(connection: Any) -> None:
    tables = ", ".join(f'"{table}"' for table in POSTGRES_TABLES)
    with connection.cursor() as cursor:
        cursor.execute("SET LOCAL lock_timeout = '5s'")
        cursor.execute(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE")


def reset_databases(
    postgres_connection: Any,
    queue_connection: sqlite3.Connection,
) -> None:
    """Reset both stores, rolling SQLite back if PostgreSQL fails.

    PostgreSQL commits first. If the subsequent SQLite commit fails, the backups
    produced by the CLI are the recovery source; this is the only unavoidable
    cross-database commit window.
    """

    queue_connection.execute("BEGIN IMMEDIATE")
    try:
        require_idle_queue(queue_summary(queue_connection))
        clear_queue_tables(queue_connection)
        try:
            truncate_postgres(postgres_connection)
            postgres_connection.commit()
        except Exception:
            postgres_connection.rollback()
            queue_connection.rollback()
            raise
        try:
            queue_connection.commit()
        except Exception as error:
            queue_connection.rollback()
            raise SafetyError(
                "PostgreSQL was reset but the SQLite commit failed; restore from backups"
            ) from error
    except Exception:
        if queue_connection.in_transaction:
            queue_connection.rollback()
        raise


def default_backup_directory() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("backups") / f"database-reset-{timestamp}"


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(postgres: dict[str, Any], queue: dict[str, int], queue_path: Path) -> None:
    print(f"PostgreSQL database: {postgres['database']} (user: {postgres['user']})")
    for table, count in postgres["tables"].items():
        print(f"  postgres.{table}: {count}")
    print(f"  other client sessions: {postgres['other_client_sessions']}")
    print(f"SQLite queue: {queue_path}")
    for table in QUEUE_TABLES:
        print(f"  sqlite.{table}: {queue[table]}")
    print(f"  live jobs: {queue['live_jobs']}, live workers: {queue['live_workers']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely reset EBookStudio databases while preserving schemas"
    )
    parser.add_argument(
        "--postgres-url",
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL URL (prefer the DATABASE_URL environment variable)",
    )
    parser.add_argument(
        "--queue-db",
        type=Path,
        default=os.environ.get("EBOOK_QUEUE_DB_PATH"),
        help="Path to the SQLite jobs database",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        help="New directory for PostgreSQL, SQLite, and manifest backups",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually reset data; without this option the command is a dry-run",
    )
    parser.add_argument(
        "--confirm",
        help=f'Exact execution phrase: "{CONFIRMATION_PHRASE}"',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        validate_execute_request(arguments.execute, arguments.confirm)
        if not arguments.postgres_url:
            raise SafetyError("--postgres-url or DATABASE_URL is required")
        if not arguments.queue_db:
            raise SafetyError("--queue-db or EBOOK_QUEUE_DB_PATH is required")

        queue_path = validate_queue_path(Path(arguments.queue_db))
        queue_connection = sqlite3.connect(queue_path, timeout=10)
        postgres_connection = None
        try:
            queue_before = queue_summary(queue_connection)
            postgres_connection = connect_postgres(arguments.postgres_url)
            postgres_before = postgres_summary(postgres_connection)
            postgres_connection.rollback()
            print_summary(postgres_before, queue_before, queue_path)

            if not arguments.execute:
                if queue_before["live_jobs"] or queue_before["live_workers"]:
                    print("Warning: active queue work must stop before execution.")
                if postgres_before["other_client_sessions"]:
                    print("Warning: other PostgreSQL clients must disconnect before execution.")
                print("Dry-run only: no data was changed.")
                print(
                    f'To execute, add --execute --confirm "{CONFIRMATION_PHRASE}"'
                )
                return 0

            require_idle_queue(queue_before)
            require_idle_postgres(postgres_before)

            backup_directory = (arguments.backup_dir or default_backup_directory()).resolve()
            if backup_directory.exists() and any(backup_directory.iterdir()):
                raise SafetyError(f"Backup directory is not empty: {backup_directory}")
            backup_directory.mkdir(parents=True, exist_ok=True)
            postgres_backup = backup_directory / "ebookstudio-postgres.dump"
            queue_backup = backup_directory / "jobs.db"
            manifest_path = backup_directory / "manifest.json"

            backup_postgres(arguments.postgres_url, postgres_backup)
            backup_sqlite(queue_connection, queue_backup)
            manifest: dict[str, Any] = {
                "status": "backed-up",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "postgres_database": postgres_before["database"],
                "postgres_counts_before": postgres_before["tables"],
                "queue_path": str(queue_path),
                "queue_counts_before": {
                    table: queue_before[table] for table in QUEUE_TABLES
                },
                "postgres_backup": postgres_backup.name,
                "queue_backup": queue_backup.name,
            }
            write_manifest(manifest_path, manifest)

            try:
                # Backups can take time, so repeat both liveness checks immediately before reset.
                require_idle_queue(queue_summary(queue_connection))
                postgres_pre_reset = postgres_summary(postgres_connection)
                postgres_connection.rollback()
                require_idle_postgres(postgres_pre_reset)
                reset_databases(postgres_connection, queue_connection)
                postgres_after = postgres_summary(postgres_connection)
                postgres_connection.rollback()
                queue_after = queue_summary(queue_connection)
                if any(postgres_after["tables"].values()) or any(
                    queue_after[table] for table in QUEUE_TABLES
                ):
                    raise SafetyError("Post-reset verification found remaining rows")
                manifest["status"] = "completed"
                manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
                write_manifest(manifest_path, manifest)
            except Exception as error:
                manifest["status"] = "failed"
                manifest["error"] = str(error)
                write_manifest(manifest_path, manifest)
                raise

            print(f"Database reset completed. Backups: {backup_directory}")
            return 0
        finally:
            if postgres_connection is not None:
                postgres_connection.close()
            queue_connection.close()
    except (SafetyError, sqlite3.Error, subprocess.CalledProcessError) as error:
        print(f"Safety check failed: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Database reset failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
