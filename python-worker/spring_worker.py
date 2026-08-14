"""Spring Boot API가 등록한 PDF 분석 및 MusicGen 작업을 처리하는 독립 워커."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import sqlite3
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DB_PATH = Path(os.environ.get("EBOOK_DB_PATH", PROJECT_ROOT / "users.db")).resolve()
STORAGE_ROOT = Path(os.environ.get("EBOOK_STORAGE_ROOT", PROJECT_ROOT)).resolve()
DEFAULT_MUSIC_FOLDER = STORAGE_ROOT / "defaults" / "music"
BUNDLED_DEFAULTS = BASE_DIR / "defaults"
STALE_JOB_SECONDS = int(os.environ.get("JOB_STALE_SECONDS", "900"))
HEARTBEAT_SECONDS = max(5, int(os.environ.get("JOB_HEARTBEAT_SECONDS", "30")))
MAX_ATTEMPTS = max(1, int(os.environ.get("JOB_MAX_ATTEMPTS", "3")))


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def ensure_storage_assets() -> None:
    target_defaults = STORAGE_ROOT / "defaults"
    target_defaults.mkdir(parents=True, exist_ok=True)
    DEFAULT_MUSIC_FOLDER.mkdir(parents=True, exist_ok=True)
    for relative in (
        "default.png", "emotions_20.py", "genre_bpm_connector.py", "music_genres_200.py",
        "music_index.json", "music/default_ambient.wav",
    ):
        source = BUNDLED_DEFAULTS / relative
        target = target_defaults / relative
        if source.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def ensure_schema() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, type TEXT NOT NULL, user_uuid TEXT NOT NULL,
                book_id TEXT, status TEXT NOT NULL DEFAULT 'queued', created_at INTEGER NOT NULL,
                started_at INTEGER, finished_at INTEGER, error TEXT, json_path TEXT,
                music_folder TEXT, web_path_prefix TEXT, pdf_path TEXT, book_root_folder TEXT,
                output_json TEXT, cover_file TEXT, book_title TEXT, author TEXT,
                parent_job_id TEXT, music_job_id TEXT, worker_id TEXT, heartbeat_at INTEGER,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3, available_at INTEGER,
                cancel_requested_at INTEGER
            )
            """
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
        additions = {
            "output_json": "TEXT", "cover_file": "TEXT", "book_title": "TEXT",
            "author": "TEXT", "parent_job_id": "TEXT", "music_job_id": "TEXT",
            "worker_id": "TEXT", "heartbeat_at": "INTEGER",
            "attempt_count": "INTEGER NOT NULL DEFAULT 0",
            "max_attempts": f"INTEGER NOT NULL DEFAULT {MAX_ATTEMPTS}", "available_at": "INTEGER",
            "cancel_requested_at": "INTEGER",
        }
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} {definition}")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_status_type_created "
            "ON jobs(status, type, created_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_parent_job_id ON jobs(parent_job_id)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS worker_nodes (
                id TEXT PRIMARY KEY, job_type TEXT NOT NULL, pid INTEGER NOT NULL,
                started_at INTEGER NOT NULL, heartbeat_at INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS music_prompt_cache (
                signature TEXT PRIMARY KEY, prompt TEXT NOT NULL, genre TEXT NOT NULL,
                bpm INTEGER NOT NULL, keywords_json TEXT NOT NULL,
                target_duration_sec INTEGER NOT NULL, segment_duration_sec INTEGER NOT NULL,
                generator_version TEXT NOT NULL, status TEXT NOT NULL, filename TEXT NOT NULL,
                relative_path TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
                generated_at INTEGER, last_used_at INTEGER,
                reuse_count INTEGER NOT NULL DEFAULT 0, owner_job_id TEXT, error TEXT
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_music_prompt_cache_status_updated "
            "ON music_prompt_cache(status, updated_at)"
        )


def recover_stuck_jobs(stale_seconds: int = STALE_JOB_SECONDS) -> int:
    now = int(time.time())
    cutoff = now - stale_seconds
    cancelled_jobs: list[dict[str, Any]] = []
    terminal_analysis_jobs: list[dict[str, Any]] = []
    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        terminal_analysis_jobs = [dict(row) for row in connection.execute(
            """
            SELECT * FROM jobs WHERE type='analyze' AND status='running'
              AND COALESCE(heartbeat_at, started_at, 0) < ?
              AND attempt_count >= max_attempts
            """, (cutoff,)
        )]
        retryable = connection.execute(
            """
            UPDATE jobs SET status='queued', started_at=NULL, worker_id=NULL, heartbeat_at=NULL,
                available_at=?, error='Worker heartbeat expired; retrying'
            WHERE status='running' AND COALESCE(heartbeat_at, started_at, 0) < ?
              AND attempt_count < max_attempts
            """, (now + 5, cutoff)
        ).rowcount
        failed = connection.execute(
            """
            UPDATE jobs SET status='error', finished_at=?, worker_id=NULL,
                heartbeat_at=NULL, error='Worker heartbeat expired and retry limit was reached'
            WHERE status='running' AND COALESCE(heartbeat_at, started_at, 0) < ?
              AND attempt_count >= max_attempts
            """, (now, cutoff)
        ).rowcount
        for job in terminal_analysis_jobs:
            connection.execute(
                """
                UPDATE jobs SET status='cancelled', finished_at=?, cancel_requested_at=?,
                    available_at=NULL, error='Parent analysis failed'
                WHERE parent_job_id=? AND status='queued'
                """, (now, now, job["id"])
            )
            connection.execute(
                """
                UPDATE jobs SET status='cancel_requested', cancel_requested_at=?
                WHERE parent_job_id=? AND status='running'
                """, (now, job["id"])
            )
        cancelled_jobs = [dict(row) for row in connection.execute(
            """
            SELECT * FROM jobs
            WHERE status='cancel_requested' AND COALESCE(heartbeat_at, started_at, 0) < ?
            """, (cutoff,)
        )]
        for job in cancelled_jobs:
            connection.execute(
                """
                UPDATE jobs SET status='cancelled', finished_at=?, cancel_requested_at=?,
                    available_at=NULL, error=NULL
                WHERE parent_job_id=? AND status='queued'
                """, (now, now, job["id"])
            )
            connection.execute(
                """
                UPDATE jobs SET status='cancel_requested', cancel_requested_at=?
                WHERE parent_job_id=? AND status='running'
                """, (now, job["id"])
            )
        cancelled = connection.execute(
            """
            UPDATE jobs SET status='cancelled', finished_at=?, worker_id=NULL, heartbeat_at=NULL,
                available_at=NULL, error=NULL
            WHERE status='cancel_requested' AND COALESCE(heartbeat_at, started_at, 0) < ?
            """, (now, cutoff)
        ).rowcount
        connection.execute("COMMIT")
    for job in cancelled_jobs + terminal_analysis_jobs:
        cleanup_analysis_artifacts(job)
    return retryable + failed + cancelled


def claim_job(job_type: str, worker_id: str | None = None) -> dict[str, Any] | None:
    worker_id = worker_id or f"manual:{os.getpid()}"
    now = int(time.time())
    connection = connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT id FROM jobs
            WHERE status='queued' AND type=? AND attempt_count < max_attempts
              AND COALESCE(available_at, 0) <= ?
            ORDER BY created_at ASC LIMIT 1
            """, (job_type, now)
        ).fetchone()
        if row is None:
            connection.execute("COMMIT")
            return None
        updated = connection.execute(
            """
            UPDATE jobs SET status='running', started_at=?, heartbeat_at=?, worker_id=?,
                attempt_count=attempt_count+1, error=NULL
            WHERE id=? AND status='queued'
            """, (now, now, worker_id, row["id"])
        ).rowcount
        claimed = connection.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()
        connection.execute("COMMIT")
        return dict(claimed) if updated == 1 and claimed is not None else None
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


class JobCancelledError(RuntimeError):
    pass


def cancellation_requested(job_id: str, worker_id: str | None = None) -> bool:
    query = "SELECT status FROM jobs WHERE id=?"
    parameters: tuple[Any, ...] = (job_id,)
    if worker_id is not None:
        query += " AND worker_id=?"
        parameters = (job_id, worker_id)
    with connect() as connection:
        row = connection.execute(query, parameters).fetchone()
    return row is not None and row["status"] in {"cancel_requested", "cancelled"}


def raise_if_cancelled(job_id: str, worker_id: str) -> None:
    if cancellation_requested(job_id, worker_id):
        raise JobCancelledError("Job cancellation requested")


def mark_job_cancelled(job_id: str, worker_id: str | None = None) -> bool:
    now = int(time.time())
    worker_clause = "" if worker_id is None else " AND worker_id=?"
    parameters: tuple[Any, ...] = (now, job_id) if worker_id is None else (now, job_id, worker_id)
    with connect() as connection:
        updated = connection.execute(
            f"""
            UPDATE jobs SET status='cancelled', finished_at=?, worker_id=NULL,
                heartbeat_at=NULL, available_at=NULL, error=NULL
            WHERE id=? AND status='cancel_requested'{worker_clause}
            """, parameters
        ).rowcount
        connection.execute(
            """
            UPDATE jobs SET status='cancelled', finished_at=?, cancel_requested_at=?,
                available_at=NULL, error=NULL
            WHERE parent_job_id=? AND status='queued'
            """, (now, now, job_id)
        )
        connection.execute(
            """
            UPDATE jobs SET status='cancel_requested', cancel_requested_at=?
            WHERE parent_job_id=? AND status='running'
            """, (now, job_id)
        )
    return updated == 1


def cleanup_analysis_artifacts(job: dict[str, Any]) -> None:
    if job.get("type") != "analyze" or not job.get("book_root_folder"):
        return
    try:
        users_root = (STORAGE_ROOT / "users").resolve()
        target = Path(job["book_root_folder"]).resolve()
        relative = target.relative_to(users_root)
        if len(relative.parts) < 2:
            raise ValueError(f"Refusing to delete non-book path: {target}")
        if target.is_dir():
            shutil.rmtree(target)
    except Exception:
        traceback.print_exc()


cleanup_cancelled_analysis = cleanup_analysis_artifacts


def heartbeat(job_id: str, worker_id: str) -> None:
    with connect() as connection:
        connection.execute(
            """
            UPDATE jobs SET heartbeat_at=?
            WHERE id=? AND worker_id=? AND status IN ('running', 'cancel_requested')
            """,
            (int(time.time()), job_id, worker_id),
        )


class JobHeartbeat:
    def __init__(self, job_id: str, worker_id: str):
        self.job_id = job_id
        self.worker_id = worker_id
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        heartbeat(self.job_id, self.worker_id)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop_event.set()
        self.thread.join(timeout=2)

    def _run(self) -> None:
        while not self.stop_event.wait(HEARTBEAT_SECONDS):
            try:
                heartbeat(self.job_id, self.worker_id)
            except Exception:
                traceback.print_exc()


def finish_job(job_id: str, *, status: str, error: str | None = None,
               result: dict[str, Any] | None = None, worker_id: str | None = None) -> bool:
    result = result or {}
    where = "id=? AND status='running'" if worker_id is None \
        else "id=? AND worker_id=? AND status='running'"
    parameters: list[Any] = [
        status, int(time.time()), error, result.get("text_file"), result.get("cover_image"),
        result.get("title"), result.get("real_author"), result.get("music_job_id"), job_id,
    ]
    if worker_id is not None:
        parameters.append(worker_id)
    with connect() as connection:
        updated = connection.execute(
            f"""
            UPDATE jobs SET status=?, finished_at=?, error=?, output_json=?, cover_file=?,
                book_title=?, author=?, music_job_id=?, heartbeat_at=NULL, worker_id=NULL
            WHERE {where}
            """, parameters
        ).rowcount
    if updated == 0 and cancellation_requested(job_id):
        mark_job_cancelled(job_id, worker_id)
    return updated == 1


def fail_job(job: dict[str, Any], worker_id: str, error: Exception) -> None:
    if cancellation_requested(job["id"], worker_id):
        mark_job_cancelled(job["id"], worker_id)
        cleanup_analysis_artifacts(job)
        return
    attempt = int(job.get("attempt_count") or 1)
    maximum = int(job.get("max_attempts") or MAX_ATTEMPTS)
    retryable = bool(getattr(error, "retryable", True))
    terminal = not retryable or attempt >= maximum
    now = int(time.time())
    message = str(error)[:2000]
    with connect() as connection:
        if not terminal:
            delay = min(300, 5 * (2 ** max(0, attempt - 1)))
            connection.execute(
                """
                UPDATE jobs SET status='queued', started_at=NULL, heartbeat_at=NULL, worker_id=NULL,
                    available_at=?, error=? WHERE id=? AND worker_id=?
                """, (now + delay, message, job["id"], worker_id)
            )
        else:
            connection.execute(
                """
                UPDATE jobs SET status='error', finished_at=?, heartbeat_at=NULL, worker_id=NULL,
                    error=? WHERE id=? AND worker_id=?
                """, (now, message, job["id"], worker_id)
            )
            if job.get("type") == "analyze":
                connection.execute(
                    """
                    UPDATE jobs SET status='cancelled', finished_at=?, cancel_requested_at=?,
                        available_at=NULL, error='Parent analysis failed'
                    WHERE parent_job_id=? AND status='queued'
                    """, (now, now, job["id"])
                )
                connection.execute(
                    """
                    UPDATE jobs SET status='cancel_requested', cancel_requested_at=?
                    WHERE parent_job_id=? AND status='running'
                    """, (now, job["id"])
                )
    if terminal:
        cleanup_analysis_artifacts(job)


def enqueue_music_generation(source_job: dict[str, Any], json_path: str) -> str:
    job_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"ebookstudio:{source_job['id']}:music"))
    with connect() as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO jobs(
                id, type, user_uuid, book_id, status, created_at, json_path,
                music_folder, web_path_prefix, parent_job_id, max_attempts
            ) VALUES (?, 'music_generation', ?, ?, 'queued', ?, ?, ?, ?, ?, ?)
            """, (
                job_id, source_job["user_uuid"], source_job["book_id"], int(time.time()),
                json_path, source_job.get("music_folder") or str(DEFAULT_MUSIC_FOLDER),
                source_job.get("web_path_prefix"), source_job["id"], MAX_ATTEMPTS,
            )
        )
    return job_id


def finish_analysis_with_music_job(
    job: dict[str, Any],
    worker_id: str,
    json_path: str,
    result: dict[str, Any],
) -> bool:
    music_job_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"ebookstudio:{job['id']}:music"))
    now = int(time.time())
    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        updated = connection.execute(
            """
            UPDATE jobs SET status='done', finished_at=?, error=NULL, output_json=?,
                cover_file=?, book_title=?, author=?, music_job_id=?, heartbeat_at=NULL,
                worker_id=NULL
            WHERE id=? AND worker_id=? AND status='running'
            """,
            (
                now, result.get("text_file"), result.get("cover_image"),
                result.get("title"), result.get("real_author"), music_job_id,
                job["id"], worker_id,
            ),
        ).rowcount
        if updated != 1:
            connection.execute("ROLLBACK")
            if cancellation_requested(job["id"], worker_id):
                mark_job_cancelled(job["id"], worker_id)
            return False
        connection.execute(
            """
            INSERT OR IGNORE INTO jobs(
                id, type, user_uuid, book_id, status, created_at, json_path,
                music_folder, web_path_prefix, parent_job_id, max_attempts
            ) VALUES (?, 'music_generation', ?, ?, 'queued', ?, ?, ?, ?, ?, ?)
            """,
            (
                music_job_id, job["user_uuid"], job["book_id"], now, json_path,
                job.get("music_folder") or str(DEFAULT_MUSIC_FOLDER),
                job.get("web_path_prefix"), job["id"], MAX_ATTEMPTS,
            ),
        )
        connection.execute("COMMIT")
    result["music_job_id"] = music_job_id
    return True


def process_analysis(job: dict[str, Any], worker_id: str) -> None:
    from analyzer import process_full_book_for_offline

    raise_if_cancelled(job["id"], worker_id)
    result = process_full_book_for_offline(
        pdf_path=job["pdf_path"], book_root_folder=job["book_root_folder"],
        music_folder=job.get("music_folder") or str(DEFAULT_MUSIC_FOLDER),
        web_path_prefix=job["web_path_prefix"],
    )
    raise_if_cancelled(job["id"], worker_id)
    if not result or not result.get("text_file"):
        raise RuntimeError("Analyzer did not return an output JSON file")
    json_path = str(Path(job["book_root_folder"]) / result["text_file"])
    raise_if_cancelled(job["id"], worker_id)
    if not finish_analysis_with_music_job(job, worker_id, json_path, result):
        cleanup_analysis_artifacts(job)


def process_music_generation(job: dict[str, Any], worker_id: str) -> None:
    from background_music_jobs import process_book_background
    from indexer import create_music_index

    raise_if_cancelled(job["id"], worker_id)
    summary = process_book_background(
        job["json_path"], job.get("music_folder") or str(DEFAULT_MUSIC_FOLDER),
        job.get("web_path_prefix"), job.get("user_uuid"), job.get("book_id"),
        should_cancel=lambda: cancellation_requested(job["id"], worker_id),
        catalog_db_path=str(DB_PATH), job_id=job["id"],
    )
    raise_if_cancelled(job["id"], worker_id)
    create_music_index(job.get("music_folder") or str(DEFAULT_MUSIC_FOLDER))
    raise_if_cancelled(job["id"], worker_id)
    finish_job(job["id"], status="done", result=summary, worker_id=worker_id)


def update_worker_node(worker_id: str, job_type: str) -> None:
    now = int(time.time())
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO worker_nodes(id, job_type, pid, started_at, heartbeat_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET heartbeat_at=excluded.heartbeat_at
            """, (worker_id, job_type, os.getpid(), now, now)
        )


def run_loop(job_type: str, idle_seconds: float) -> None:
    worker_id = f"{job_type}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    print(f"[SpringWorker:{job_type}] started id={worker_id} db={DB_PATH}")
    last_recovery = 0.0
    while True:
        job = None
        try:
            update_worker_node(worker_id, job_type)
            if time.monotonic() - last_recovery > 30:
                recover_stuck_jobs()
                last_recovery = time.monotonic()
            job = claim_job(job_type, worker_id)
            if job is None:
                time.sleep(idle_seconds)
                continue
            with JobHeartbeat(job["id"], worker_id):
                if job_type == "analyze":
                    process_analysis(job, worker_id)
                else:
                    process_music_generation(job, worker_id)
        except KeyboardInterrupt:
            return
        except JobCancelledError:
            if job is not None:
                mark_job_cancelled(job["id"], worker_id)
                cleanup_cancelled_analysis(job)
        except Exception as error:
            traceback.print_exc()
            if job is not None:
                fail_job(job, worker_id, error)
            time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="EBookStudio background worker")
    parser.add_argument("--role", choices=("all", "analyze", "music_generation"))
    arguments = parser.parse_args()
    role = arguments.role or os.environ.get("WORKER_ROLE", "all").strip().lower()
    if role not in {"all", "analyze", "music_generation"}:
        parser.error("WORKER_ROLE must be all, analyze, or music_generation")

    ensure_schema()
    ensure_storage_assets()
    recover_stuck_jobs()
    if role == "analyze":
        run_loop("analyze", 1.0)
        return
    if role == "music_generation":
        run_loop("music_generation", 2.0)
        return

    analysis = multiprocessing.Process(target=run_loop, args=("analyze", 1.0), name="Analyzer")
    generation = multiprocessing.Process(target=run_loop, args=("music_generation", 2.0), name="Generator")
    analysis.start()
    generation.start()
    analysis.join()
    generation.join()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()